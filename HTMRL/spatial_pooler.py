import math

import numpy as np
import statistics as stats
from collections import deque

import HTMRL.log as log


class SpatialPooler:
    def __init__(self, input_size, acts_n, boost_strength=1.0, reward_scaled_reinf=True, boost_scaled_reinf=False,
                 only_reinforce_selected=True, normalize_rewards=True, cell_count=2048, active_count=40, boost_until=0,
                 reward_window=1000):

        self.input_size = input_size
        self.input_size_flat = np.prod(input_size)

        self.i = 0
        self.size = max(2, math.floor(cell_count / acts_n)) * acts_n
        self.stimulus_thresh = 0  # not implemented
        self.active_columns_count = active_count
        self.init_synapse_count = min(5 * self.active_columns_count,
                                      int(self.input_size_flat * 0.5))
        self.init_synapse_count = max(1, self.init_synapse_count)
        self.connected_perm_thresh = 0.5

        self.perm_inc_step = 0.15
        self.perm_dec_step = 0.02
        self.perm_min = 0.01
        self.perm_max = 1.01

        self.acts_n = acts_n
        # print(self.size, self.acts_n)
        assert (self.size % self.acts_n == 0)
        self.cells_per_act = int(self.size / self.acts_n)

        self.active_duty_cycles = np.zeros(self.size)

        self.boost_strength = boost_strength * 2.0
        self.boost_factors = np.ones(self.size, dtype=np.float32)
        self.boost_anneal_until = boost_until  # 0#500000
        self.boost_strength_init = boost_strength

        self.permanences = self._get_initialized_permanences()

        self.synapse_reinf_coeffs = np.zeros((self.input_size_flat, self.size), dtype=float)
        self.discount = 0.0

        self._tie_break_scale = 0.00001
        self._tie_breaker = np.random.rand(self.size) * self._tie_break_scale

        self.reward_scaled_reinf = reward_scaled_reinf
        self.boost_scaled_reinf = boost_scaled_reinf
        self.only_reinforce_selected = only_reinforce_selected
        self.normalize_rewards = normalize_rewards

        self._rewards = deque(maxlen=reward_window)
        self._reinf_buf = None

    def _get_initialized_permanences(self):
        # Permanences are represented by a 2D matrix (input size X SP size)
        # If there is a synapse between an input cell and SP cell, the matrix gives its permanence
        # otherwise the matrix value is NaN
        permanences = np.empty((self.input_size_flat, self.size), dtype=float)
        permanences[:, :] = np.nan  # Init all to NaN first
        for col in range(self.size):
            # First choose which synapses to grow
            rand_selection = np.random.choice(self.input_size_flat, self.init_synapse_count, replace=False)
            # Then choose their initial permanences
            permanences[rand_selection, col] = self._get_initialized_segment()
        if log.has_debug():
            log.debug("SP has {} initialized synapses".format(np.count_nonzero(~np.isnan(permanences))))
        return permanences

    def _get_initialized_segment(self):
        vals = np.zeros((self.init_synapse_count,), dtype=float)
        # ~50% of all synapses should be active (permanence high enough); first decide which
        is_actives = [randval > 0.5 for randval in np.random.random(self.init_synapse_count)]
        # Then determine each permanence uniformly randomly in [min, thresh] or [thresh,max]
        for i in range(self.init_synapse_count):
            if is_actives[i]:
                vals[i] = self.connected_perm_thresh + (self.perm_max - self.connected_perm_thresh) * np.random.random()
            else:
                vals[i] = self.connected_perm_thresh * np.random.random()
        if (log.has_trace()):
            log.debug("Median:", np.median(vals))
            log.debug("Active: {}, average {}".format(len(vals[vals > self.connected_perm_thresh]),
                                                      np.mean(vals[vals > self.connected_perm_thresh])))
            log.debug("Inactive: {}, average {}".format(len(vals[vals < self.connected_perm_thresh]),
                                                        np.mean(vals[vals < self.connected_perm_thresh])))
        return vals

    def _perms_to_activateds(self, inputs, perms):
        connecteds = np.array((perms - self.connected_perm_thresh).clip(min=0), dtype=bool) * (
            ~ np.isnan(self.permanences))

        # Count the number of connected active input cells for each column
        conn_counts = np.dot(np.expand_dims(inputs, 0), np.array(connecteds, dtype=int))
        conn_counts = np.squeeze(conn_counts)
        conn_counts = np.add(conn_counts, self._tie_breaker, casting='unsafe')

        # Get the top-k columns in terms of connected active input cells
        # impl: argpartition calculates the indices that sort arg0 such that
        # at least the first arg1 entries are the arg1 smallest values, in order.
        # Makes no guarantees about the rest of the values (but we don't need those)
        activated = np.argpartition(- conn_counts, self.active_columns_count)[:self.active_columns_count, ]
        return activated

    def _get_activated_cols(self, inputs, learn=True):
        if not learn:
            if not hasattr(self, '_inference_cache'):
                import scipy.sparse
                if scipy.sparse.issparse(self.permanences):
                    dense_perms = self.permanences.toarray()
                else:
                    dense_perms = self.permanences

                if self.boost_strength:
                    dense_perms = dense_perms * self.boost_factors

                connecteds = np.array((dense_perms - self.connected_perm_thresh).clip(min=0), dtype=bool) * (~ np.isnan(dense_perms))

                # Transpose the cache so we can do a rapid matmul instead of slow row iteration
                self._inference_cache = np.array(connecteds, dtype=int).T

            # Ultra-fast inference dot product using transposed cache
            conn_counts = np.dot(self._inference_cache, inputs.astype(int))
            conn_counts = np.add(conn_counts, self._tie_breaker, casting='unsafe')
            activated = np.argpartition(-conn_counts, self.active_columns_count)[:self.active_columns_count]
            return activated
        else:
            if self.boost_strength:
                boost_perms = self.permanences * self.boost_factors
            else:
                boost_perms = self.permanences
            activated = self._perms_to_activateds(inputs, boost_perms)
            return activated

    def step(self, inputs, learn=True):
        activated_cols = self._get_activated_cols(inputs, learn=learn)
        if learn:
            self._reinf_buf = (inputs, activated_cols)
            self._updateDutyCycle(activated_cols)
            self._init_next_step()
        return activated_cols

    # For debugging and analysis
    def visualize_cell_usage(self, all_possible_states, outdir):

        import matplotlib.pyplot as plt

        # Disable boosting
        self.boost_strength = 0.0
        self.boost_anneal_until = 0

        # for visualization purposes, round up to square number
        square_length = math.ceil(math.sqrt(self.size))
        size_for_square = square_length ** 2
        col_counts = np.zeros((size_for_square,))

        for state in all_possible_states:
            activateds = self.step(state, False)

            col_counts[activateds] = col_counts[activateds] + 1

        plt.figure(10)
        fig, ax = plt.subplots(figsize=(20, 20))

        col_counts = np.reshape(col_counts, (square_length, square_length))
        ax.matshow(col_counts, cmap=plt.cm.Reds, vmin=0, vmax=10)

        for i in range(square_length):
            for j in range(square_length):
                c = col_counts[j, i]
                if c > 0:
                    ax.text(i, j, str(c), va='center', ha='center')
        plt.savefig(outdir + "sp_usage.png")
