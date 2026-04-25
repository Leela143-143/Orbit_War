
import numpy as np
import math

class ScalarEncoder:
    def __init__(self, size, active_bits, min_val, max_val):
        self.size = size
        self.active_bits = active_bits
        self.min_val = min_val
        self.max_val = max_val
        self.range = max(1e-6, max_val - min_val)
        self.buckets = size - active_bits + 1

    def encode(self, value):
        state = np.zeros(self.size, dtype=bool)
        v = min(max(value, self.min_val), self.max_val)
        bucket = int((v - self.min_val) / self.range * (self.buckets - 1))
        bucket = min(max(bucket, 0), self.buckets - 1)
        state[bucket : bucket + self.active_bits] = True
        return state

    def encode_batch(self, values):
        # returns an array of bucket start indices
        v = np.clip(values, self.min_val, self.max_val)
        buckets = ((v - self.min_val) / self.range * (self.buckets - 1)).astype(int)
        return np.clip(buckets, 0, self.buckets - 1)

class CyclicEncoder:
    def __init__(self, size, active_bits, min_val=0.0, max_val=2*math.pi):
        self.size = size
        self.active_bits = active_bits
        self.min_val = min_val
        self.max_val = max_val
        self.range = max_val - min_val

    def encode(self, value):
        state = np.zeros(self.size, dtype=bool)
        norm_val = ((value - self.min_val) % self.range) / self.range
        if norm_val < 0: norm_val += 1.0
        center_idx = int(norm_val * self.size)
        half_active = self.active_bits // 2
        for i in range(self.active_bits):
            idx = (center_idx - half_active + i) % self.size
            state[idx] = True
        return state

    def encode_batch(self, values):
        # returns an array of center indices
        norm_vals = ((values - self.min_val) % self.range) / self.range
        norm_vals[norm_vals < 0] += 1.0
        return (norm_vals * self.size).astype(int)

class TileGeospatialEncoder:
    def __init__(self, size=2000, active_bits=50, is_fleet=False):
        self.size = size
        self.active_bits = active_bits
        self.is_fleet = is_fleet

        if self.is_fleet:
            self.angle_encoder = CyclicEncoder(1000, 25, 0, 2*math.pi)
            self.dist_encoder = ScalarEncoder(1000, 25, 0, 150)
            self.angle_encoder_phase = CyclicEncoder(1000, 25, 0, 2*math.pi)
            self.heading_encoder = CyclicEncoder(1000, 25, 0, 2*math.pi)
        else:
            self.angle_encoder = CyclicEncoder(1000, 25, 0, 2*math.pi)
            self.dist_encoder = ScalarEncoder(1000, 25, 0, 150)

    def encode(self, dx, dy, heading=None):
        pass # Not used directly anymore, replaced by encode_union_topk which is fully vectorized

    def encode_union_topk(self, origin_x, origin_y, targets):
        final_state = np.zeros(self.size, dtype=bool)
        n = len(targets)
        if n == 0:
            return final_state

        dx = np.array([t.x - origin_x for t in targets], dtype=np.float32)
        dy = np.array([t.y - origin_y for t in targets], dtype=np.float32)

        dist = np.hypot(dx, dy)
        angles = np.arctan2(dy, dx)
        angles[angles < 0] += 2 * math.pi

        weights = 1.0 / (1.0 + dist)

        float_state = np.zeros(self.size, dtype=np.float32)

        # Angle 1
        a_centers = self.angle_encoder.encode_batch(angles)
        half_a = self.angle_encoder.active_bits // 2
        for i in range(self.angle_encoder.active_bits):
            idx = (a_centers - half_a + i) % self.angle_encoder.size
            np.add.at(float_state, idx, weights)

        # Distance
        d_buckets = self.dist_encoder.encode_batch(dist)
        # Offset for distance array
        d_offset = 1000
        for i in range(self.dist_encoder.active_bits):
            idx = d_offset + d_buckets + i
            np.add.at(float_state, idx, weights)

        if self.is_fleet:
            # Angle 2 Phase Shifted
            a_shifted = (angles + math.pi) % (2*math.pi)
            a_centers_2 = self.angle_encoder_phase.encode_batch(a_shifted)
            a2_offset = 2000
            for i in range(self.angle_encoder_phase.active_bits):
                idx = a2_offset + ((a_centers_2 - half_a + i) % self.angle_encoder_phase.size)
                np.add.at(float_state, idx, weights)

            # Flight Heading
            headings = np.array([getattr(t, 'angle', 0.0) or 0.0 for t in targets], dtype=np.float32)
            h_centers = self.heading_encoder.encode_batch(headings)
            h_offset = 3000
            for i in range(self.heading_encoder.active_bits):
                idx = h_offset + ((h_centers - half_a + i) % self.heading_encoder.size)
                np.add.at(float_state, idx, weights)

        non_zero = float_state > 0
        if not np.any(non_zero):
            return final_state

        # Fast Top-K
        k = min(self.active_bits, np.count_nonzero(float_state))
        if k > 0:
            top_k_indices = np.argpartition(float_state, -k)[-k:]
            final_state[top_k_indices] = True

        return final_state
