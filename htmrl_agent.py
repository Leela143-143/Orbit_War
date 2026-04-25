import math
import numpy as np
import pickle
import os

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet
import HTMRL.spatial_pooler as spatial_pooler
import HTMRL.temporal_memory as temporal_memory

from HTMRL.encoders import ScalarEncoder, CyclicEncoder, TileGeospatialEncoder
from HTMRL.decoders import action_decode
def encoding_to_action(encoding, actions, sp_size):
    buckets = np.floor(encoding / (float(sp_size) / actions))
    buckets = buckets.astype(np.int32)
    counts = np.bincount(buckets, minlength=actions)
    return counts.argmax()


# 2.5% Sparsity rule
# My Ships: Size 1000, Active 25
# 3 Planet Channels: 2000 size (50 active) = 6,000
# 2 Fleet Channels: 4000 size (100 active) = 8,000
# Total INPUT_SIZE = 15000
INPUT_SIZE = 15000

class OrbitWarsEncoder:
    def __init__(self):
        self.size = INPUT_SIZE
        self.ships_encoder = ScalarEncoder(1000, 25, 0, 500)
        self.geo_planet_encoder = TileGeospatialEncoder(2000, 50, is_fleet=False)
        self.geo_fleet_encoder = TileGeospatialEncoder(4000, 100, is_fleet=True)

    def encode_fast(self, my_planet, enemy_planets, neutral_planets, friendly_planets, enemy_fleets, friendly_fleets):
        state = np.zeros(self.size, dtype=bool)
        state[0:1000] = self.ships_encoder.encode(my_planet.ships)
        
        # Remove the ego planet from the friendly planets list so it doesn't encode distance 0
        filtered_friendly = [p for p in friendly_planets if p.id != my_planet.id]

        offset = 1000
        state[offset:offset+2000] = self.geo_planet_encoder.encode_union_topk(my_planet.x, my_planet.y, enemy_planets)
        offset += 2000
        state[offset:offset+2000] = self.geo_planet_encoder.encode_union_topk(my_planet.x, my_planet.y, neutral_planets)
        offset += 2000
        state[offset:offset+2000] = self.geo_planet_encoder.encode_union_topk(my_planet.x, my_planet.y, filtered_friendly)
        offset += 2000
        state[offset:offset+4000] = self.geo_fleet_encoder.encode_union_topk(my_planet.x, my_planet.y, enemy_fleets)
        offset += 4000
        state[offset:offset+4000] = self.geo_fleet_encoder.encode_union_topk(my_planet.x, my_planet.y, friendly_fleets)
        
        return state

    def encode(self, my_planet, planets, fleets, player):
        pass # Replaced by encode_fast
def encoding_to_action(encoding, actions, sp_size):
    buckets = np.floor(encoding / (float(sp_size) / actions))
    buckets = buckets.astype(np.int32)
    counts = np.bincount(buckets, minlength=actions)
    return counts.argmax()

class HTMRLAgent:
    def __init__(self, load_path=None):
        self.encoder = OrbitWarsEncoder()
        if load_path and os.path.exists(load_path):
            with open(load_path, "rb") as f:
                data = pickle.load(f)
                if isinstance(data, dict) and "sp" in data:
                    self.sp = data["sp"]
                    self.tm = data.get("tm", temporal_memory.TemporalMemory())
                else:
                    self.sp = data
                    self.tm = temporal_memory.TemporalMemory()
        else:
            self.sp = spatial_pooler.SpatialPooler(
                input_size=(INPUT_SIZE,), 
                acts_n=1,
                cell_count=2048,
                active_count=41
            )
            self.tm = temporal_memory.TemporalMemory()
        self.tm_size = 2048 * 32
        self.tm_states = {}
        self.last_actions = {}
        self.last_states = {}

    def get_moves(self, obs, learn=False, reward=0):
        moves = []
        player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
        raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
        raw_fleets = obs.get("fleets", []) if isinstance(obs, dict) else getattr(obs, "fleets", [])

        planets = [Planet(*p) for p in raw_planets]
        from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet
        fleets = [Fleet(*f) for f in raw_fleets]
        
        my_planets = [p for p in planets if p.owner == player]
        
        # Optimize: Filter entity arrays exactly once per turn instead of per-planet
        enemy_planets = [p for p in planets if p.owner != player and p.owner != -1]
        neutral_planets = [p for p in planets if p.owner == -1]
        friendly_planets = my_planets
        enemy_fleets = [f for f in fleets if f.owner != player]
        friendly_fleets = [f for f in fleets if f.owner == player]
        
        for mine in my_planets:
            state = self.encoder.encode_fast(mine, enemy_planets, neutral_planets, friendly_planets, enemy_fleets, friendly_fleets)
            encoding = self.sp.step(state, learn=False)
            
            if len(encoding) > 0:
                action = encoding_to_action(encoding, 25, self.sp.size)
            else:
                action = 0
            
            if action == 0:
                continue
                
            if action <= 12:
                ships = max(1, int(mine.ships * 0.5))
                sector = action - 1
            else:
                ships = mine.ships
                sector = action - 13
                
            angle = (sector * (2 * math.pi / 12))
            moves.append([mine.id, angle, ships])
                
        return moves

_cached_agents = {}

def agent_fn(obs):
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    if player not in _cached_agents:
        _cached_agents[player] = HTMRLAgent(load_path="best_bot.pkl")
    return _cached_agents[player].get_moves(obs, learn=False)
