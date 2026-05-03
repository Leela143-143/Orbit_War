import math
from collections import defaultdict
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

# --- CONSTANTS ---
BOARD = 100.0
CENTER_X, CENTER_Y = 50.0, 50.0
SUN_R = 10.0
MAX_SPEED = 6.0
TOTAL_STEPS = 500

# ==========================================
# GEOMETRY & PHYSICS
# ==========================================
def dist(ax, ay, bx, by): return math.hypot(ax - bx, ay - by)

def fleet_speed(ships):
    if ships <= 1: return 1.0
    ratio = math.log(max(1, ships)) / math.log(1000.0)
    return 1.0 + (MAX_SPEED - 1.0) * (max(0.0, min(1.0, ratio)) ** 1.5)

def segment_hits_circle(x1, y1, x2, y2, cx, cy, r):
    if x1 < x2:
        if cx + r < x1 or cx - r > x2: return False
    else:
        if cx + r < x2 or cx - r > x1: return False
    if y1 < y2:
        if cy + r < y1 or cy - r > y2: return False
    else:
        if cy + r < y2 or cy - r > y1: return False

    dx, dy = x2 - x1, y2 - y1
    fx, fy = x1 - cx, y1 - cy
    a = dx*dx + dy*dy
    if a < 1e-9: return math.hypot(fx, fy) <= r
    b = 2*(fx*dx + fy*dy)
    c = fx*fx + fy*fy - r*r
    disc = b*b - 4*a*c
    if disc < 0: return False
    disc = math.sqrt(disc)
    t1 = (-b - disc) / (2*a)
    t2 = (-b + disc) / (2*a)
    if (0 <= t1 <= 1) or (0 <= t2 <= 1): return True
    if t1 < 0 and t2 > 1: return True
    return False

def estimate_arrival(sx, sy, tx, ty, ships, r_src=0.0, r_tgt=0.0):
    d = max(0.0, dist(sx, sy, tx, ty) - r_src - 0.1 - r_tgt)
    sp = fleet_speed(ships)
    return math.atan2(ty - sy, tx - sx), max(1, int(math.ceil(d / sp)))

# ==========================================
# ORBITAL PREDICTION
# ==========================================
def predict_planet_pos(planet, initial_by_id, ang_vel, turns):
    init = initial_by_id.get(planet.id)
    if init is None: return planet.x, planet.y
    orbital_r = dist(init.x, init.y, CENTER_X, CENTER_Y)
    if orbital_r + init.radius >= 50.0: return planet.x, planet.y
    cur_ang = math.atan2(planet.y - CENTER_Y, planet.x - CENTER_X)
    new_ang = cur_ang + ang_vel * turns
    return CENTER_X + orbital_r * math.cos(new_ang), CENTER_Y + orbital_r * math.sin(new_ang)

def get_comet_lifespan(planet_id, comets):
    for g in comets:
        pids = g.get("planet_ids", [])
        if planet_id not in pids: continue
        idx = pids.index(planet_id)
        paths = g.get("paths", [])
        path_index = g.get("path_index", 0)
        if idx >= len(paths): return 0
        path = paths[idx]
        return max(0, len(path) - path_index)
    return 500

def predict_comet_pos(planet_id, comets, turns):
    for g in comets:
        pids = g.get("planet_ids", [])
        if planet_id not in pids: continue
        idx = pids.index(planet_id)
        paths = g.get("paths", [])
        path_index = g.get("path_index", 0)
        if idx >= len(paths): return None
        path = paths[idx]
        future_idx = path_index + int(turns)
        if 0 <= future_idx < len(path):
            return path[future_idx][0], path[future_idx][1]
        return None
    return None

def predict_pos(planet, initial_by_id, ang_vel, comets, comet_ids, turns):
    if planet.id in comet_ids:
        return predict_comet_pos(planet.id, comets, turns)
    return predict_planet_pos(planet, initial_by_id, ang_vel, turns)

def precompute_trajectories(planets, initial_by_id, ang_vel, comets, comet_ids, max_turns):
    traj = {}
    for p in planets:
        traj[p.id] = [predict_pos(p, initial_by_id, ang_vel, comets, comet_ids, t) for t in range(1, max_turns + 1)]
    return traj

# ==========================================
# THREAT MAP
# ==========================================
def build_threat_map(fleets, planets, traj, max_turns=150):
    arrivals = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for f in fleets:
        sp = fleet_speed(f.ships)
        vx, vy = math.cos(f.angle) * sp, math.sin(f.angle) * sp
        fx, fy = f.x, f.y
        for t in range(1, max_turns + 1):
            nx, ny = fx + vx, fy + vy
            if not (0 <= nx <= BOARD and 0 <= ny <= BOARD): break
            if segment_hits_circle(fx, fy, nx, ny, CENTER_X, CENTER_Y, SUN_R): break

            hit_pid = None
            for p in planets:
                if t - 1 >= len(traj[p.id]): continue
                pos = traj[p.id][t - 1]
                if pos is None: continue
                px, py = pos
                if segment_hits_circle(fx, fy, nx, ny, px, py, p.radius):
                    hit_pid = p.id
                    break

            if hit_pid is not None:
                arrivals[hit_pid][t][f.owner] += f.ships
                break
            else:
                fx, fy = nx, ny
    return arrivals

# ==========================================
# DISCRETE EVENT SIMULATION
# ==========================================
def simulate_planet(planet, arrivals, test_fleet=None, max_turn=100):
    owner = planet.owner
    ships = planet.ships
    prod = planet.production

    events = defaultdict(lambda: defaultdict(int))
    if arrivals:
        for t, arrs in arrivals.items():
            if t > max_turn: continue
            for o, s in arrs.items():
                events[t][o] += s

    if test_fleet:
        arr_t, s, o = test_fleet
        if arr_t <= max_turn:
            events[arr_t][o] += s

    if not events:
        if owner != -1:
            ships += prod * max_turn
        return owner, ships

    max_event_t = max(events.keys())
    actual_max = min(max_turn, max_event_t)

    for t in range(1, actual_max + 1):
        if owner != -1:
            ships += prod

        if t in events:
            arrs = events[t]
            att_forces = []
            for o, s in arrs.items():
                att_forces.append((s, o))

            if att_forces:
                att_forces.sort(reverse=True)
                if len(att_forces) == 1:
                    surv_s, surv_o = att_forces[0]
                else:
                    surv_s = att_forces[0][0] - att_forces[1][0]
                    surv_o = att_forces[0][1] if surv_s > 0 else -1

                if surv_s > 0 and surv_o != -1:
                    if surv_o == owner:
                        ships += surv_s
                    else:
                        if surv_s > ships:
                            owner = surv_o
                            ships = surv_s - ships
                        elif surv_s == ships:
                            owner = -1
                            ships = 0
                        else:
                            ships -= surv_s

    if actual_max < max_turn and owner != -1:
        ships += prod * (max_turn - actual_max)

    return owner, ships

def evaluate_timeline(planet, arrivals, player, remaining_steps, is_ffa, comet_lifespan, test_fleet=None):
    max_t = min(remaining_steps, comet_lifespan)
    last_event = 0
    if arrivals: last_event = max(arrivals.keys())
    if test_fleet: last_event = max(last_event, test_fleet[0])

    sim_t = min(max_t, last_event)
    owner, ships = simulate_planet(planet, arrivals, test_fleet, sim_t)

    post_turns = max_t - sim_t
    if owner != -1: ships += post_turns * planet.production

    if owner == player: return ships
    elif owner == -1: return 0
    else: return 0 if is_ffa else -ships

def safe_reserve(planet, arrivals, player, remaining_steps):
    low, high = 0, planet.ships
    best = planet.ships
    while low <= high:
        mid = (low + high) // 2
        dummy_p = Planet(planet.id, planet.owner, planet.x, planet.y, planet.radius, mid, planet.production)
        owner_end, _ = simulate_planet(dummy_p, arrivals, max_turn=min(150, remaining_steps))
        if owner_end == player:
            best = mid
            high = mid - 1
        else:
            low = mid + 1
    return best

# ==========================================
# AIMING & VERIFICATION
# ==========================================
def path_blocked_by_planet(src, target, angle, ships, planets, traj, turns):
    sp = fleet_speed(ships)
    vx, vy = math.cos(angle) * sp, math.sin(angle) * sp
    fx, fy = src.x + math.cos(angle)*(src.radius + 0.1), src.y + math.sin(angle)*(src.radius + 0.1)

    for t in range(1, turns + 1):
        nx, ny = fx + vx, fy + vy
        if segment_hits_circle(fx, fy, nx, ny, CENTER_X, CENTER_Y, SUN_R): return True
        for p in planets:
            if p.id == target.id or p.id == src.id: continue
            if t - 1 >= len(traj[p.id]): continue
            pos = traj[p.id][t - 1]
            if pos is None: continue
            px, py = pos
            if segment_hits_circle(fx, fy, nx, ny, px, py, p.radius): return True
        fx, fy = nx, ny
    return False

def aim_and_need(src, target, arrivals, player, remaining_steps, planets, traj, initial_by_id, ang_vel, comets, comet_ids):
    low, high = 1, 1500
    best = None

    while low <= high:
        mid = (low + high) // 2
        tx, ty = target.x, target.y
        for _ in range(5):
            _, turns = estimate_arrival(src.x, src.y, tx, ty, mid, src.radius, target.radius)
            pos = predict_pos(target, initial_by_id, ang_vel, comets, comet_ids, turns)
            if pos is None: break
            tx, ty = pos[0], pos[1]

        if pos is None:
            high = mid - 1
            continue

        angle, turns = estimate_arrival(src.x, src.y, tx, ty, mid, src.radius, target.radius)
        owner, _ = simulate_planet(target, arrivals, test_fleet=(turns, mid, player), max_turn=min(150, remaining_steps))

        if owner == player:
            best = mid
            high = mid - 1
        else:
            low = mid + 1

    if best is None: return None

    send = max(10, int(best * 1.15))

    tx, ty = target.x, target.y
    for _ in range(5):
        angle, turns = estimate_arrival(src.x, src.y, tx, ty, send, src.radius, target.radius)
        pos = predict_pos(target, initial_by_id, ang_vel, comets, comet_ids, turns)
        if pos is None: return None
        tx, ty = pos[0], pos[1]
    angle, turns = estimate_arrival(src.x, src.y, tx, ty, send, src.radius, target.radius)

    if path_blocked_by_planet(src, target, angle, send, planets, traj, turns): return None
    return send, angle, turns

# ==========================================
# MAIN AGENT
# ==========================================
def agent(obs):
    get = obs.get if isinstance(obs, dict) else lambda k, d=None: getattr(obs, k, d)
    player        = get("player", 0)
    step          = get("step", 0) or 0
    planets       = [Planet(*p) for p in get("planets", [])]
    fleets        = [Fleet(*f) for f in get("fleets", [])]
    ang_vel       = get("angular_velocity", 0.0) or 0.0
    initial_by_id = {Planet(*p).id: Planet(*p) for p in get("initial_planets", [])}
    comets        = get("comets", []) or []
    comet_ids     = set(get("comet_planet_ids", []) or [])
    my_planets    = [p for p in planets if p.owner == player]

    if not my_planets: return []

    remaining = max(1, TOTAL_STEPS - step)
    n_players = len(set([p.owner for p in planets if p.owner != -1]))
    is_ffa = n_players > 2

    traj = precompute_trajectories(planets, initial_by_id, ang_vel, comets, comet_ids, max_turns=250)
    arrivals = build_threat_map(fleets, planets, traj, max_turns=150)
    moves = []


    for src in my_planets:
        lifespan = get_comet_lifespan(src.id, comets) if src.id in comet_ids else 500
        res = safe_reserve(src, arrivals.get(src.id, {}), player, remaining)

        # Strategic Sacrifice: Free up defensive ships on weak planets for aggressive use
        if src.production <= 2 and res > src.ships * 0.8:
            res = int(src.ships * 0.5)
        available = src.ships - res

        # --- COMET EVACUATION PROTOCOL ---
        if lifespan <= 5:
            available = src.ships
            if available > 0:
                friends = [p for p in my_planets if p.id != src.id and p.id not in comet_ids]
                if friends:
                    best_f = min(friends, key=lambda f: dist(src.x, src.y, f.x, f.y))
                    tx, ty = best_f.x, best_f.y
                    for _ in range(5):
                        _, turns = estimate_arrival(src.x, src.y, tx, ty, available, src.radius, best_f.radius)
                        pos = predict_pos(best_f, initial_by_id, ang_vel, comets, comet_ids, turns)
                        if pos is None: break
                        tx, ty = pos[0], pos[1]
                    angle, _ = estimate_arrival(src.x, src.y, tx, ty, available, src.radius, best_f.radius)
                else:
                    corners = [(0,0), (0,100), (100,0), (100,100)]
                    best_c = max(corners, key=lambda c: dist(src.x, src.y, c[0], c[1]))
                    angle = math.atan2(best_c[1] - src.y, best_c[0] - src.x)
                moves.append([src.id, float(angle), int(available)])
            continue

        if available < 10: continue

        candidates = []
        for tgt in planets:
            if src.id == tgt.id: continue

            d = max(1.0, dist(src.x, src.y, tgt.x, tgt.y))
            score = tgt.production / d

            # Dynamic multiplier based on state (offense, defense, neutral expansion)
            if tgt.owner != player and tgt.owner != -1:
                # Offensive target
                score *= 1.5
            elif tgt.owner == -1:
                # Neutral target: Prioritize them significantly since they are cheap
                score *= 1.25
            elif tgt.owner == player:
                # Defense: If it's a friendly planet, is it under threat?
                arrs = arrivals.get(tgt.id, {})
                threatened = False
                for t, o_s in arrs.items():
                    for o, s in o_s.items():
                        if o != player and o != -1 and s > 0:
                            threatened = True
                            break
                    if threatened: break

                if threatened:
                    # Defense is highly valuable to save production swing
                    score *= 1.8
                else:
                    score = 0.0 # Don't reinforce safe planets

            candidates.append((score, tgt))

        candidates.sort(key=lambda x: -x[0])

        for _, tgt in candidates[:12]:
            if available < 10: break

            result = aim_and_need(src, tgt, arrivals.get(tgt.id, {}), player, remaining, planets, traj, initial_by_id, ang_vel, comets, comet_ids)
            if result is None: continue

            send, angle, turns = result
            if turns > remaining: continue
            if send > available: continue

            tgt_life = get_comet_lifespan(tgt.id, comets) if tgt.id in comet_ids else 500

            V_A = evaluate_timeline(tgt, arrivals.get(tgt.id, {}), player, remaining, is_ffa, tgt_life)
            V_B = evaluate_timeline(tgt, arrivals.get(tgt.id, {}), player, remaining, is_ffa, tgt_life, test_fleet=(turns, send, player))
            profit = (V_B - send) - V_A

            if profit > 0:
                # If we are defending, we don't send ships if they would arrive AFTER the planet falls
                moves.append([src.id, float(angle), int(send)])
                arrivals[tgt.id][turns][player] += send
                available -= send

    return moves
