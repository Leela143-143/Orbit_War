import re

with open('main.py', 'r') as f:
    content = f.read()

agent_start = content.find("def agent(obs):")

new_agent_code = """def agent(obs):
    import time
    start_time = time.time()

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

    # Pre-calculate base values and available ships for each of our planets
    planet_state = []

    for src in my_planets:
        lifespan = get_comet_lifespan(src.id, comets) if src.id in comet_ids else 500
        res = safe_reserve(src, arrivals.get(src.id, {}), player, remaining)

        # Strategic Sacrifice from improvement.py
        if src.production <= 2 and res > src.ships * 0.8:
            res = int(src.ships * 0.5)

        available = src.ships - res

        # --- COMET EVACUATION PROTOCOL ---
        evac_move = None
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
                evac_move = [src.id, float(angle), int(available)]

        if evac_move:
            planet_state.append({'src': src, 'available': 0, 'evac_move': evac_move})
        else:
            planet_state.append({'src': src, 'available': available, 'evac_move': None})

    # Beam Search configuration
    BEAM_WIDTH = 5
    # Beam state: (score, moves_list, remaining_available_dict, modified_arrivals_dict)
    # We use a simplified arrivals dictionary delta to track simulated moves.
    import copy

    def copy_arrivals(arrs):
        new_arrs = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        for pid, turns_dict in arrs.items():
            for t, owner_dict in turns_dict.items():
                for o, count in owner_dict.items():
                    new_arrs[pid][t][o] = count
        return new_arrs

    beam = [(0.0, [], {ps['src'].id: ps['available'] for ps in planet_state}, copy_arrivals(arrivals))]

    # Process each planet sequentially to build up combination of moves
    for ps in planet_state:
        src = ps['src']

        # If evacuating, force the move and continue to next planet
        if ps['evac_move']:
            new_beam = []
            for score, moves, avail_dict, arrs in beam:
                new_moves = moves + [ps['evac_move']]
                new_beam.append((score, new_moves, avail_dict, arrs))
            beam = new_beam
            continue

        base_available = ps['available']
        if base_available < 10:
            continue

        # Find candidates for this planet based on distance and production (from improvement.py)
        candidates = []
        for tgt in planets:
            if src.id == tgt.id: continue

            d = max(1.0, dist(src.x, src.y, tgt.x, tgt.y))
            tgt_score = tgt.production / d

            # Dynamic multiplier based on state
            if tgt.owner != player and tgt.owner != -1:
                tgt_score *= 1.5
            elif tgt.owner == -1:
                tgt_score *= 1.25
            elif tgt.owner == player:
                arrs = arrivals.get(tgt.id, {})
                threatened = False
                for t, o_s in arrs.items():
                    for o, s in o_s.items():
                        if o != player and o != -1 and s > 0:
                            threatened = True
                            break
                    if threatened: break

                if threatened:
                    tgt_score *= 1.8
                else:
                    tgt_score = 0.0

            if tgt_score > 0:
                candidates.append((tgt_score, tgt))

        candidates.sort(key=lambda x: -x[0])
        top_candidates = candidates[:8] # Limit candidate branching

        new_beam = []

        # For each state in current beam, try all candidate actions
        for state_score, state_moves, avail_dict, state_arrs in beam:
            available = avail_dict[src.id]

            # Action 1: Do nothing
            new_beam.append((state_score, state_moves.copy(), avail_dict.copy(), copy_arrivals(state_arrs)))

            # CPU Timeout protection
            if time.time() - start_time > 0.8:
                break

            # Action 2+: Try attacking top candidates
            for _, tgt in top_candidates:
                if available < 10: break

                result = aim_and_need(src, tgt, state_arrs.get(tgt.id, {}), player, remaining, planets, traj, initial_by_id, ang_vel, comets, comet_ids)
                if result is None: continue

                send, angle, turns = result
                if turns > remaining: continue
                if send > available: continue

                tgt_life = get_comet_lifespan(tgt.id, comets) if tgt.id in comet_ids else 500

                V_A = evaluate_timeline(tgt, state_arrs.get(tgt.id, {}), player, remaining, is_ffa, tgt_life)
                V_B = evaluate_timeline(tgt, state_arrs.get(tgt.id, {}), player, remaining, is_ffa, tgt_life, test_fleet=(turns, send, player))
                profit = (V_B - send) - V_A

                if profit > 0:
                    new_moves = state_moves.copy()
                    new_moves.append([src.id, float(angle), int(send)])

                    new_avail_dict = avail_dict.copy()
                    new_avail_dict[src.id] -= send

                    new_arrs = copy_arrivals(state_arrs)
                    new_arrs[tgt.id][turns][player] += send

                    new_beam.append((state_score + profit, new_moves, new_avail_dict, new_arrs))

        # Keep only top K states to form the new beam
        new_beam.sort(key=lambda x: -x[0])
        beam = new_beam[:BEAM_WIDTH]

        # CPU Timeout protection
        if time.time() - start_time > 0.85:
            break

    # Return the moves of the best state found by the beam search
    if beam:
        best_state = max(beam, key=lambda x: x[0])
        return best_state[1]

    return []
"""

content = content[:agent_start] + new_agent_code

with open('main.py', 'w') as f:
    f.write(content)
