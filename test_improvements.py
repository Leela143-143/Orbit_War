"""Quick test to verify the improved bot runs correctly against the old one."""
from kaggle_environments import make

def run_test(n_games=5):
    wins = [0, 0]
    draws = 0
    
    for i in range(n_games):
        env = make("orbit_wars", debug=True)
        env.run(["main.py", "improvement.py"])
        
        final = env.steps[-1]
        r0 = final[0].reward
        r1 = final[1].reward
        
        if r0 > r1:
            wins[0] += 1
            result = "OLD wins"
        elif r1 > r0:
            wins[1] += 1
            result = "NEW wins"
        else:
            draws += 1
            result = "DRAW"
        
        print(f"Game {i+1}: P0(old)={r0:.0f}  P1(new)={r1:.0f}  -> {result}")
    
    print(f"\n--- Summary ({n_games} games) ---")
    print(f"OLD bot wins: {wins[0]}")
    print(f"NEW bot wins: {wins[1]}")
    print(f"Draws: {draws}")

if __name__ == "__main__":
    run_test(5)
