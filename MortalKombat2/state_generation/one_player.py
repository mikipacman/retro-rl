import retro
from MortalKombat2.state_generation.utils import *
from MortalKombat2 import *
import time


difficulties = ["VeryEasy", "Medium", "VeryHard"]
arenas = ["DeadPool", "LivingForest", "Portal"]
opponents = ["Raiden", "Jax", "SubZero", "Scorpion", "Baraka"]
versions = range(6)

counter = 0
num_to_generate = len(difficulties) * len(arenas) * len(opponents) * len(versions) * len(all_fighters)
start = time.time()

for difficulty in difficulties:
    for arena in arenas:
        for opp in opponents:
            for version in versions:
                for p1 in all_fighters:
                    env = retro.make(game_name, players=2, state=f"{difficulty}_{arena}_{opp}_{version}")
                    env.reset()

                    # Get paths for cursors
                    p1_cords = np.where(fighters_matrix == p1)
                    p1_actions = ["RIGHT", "LEFT"]
                    for _ in range(p1_cords[0][0]):
                        p1_actions.append("DOWN")

                    for _ in range(p1_cords[1][0]):
                        p1_actions.append("RIGHT")

                    p1_actions.append("A")

                    # Execute and save
                    for a1 in p1_actions:
                        env.step(get_action_vec([a1], []))
                        env.step(get_action_vec([], []))

                    wait_for_black_screen(env, False)
                    wait_for_black_screen(env, False)
                    state = env.em.get_state()
                    env.close()

                    with open(f"states/ready_to_play/1p_{difficulty}_{arena}_{p1}_vs_{opp}_{version}.state", "wb") as f:
                        f.write(gzip.compress(state))

                    counter += 1
                    print(f"\r{counter}/{num_to_generate}\t{(time.time() - start) / 60} minutes elapsed", end="")
