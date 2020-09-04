import retro
from MortalKombat2.state_generation.utils import *
from MortalKombat2 import *

for difficulty in ["VeryEasy", "Medium", "VeryHard"]:
    for arena in ["DeadPool", "LivingForest", "Portal"]:
        for opp in ["Raiden", "Jax", "SubZero", "Scorpion", "Baraka"]:
            for p1 in all_fighters:
                env = retro.make(game_name, players=2, state=f"{difficulty}_{arena}_{opp}")
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

                with open(f"states/ready_to_play/1p_{difficulty}_{arena}_{p1}_vs_{opp}.state", "wb") as f:
                    f.write(gzip.compress(state))
