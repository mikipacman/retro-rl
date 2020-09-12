import retro
from MortalKombat2.state_generation.utils import *
from MortalKombat2 import *
from PIL import Image
import os
import matplotlib.pyplot as plt


# Params:
path_for_bars = "fighters_bars/"
path_for_states_to_save = "states/1p/by_difficulty_arena_and_opponent/"
minimum_states_per_opponent = 6
difficulties_to_generate_from = ["VeryEasy", "Medium", "VeryHard"]
arenas_to_generate_from = ["DeadPool", "Portal", "LivingForest"]
# cords of opponent bar: [15:25, -145:-23, :]


# Load bars
bars = {}
for root, _, files in os.walk(path_for_bars):
    for file in files:
        im = Image.open(os.path.join(root, file))
        fighter_name = file.split("_")[0]
        bars[fighter_name] = im

counter = 0

for difficulty in difficulties_to_generate_from:
    for arena in arenas_to_generate_from:
        explored_opponents = {o: 0 for o in bars.keys()}

        env = retro.make(game_name, players=2, state=f"{difficulty}_{arena}", use_restricted_actions=retro.Actions.ALL)
        env.reset()
        prev_state = env.em.get_state()

        while min(explored_opponents.values()) < minimum_states_per_opponent:

            # Menu clicking and stuff
            env.em.set_state(prev_state)
            wait_for_next_menu(env, False)
            prev_state = env.em.get_state()
            env.step(get_action_vec(["A"], []))
            wait_n(env, 200, False)
            state = env.em.get_state()
            env.step(get_action_vec(["A"], []))
            wait_for_black_screen(env, False)
            wait_for_black_screen(env, False)
            wait_n(env, 125, False)

            # Get frame and decide which opponent is currently playing
            obs = env.render(mode="rgb_array")[15:25, -145:-23, :]
            cadidates = [k for k, v in bars.items() if np.all(np.array(v) == obs)]

            # Exceptions
            if len(cadidates) == 0:
                plt.imshow(obs)
                plt.show()
                raise Exception("No cadidates")
            elif len(cadidates) > 1:
                plt.imshow(obs)
                plt.show()
                raise Exception("Too many candidates")

            # Update counters and save
            opponent = cadidates[0]
            num_of_state = explored_opponents[opponent]
            explored_opponents[opponent] += 1

            if num_of_state < minimum_states_per_opponent:
                save_state(state, os.path.join(path_for_states_to_save,
                                               f"{difficulty}_{arena}_{opponent}_{num_of_state}.state"))
            counter += 1
            end = "" if min(explored_opponents.values()) < minimum_states_per_opponent else "\n"
            print("\r", counter, difficulty.upper(), arena.upper(), explored_opponents, end=end)

        env.close()
