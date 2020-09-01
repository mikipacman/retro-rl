import retro
from MortalKombat2.state_generation.utils import *
from MortalKombat2.constants import *


for difficulty in ["VeryEasy", "Medium", "VeryHard"]:
    for arena in ["DeadPool", "LivingForest", "Portal"]:
        n = 0
        explored_opponents = set()
        print("\n", difficulty.upper(), arena.upper())
        print("Opponents List")
        for i, o in enumerate(fighters_list):
            print(i, o)

        env = retro.make(game_name, players=2, state=f"{difficulty}_{arena}",
                         use_restricted_actions=retro.Actions.ALL)
        env.reset()
        prev_state = env.em.get_state()
        while True:
            env.em.set_state(prev_state)
            wait_for_next_menu(env)
            prev_state = env.em.get_state()

            env.step(get_action_vec(["A"], []))
            wait_n(env, 200)

            state = env.em.get_state()

            env.step(get_action_vec(["A"], []))
            wait_for_black_screen(env)
            wait_for_black_screen(env)
            wait_n(env, 100)

            opponent = input(f"{len(explored_opponents)} opponents explored. Enter opponent number: ")
            n += 1

            if opponent == "end":
                break

            opponent = int(opponent)
            explored_opponents.add(fighters_list[opponent])
            save_state(state, f"states/1p/by_difficulty_arena_and_opponent/"
                              f"{difficulty}_{arena}_{fighters_list[opponent]}.state")
        env.close()
