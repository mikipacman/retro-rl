import retro
from MortalKombat2.state_generation.utils import *
from MortalKombat2.constants import *

for difficulty in difficulties:
    env = retro.make(game_name, players=2, state=difficulty,  use_restricted_actions=retro.Actions.ALL)
    env.reset()

    finished = False
    explored_arenas = set()

    print(f"\n\n{difficulty.upper()} difficulty")
    print("Arenas list:")
    for i, arena in enumerate(arenas):
        print(i, arena)
    print("\n")

    while True:
        state = env.em.get_state()

        # Waiting...
        choose_fighter(env)
        wait_for_next_menu(env)

        # Enter arena name
        level = input(f"Already explored {len(explored_arenas)} arenas. Enter arena number: ")
        if level == "end":
            env.close()
            break
        n = int(level)
        save_state(state, f"states/1p/by_difficulty_and_arena/{difficulty}_{arenas[n]}.state")
        explored_arenas.add(arenas[n])

        # Waiting...
        wait_for_next_menu(env)
        wait_for_next_menu(env)
        env.step(get_action_vec(["START"], []))
        wait_n(env, 100)
