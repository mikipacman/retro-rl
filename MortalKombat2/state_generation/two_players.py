import retro
from MortalKombat2.state_generation.utils import *
from MortalKombat2.constants import *

for p1 in fighters_list:
    for p2 in fighters_list:
        env = retro.make(game_name, players=2, state="choose_fighters")
        env.reset()

        # Get paths for cursors
        p1_cords = np.where(fighters_matrix == p1)
        p1_actions = ["RIGHT", "LEFT"]
        for _ in range(p1_cords[0][0]):
            p1_actions.append("DOWN")

        for _ in range(p1_cords[1][0]):
            p1_actions.append("RIGHT")

        p2_cords = np.where(fighters_matrix == p2)
        p2_actions = []
        for _ in range(p2_cords[0][0]):
            p2_actions.append("DOWN")

        for _ in range(3 - p2_cords[1][0]):
            p2_actions.append("LEFT")

        p1_actions.append("A")
        p2_actions.append("A")

        if len(p1_actions) < len(p2_actions):
            for _ in range(len(p2_actions) - len(p1_actions)):
                p1_actions.append("A")
        else:
            for _ in range(len(p1_actions) - len(p2_actions)):
                p2_actions.append("A")

        # Execute and save
        for a1, a2 in zip(p1_actions, p2_actions):
            env.step(get_action_vec([a1], [a2]))
            env.step(get_action_vec([], []))

        wait_for_black_screen(env, False)
        state = env.em.get_state()
        env.close()

        with open(f"states/ready_to_play/2p_DeadPool_{p1}_vs_{p2}.state", "wb") as f:
            f.write(gzip.compress(state))
