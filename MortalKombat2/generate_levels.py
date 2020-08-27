import retro
import numpy as np
import gzip


def get_action_vec(p1_action_names, p2_action_names):
    buttons = ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']
    action_vec = np.zeros(shape=2 * len(buttons))
    for a in p1_action_names:
        action_vec[buttons.index(a)] = 1

    for a in p2_action_names:
        action_vec[buttons.index(a) + len(buttons)] = 1

    return action_vec


def wait_for_black_screen(env):
    # Waiting for the screen to be black
    while np.max(env.step(get_action_vec([], []))[0]) > 0:
        pass
    # And be normal again (game starts)
    while np.max(env.step(get_action_vec([], []))[0]) == 0:
        pass


game_name = "MortalKombatII-Genesis"
fighters_matrix = np.array([
    ["LiuKang", "KungLao", "JohnnyCage", "Reptile"],
    ["SubZero", "ShangTsung", "Kitana", "Jax"],
    ["Mileena", "Baraka", "Scorpion", "Raiden"]
])

fighters_list = np.concatenate(fighters_matrix)
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

        wait_for_black_screen(env)
        state = env.em.get_state()
        env.close()

        with open(f"states/{p1}_vs_{p2}_2p.state", "wb") as f:
            f.write(gzip.compress(state))

for p1 in fighters_list:
    env = retro.make(game_name, players=2, state="choose_fighter")
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

    wait_for_black_screen(env)
    wait_for_black_screen(env)
    state = env.em.get_state()
    env.close()

    with open(f"states/{p1}_vs_Jax_1p.state", "wb") as f:
        f.write(gzip.compress(state))
