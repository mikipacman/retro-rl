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


def wait_for_black_screen(env, render=True):
    # Waiting for the screen to be black
    while np.max(env.step(get_action_vec([], []))[0]) > 0:
        if render:
            env.render()
    # And be normal again (game starts)
    while np.max(env.step(get_action_vec([], []))[0]) == 0:
        if render:
            env.render()


def wait_n(env, n):
    for _ in range(n):
        env.step(get_action_vec([], []))
        env.render()


def choose_fighter(env):
    # Choose Fighter
    env.step(get_action_vec(["A"], []))
    wait_n(env, 100)

    env.step(get_action_vec([], []))
    wait_n(env, 100)

    # Change your player here
    env.step(get_action_vec(["A"], []))
    wait_for_black_screen(env)
    wait_for_black_screen(env)

    # Save State here
    wait_n(env, 100)


def wait_for_next_menu(env):
    wait_for_black_screen(env)
    wait_n(env, 25)
    env.step(get_action_vec(["START"], []))
    wait_n(env, 100)


def save_state(state, name):
    with open(name, "wb") as f:
        f.write(gzip.compress(state))
