import MortalKombat2
import time

env = MortalKombat2.make_mortal_kombat2_env(difficulties=MortalKombat2.available_difficulties,
                                            arenas=MortalKombat2.available_arenas,
                                            left_players=MortalKombat2.all_fighters,
                                            right_players=MortalKombat2.available_opponents,
                                            controllable_players=1,
                                            actions="ALL")
for _ in range(5):
    env.reset()
    done = False
    while not done:
        _, _, done, _ = env.step(env.action_space.sample())
        env.render()
        time.sleep(1 / 120)
