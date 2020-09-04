import numpy as np
from MortalKombat2.wrappers import MK2Wrapper, MaxEpLenWrapper, FrameskipWrapper
import retro

# Constants
game_name = "MortalKombatII-Genesis"

# All game data
fighters_matrix = np.array([
    ["LiuKang", "KungLao", "JohnnyCage", "Reptile"],
    ["SubZero", "ShangTsung", "Kitana", "Jax"],
    ["Mileena", "Baraka", "Scorpion", "Raiden"]
])
all_fighters = np.concatenate(fighters_matrix)
all_difficulties = [
    "VeryEasy",
    "Easy",
    "Medium",
    "Hard",
    "VeryHard"
]
all_arenas = [
    "DeadPool",
    "KombatTomb",
    "Wasteland",
    "Tower",
    "LivingForest",
    "Armory",
    "Pit",
    "Portal",
    "KahnsArena",
]

# Available game data
# States are hard to generate and their number grows quickly so not all
# opponents arenas and difficulties are available, however it is possible to generate more
available_opponents = [
    "Raiden",
    "Jax",
    "Baraka",
    "SubZero",
    "Scorpion"
]
available_arenas = [
    "DeadPool",
    "LivingForest",
    "Portal"
]
available_difficulties = [
    "VeryEasy",
    "Medium",
    "VeryHard",
]
available_two_players_arenas = [
    "DeadPool"
]


# Helper function for making nice env
# The env makes random combination of (difficulties, arenas, left_players, right_players) after each env.reset()
def make_mortal_kombat2_env(difficulties, arenas, left_players, right_players, controllable_players, actions,
                            reward_weights="default", done_after="round"):
    states = []

    if controllable_players == 1:
        assert all([d in available_difficulties for d in difficulties])
        assert all([a in available_arenas for a in arenas])
        assert all([r in available_opponents for r in right_players])
        assert all([l in all_fighters for l in left_players])

        for d in difficulties:
            for a in arenas:
                for l in left_players:
                    for r in right_players:
                        states.append(f"1p_{d}_{a}_{l}_vs_{r}")
    else:
        assert controllable_players == 2
        assert difficulties == []
        assert all([r in all_fighters for r in right_players])
        assert all([l in all_fighters for l in left_players])
        assert all([a in available_two_players_arenas for a in arenas])

        for a in arenas:
            for l in left_players:
                for r in right_players:
                    states.append(f"2p_{a}_{l}_vs_{r}")

    actions_map = {
        "ALL": retro.Actions.ALL,
        "FILTERED": retro.Actions.FILTERED,
        "DISCRETE": retro.Actions.DISCRETE,
        "MULTI_DISCRETE": retro.Actions.MULTI_DISCRETE,
    }
    assert actions in actions_map
    assert done_after in ["match", "round"]

    if reward_weights == "default":
        reward_weights = {
            "opp_health_factor": 0.1,
            "own_health_factor": -0.1,
            "win_factor": 10,
            "lose_factor": -10,
        }

    env = retro.make(game=game_name, players=controllable_players,
                     use_restricted_actions=actions_map[actions], state=retro.State.NONE)
    env = MK2Wrapper(env=env, states=states, done_after=done_after, **reward_weights)

    return env
