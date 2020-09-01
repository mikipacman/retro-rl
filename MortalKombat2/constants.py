import numpy as np


game_name = "MortalKombatII-Genesis"

# All game data

fighters_matrix = np.array([
    ["LiuKang", "KungLao", "JohnnyCage", "Reptile"],
    ["SubZero", "ShangTsung", "Kitana", "Jax"],
    ["Mileena", "Baraka", "Scorpion", "Raiden"]
])

fighters_list = np.concatenate(fighters_matrix)

difficulties = [
    "VeryEasy",
    "Easy",
    "Medium",
    "Hard",
    "VeryHard"
]

arenas = [
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

# States are hard to generate and their number grows quickly so not all
# opponents arenas and difficulties are available, however it is possible
# to generate more

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
