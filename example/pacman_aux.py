# Helper functions for the Ms. Pac-Man agent

import csv

from MAMEToolkit.emulator import Action
import numpy as np
from scipy.spatial import distance


MAP_FILE = "unique_map.csv"
DOTS_FILE = "unique_dots.csv"

MAP = []

DOTS = []
ENERGIZERS = np.asarray([
	(32.0, 28.0),
	(32.0, 228.0),
	(224.0, 28.0),
	(224.0, 228.0)
], dtype=np.float32)

X_MAP = []
Y_MAP = []

# 20, 28 upper right; 244, 228 lower left
CENTER_X = 132.0
CENTER_Y = 128.0

UNEXPLORED = []


# Get maze locations Ms. Pac-Man can use into a variable
# Used to prevent moving into maze walls
def load_legal_spaces():
	global MAP
	with open(MAP_FILE) as f:
		reader = csv.DictReader(f)
		for row in reader:
			MAP.append(row)


# Get dot locations in maze
# Used to locate proximity to closest uneaten dot
def load_dots():
	global DOTS
	global UNEXPLORED
	x_dots = []
	y_dots = []
	with open(DOTS_FILE) as f:
		reader = csv.DictReader(f)
		for row in reader:
			x_dots.append(np.float32(row["x"]))
			y_dots.append(np.float32(row["y"]))
	DOTS = np.array(np.transpose([x_dots, y_dots]))
	UNEXPLORED = np.copy(DOTS)


# Load maze locations into a variable to determine legal horizontal moves
def load_x_map():
	global MAP
	global X_MAP
	max_x = 264
	for _ in range(max_x+1):
		X_MAP.append([])
	for cell in MAP:
		X_MAP[int(cell["x"])].append(int(cell["y"]))


# Load maze locations into a variable to determine legal vertical moves
def load_y_map():
	global MAP
	global Y_MAP
	max_y = 256
	for _ in range(max_y+1):
		Y_MAP.append([])
	for cell in MAP:
		Y_MAP[int(cell["y"])].append(int(cell["x"]))


# Inverse horizontal or vertical distance between two objects
def get_measure_proximity(measure_from, measure_to):
	if measure_from == measure_to:
		return 0
	return 1 / (measure_from - measure_to)


# Inverse horizontal and vertical distance between two objects
# Example: From Pacman to Pinky
# positive value == located in that direction
# negative value == opposite
# 0 value == off map, no proximity
def get_object_proximity(from_x, from_y, to_x, to_y):
	up_prox = 0.0
	down_prox = 0.0
	left_prox = 0.0
	right_prox = 0.0

	if (from_x == 0 and from_y == 0) or (to_x == 0 and to_y == 0):
		return up_prox, down_prox, left_prox, right_prox

	up_prox = get_measure_proximity(from_y, to_y)
	down_prox = -up_prox

	right_prox = get_measure_proximity(from_x, to_x)
	left_prox = -right_prox

	return up_prox, down_prox, left_prox, right_prox


# Avoid wall collisions
def get_legal_move(from_x, to_x, from_y, to_y):
	# 1.0 if yes, 0.0 if no
	global X_MAP
	global Y_MAP
	if from_x == to_x:
		y_vals = Y_MAP[from_x]
		for offset in range(10):
			if from_y < to_y:
				if to_y + offset in y_vals:
					return 1.0
			else:
				if to_y - offset in y_vals:
					return 1.0
		return 0.0
	elif from_y == to_y:
		x_vals = X_MAP[from_y]
		for offset in range(10):
			if from_x < to_x:
				if to_x + offset in x_vals:
					return 1.0
			else:
				if to_x - offset in x_vals:
					return 1.0
		return 0.0


# Can Ms. Pac-Man move up without hitting a wall?
def is_legal_up(pacman_x, pacman_y):
	return get_legal_move(pacman_x, pacman_x, pacman_y, pacman_y - 1)


# Can Ms. Pac-Man move down without hitting a wall?
def is_legal_down(pacman_x, pacman_y):
	return get_legal_move(pacman_x, pacman_x, pacman_y, pacman_y + 1)


# Can Ms. Pac-Man move left without hitting a wall?
def is_legal_left(pacman_x, pacman_y):
	return get_legal_move(pacman_x, pacman_x + 1, pacman_y, pacman_y)


# Can Ms. Pac-Man move right without hitting a wall?
def is_legal_right(pacman_x, pacman_y):
	return get_legal_move(pacman_x, pacman_x - 1, pacman_y, pacman_y)


# Convert coordinate values into a tuple of floats
def get_current_cell(pacman_x, pacman_y):
	return (np.float32(pacman_x), np.float32(pacman_y))


# If Ms. Pac-Man eats a dot or energizer, remove it from appropriate variable(s)
def mark_explored(pacman_loc):
	global UNEXPLORED
	global ENERGIZERS
	index = np.where((UNEXPLORED == pacman_loc).all(axis=1))
	if len(index) == 1:
		UNEXPLORED = np.delete(UNEXPLORED, index, axis=0)
	index = np.where((ENERGIZERS == pacman_loc).all(axis=1))
	if len(index) == 1:
		ENERGIZERS = np.delete(ENERGIZERS, index, axis=0)


# How close is Ms. Pac-Man to the nearest uneaten dot?
def get_closest_dot_proximity(pacman_loc):
	global UNEXPLORED

	if len(UNEXPLORED) == 0:
		return 0.0, 0.0, 0.0, 0.0

	index = distance.cdist([pacman_loc], UNEXPLORED).argmin()
	closest = UNEXPLORED[index]
	return get_object_proximity(pacman_loc[0], closest[0], pacman_loc[1], closest[1])


# How close is Ms. Pac-Man to the nearest uneaten energizer?
def get_closest_energizer_proximity(pacman_loc):
	global ENERGIZERS

	if len(ENERGIZERS) == 0:
		return 0.0, 0.0, 0.0, 0.0

	index = distance.cdist([pacman_loc], ENERGIZERS).argmin()
	closest = ENERGIZERS[index]
	return get_object_proximity(pacman_loc[0], closest[0], pacman_loc[1], closest[1])


# How close is Ms. Pac-Man to the maze center?
def get_center_proximity(pacman_loc):
	if pacman_loc[0] == 0.0 and pacman_loc[1] == 0.0:
		return 0.0, 0.0, 0.0, 0.0
	return get_object_proximity(pacman_loc[0], CENTER_X, pacman_loc[1], CENTER_Y)


# Convert each MAMEToolkit.emulator.Action into an integer (needed for model)
def fix_actions(actions):
    fixed_actions = []
    for action in actions:
        if type(action) == list and len(action) == 1 and type(action[0] == Action):
            if action[0].field == "P1 Left":
                fixed_actions.append(1)
            if action[0].field == "P1 Up":
                fixed_actions.append(2)
            if action[0].field == "P1 Right":
                fixed_actions.append(3)
            if action[0].field == "P1 Down":
                fixed_actions.append(4)
        elif type(action) == list and len(action) == 0:
            fixed_actions.append(0)
        else:
            fixed_actions.append(action)
    return fixed_actions
