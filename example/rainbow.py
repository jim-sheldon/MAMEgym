import collections

from MAMEToolkit.emulator import Emulator
from MAMEToolkit.emulator import Address
import numpy as np
import torch

from ..agent.rainbow import run_agent
from ..environment.actions import Actions
from ..util.logger import setup_logger
from .pacman_aux import fix_actions

from .pacman_aux import (
    load_legal_spaces, load_dots, load_x_map, load_y_map,
    get_object_proximity, is_legal_up, is_legal_down,
    is_legal_left, is_legal_right, mark_explored, get_closest_dot_proximity,
    get_closest_energizer_proximity, get_center_proximity, get_current_cell
)

roms_path = "./"
game_id = "mspacman"

# From https://github.com/BleuLlama/GameDocs/blob/master/disassemble/mspac.asm
OBSERVATIONS = {
    "fruit_x": Address('0x4dd2', 'u8'),
    "fruit_y": Address('0x4dd2', 'u8'),
    "fruit_value": Address('0x4dd4', 'u8'),
    "dots_eaten": Address('0x4e0e', 'u8'),
    "blinky_x": Address('0x4d00', 'u8'),
    "blinky_y": Address('0x4d01', 'u8'),
    "pinky_x": Address('0x4d02', 'u8'),
    "pinky_y": Address('0x4d03', 'u8'),
    "inky_x": Address('0x4d04', 'u8'),
    "inky_y": Address('0x4d05', 'u8'),
    "sue_x": Address('0x4d06', 'u8'),
    "sue_y": Address('0x4d07', 'u8'),
    "pacman_x": Address('0x4d08', 'u8'),
    "pacman_y": Address('0x4d09', 'u8'),
    "blinky_flag": Address('0x4da7', 'u8'),
    "pinky_flag": Address('0x4da8', 'u8'),
    "inky_flag": Address('0x4da9', 'u8'),
    "sue_flag": Address('0x4daa', 'u8')
}

MEMORY_ADDRESSES = OBSERVATIONS.copy()
MEMORY_ADDRESSES.extend({
    "score_a": Address('0x4e80', 's8'),
    "score_b": Address('0x4e81', 's8'),
    "score_c": Address('0x4e82', 's8')
})

ACTIONS = [
    [],
    [Actions.P1_LEFT.value],
    [Actions.P1_UP.value],
    [Actions.P1_RIGHT.value],
    [Actions.P1_DOWN.value]
]

EMPTY_STATE = np.zeros(64)

NEW_START = True
START_LEVEL = 0
Score = 0


def get_frame_data(data):
    pacman_x = data.get("pacman_x")
    pacman_y = data.get("pacman_y")
    pac_loc = get_current_cell(pacman_x, pacman_y)

    u_center, d_center, l_center, r_center = get_center_proximity(pac_loc)
    u_dot, d_dot, l_dot, r_dot = get_closest_dot_proximity(pac_loc)
    u_pp, d_pp, l_pp, r_pp = get_closest_energizer_proximity(pac_loc)
    legal_up = is_legal_up(pacman_x, pacman_y)
    legal_down = is_legal_down(pacman_x, pacman_y)
    legal_left = is_legal_left(pacman_x, pacman_y)
    legal_right = is_legal_right(pacman_x, pacman_y)

    blinky_x = data.get("blinky_x")
    blinky_y = data.get("blinky_y")
    pinky_x = data.get("pinky_x")
    pinky_y = data.get("pinky_y")
    inky_x = data.get("inky_x")
    inky_y = data.get("inky_y")
    sue_x = data.get("sue_x")
    sue_y = data.get("sue_y")

    u_mb, d_mb, l_mb, r_mb = get_object_proximity(pacman_x, pacman_y, blinky_x, blinky_y)
    u_mp, d_mp, l_mp, r_mp = get_object_proximity(pacman_x, pacman_y, pinky_x, pinky_y)
    u_mi, d_mi, l_mi, r_mi = get_object_proximity(pacman_x, pacman_y, inky_x, inky_y)
    u_ms, d_ms, l_ms, r_ms = get_object_proximity(pacman_x, pacman_y, sue_x, sue_y)

    u_bp, d_bp, l_bp, r_bp = get_object_proximity(blinky_x, blinky_y, pinky_x, pinky_y)
    u_bi, d_bi, l_bi, r_bi = get_object_proximity(blinky_x, blinky_y, inky_x, inky_y)
    u_bs, d_bs, l_bs, r_bs = get_object_proximity(blinky_x, blinky_y, sue_x, sue_y)
    u_pi, d_pi, l_pi, r_pi = get_object_proximity(pinky_x, pinky_y, inky_x, inky_y)
    u_ps, d_ps, l_ps, r_ps = get_object_proximity(pinky_x, pinky_y, sue_x, sue_y)
    u_is, d_is, l_is, r_is = get_object_proximity(inky_x, inky_y, sue_x, sue_y)

    fruit_x = data.get("fruit_x")
    fruit_y = data.get("fruit_y")

    u_mf, d_mf, l_mf, r_mf = get_object_proximity(pacman_x, pacman_y, fruit_x, fruit_y)

    pacman_x = data.get("pacman_x")
    pacman_y = data.get("pacman_y")
    pacman_loc = get_current_cell(pacman_x, pacman_y)
    mark_explored(pacman_loc)

    return np.array([
        u_center,
        d_center,
        l_center,
        r_center,
        u_dot,
        d_dot,
        l_dot,
        r_dot,
        u_pp,
        d_pp,
        l_pp,
        r_pp,
        legal_up,
        legal_down,
        legal_left,
        legal_right,
        u_mb,
        d_mb,
        l_mb,
        r_mb,
        u_mp,
        d_mp,
        l_mp,
        r_mp,
        u_mi,
        d_mi,
        l_mi,
        r_mi,
        u_ms,
        d_ms,
        l_ms,
        r_ms,
        u_bp,
        d_bp,
        l_bp,
        r_bp,
        u_bi,
        d_bi,
        l_bi,
        r_bi,
        u_bs,
        d_bs,
        l_bs,
        r_bs,
        u_pi,
        d_pi,
        l_pi,
        r_pi,
        u_ps,
        d_ps,
        l_ps,
        r_ps,
        u_is,
        d_is,
        l_is,
        r_is,
        u_mf,
        d_mf,
        l_mf,
        r_mf,
        data.get("blinky_flag"),
        data.get("pinky_flag"),
        data.get("inky_flag"),
        data.get("sue_flag")
    ]).astype(np.float32)


def get_reward(data):
    global Score
    reward = 0
    positive = convert_score(data["score_a"], data["score_b"], data["score_c"])
    if positive > Score:
        reward = positive - Score
        Score = positive
    return reward


def convert_score(a, b, c):
    try:
        return int("".join([hex(x).replace("0x", "").replace("-", "") for x in [c, b, a]]))
    except ValueError:
        print(f'Score (dec): {"|".join([str(x) for x in [c, b, a]])}')
        print(f'Score (hex): {"|".join([hex(x) for x in [c, b, a]])}')
        raise


def check_done(player_dead, current_level):
    if player_dead:
        return True
    if START_LEVEL != current_level:
        return True
    return False


def wait_for_reset(emulator):
    for _ in range(200):
        data = emulator.step([])
        if data["pacman_x"] == 196 and data["pacman_y"] == 128:
            for _ in range(38):
                _ = emulator.step([])
            break


def reset(emulator):
    global NEW_START
    global Score
    load_legal_spaces()
    load_dots()
    load_x_map()
    load_y_map()
    state = collections.deque([EMPTY_STATE for _ in range(4)], maxlen=4)
    if NEW_START:
        NEW_START = False
        for _ in range(38):
            _ = emulator.step([])
    else:
        emulator.console.writeln("manager:machine():soft_reset()")
        # janky but working
        for _ in range(200):
            _ = emulator.step([])
        _ = emulator.step([Actions.COIN_P1.value])
        _ = emulator.step([Actions.P1_START.value])
        Score = 0
        wait_for_reset(emulator)
    return state


FUNCTIONS = {
    "fix_actions": fix_actions,
    "get_frame_data": get_frame_data,
    "get_reward": get_reward,
    "check_done": check_done,
    "reset": reset
}

if __name__ == "__main__":
    setup_logger()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emulator = Emulator("env1", roms_path, game_id, MEMORY_ADDRESSES)
    run_agent(emulator, device, len(OBSERVATIONS), ACTIONS, FUNCTIONS)
