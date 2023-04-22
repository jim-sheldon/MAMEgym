from enum import Enum

from MAMEToolkit.emulator import Action


# An enumerable class used to specify which actions can be used to interact with a game
# Specifies the Lua engine port and field names required for performing an action
class Actions(Enum):
    # Starting
    SERVICE =   Action(':IN0', 'Service 1')
    COIN_P1 =   Action(':IN0', 'Coin 1')
    P1_START =  Action(':IN1', '1 Player Start')

    # Movement
    P1_UP =     Action(':IN0', 'P1 Up')
    P1_DOWN =   Action(':IN0', 'P1 Down')
    P1_LEFT =   Action(':IN0', 'P1 Left')
    P1_RIGHT =  Action(':IN0', 'P1 Right')
