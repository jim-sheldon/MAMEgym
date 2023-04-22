# MAME Gymnasium

This repository exists to combine the [MAMEToolkit](https://github.com/M-J-Murray/MAMEToolkit), and models from the book [Deep Reinforcement Learning Hands-On: Second Edition](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition) into one library to facilitate building and training RL agents for arcade games.
Those authors did excellent work, and I encourage supporting them.

## Installation

Dependencies work in Linux, but not in Mac OS.

To get up and running, run the following:
```bash
pip install -r requirements.txt
```

## Usage

Users need to do a bit of setup to use the provided agents.

### Defining agent actions

To let an agent send actions to the environment, users must create an `Enum` containing action names and their corresponding input ports and fields.
 For example, starting Pac-Man requires inserting a coin and pushing the start button:
```python
from enum import Enum
from MAMEToolkit.emulator import Action

class StartActions(Enum):
    COIN_P1 =   Action(':IN0', 'Coin 1')
    P1_START =  Action(':IN1', '1 Player Start')
```

And moving in Pac-Man requires pushing the joystick in a cardinal direction:
```python
from enum import Enum
from MAMEToolkit.emulator import Action

class GameActions(Enum):
    P1_UP =     Action(':IN0', 'P1 Up')
    P1_DOWN =   Action(':IN0', 'P1 Down')
    P1_LEFT =   Action(':IN0', 'P1 Left')
    P1_RIGHT =  Action(':IN0', 'P1 Right')
```

To see all available actions:
```python
from MAMEToolkit.emulator import list_actions

print(list_actions(PATH_TO_ROM, GAME_ID))
```


### Defining agent observations

Providing values from a game requires mapping labels to memory addresses (including data type).  For example, to observe X and Y coordinate values for Pac-Man, ghosts, and bonus fruit:
```python
from MAMEToolkit.emulator import Address

OBSERVATIONS = {
    "fruit_x": Address('0x4dd2', 'u8'),
    "fruit_y": Address('0x4dd2', 'u8'),
    "blinky_x": Address('0x4d00', 'u8'),
    "blinky_y": Address('0x4d01', 'u8'),
    "pinky_x": Address('0x4d02', 'u8'),
    "pinky_y": Address('0x4d03', 'u8'),
    "inky_x": Address('0x4d04', 'u8'),
    "inky_y": Address('0x4d05', 'u8'),
    "sue_x": Address('0x4d06', 'u8'),
    "sue_y": Address('0x4d07', 'u8'),
    "pacman_x": Address('0x4d08', 'u8'),
    "pacman_y": Address('0x4d09', 'u8')
}
```

And to access the score:
```python
from MAMEToolkit.emulator import Address

# To concatenate later
SCORE = {
    "score_a": Address('0x4e80', 's8'),
    "score_b": Address('0x4e81', 's8'),
    "score_c": Address('0x4e82', 's8')
}
```

### Defining state/frame data

Using values from a game as an agent's inputs requires retrieving them from emulator data.
For example, to get Pac-Man's coordinates:
```python
from MAMEToolkit.emulator import Emulator

emulator = Emulator("env1", PATH_TO_ROM, GAME_ID, OBSERVATIONS)
# No action taken
data = emulator.step([])
pacman_x = data.get("pacman_x")
pacman_y = data.get("pacman_y")
print(f"Pac-Man coordinates: ({pacman_x},{pacman_y})")
```

### Setting hyperparameters 

A user should set model parameter values via environment variables per [Twelve-Factor App guidelines](https://12factor.net/config).
A user should store them in a `.env` file and use them by running `export $(grep -v '^#' ENV_FILE_NAME | xargs)` or a library like [`dotenv`](https://github.com/theskumar/python-dotenv).

### Writing a reward function

A user must define positive and/or negative reinforcement for the agent.
For example, to access the score in Pac-Man:
```python
from MAMEToolkit.emulator import Emulator

emulator = Emulator("env1", PATH_TO_ROM, GAME_ID, OBSERVATIONS)
# No action taken
data = emulator.step([])

a = data["score_a"]
b = data["score_b"]
c = data["score_c"]
score = int("".join([hex(x).replace("0x", "").replace("-", "") for x in [c, b, a]]))
print(f"Game score: {score}")
```

### Writing an environment reset function

A user can reset the environment using `emulator.console.writeln("manager:machine():soft_reset()")`.
Combining a reset with starting a new episode depends on game-specific timing and actions.
For more, see the example Pac-Man function.

### Writing a "fix action" function

Models in the library use integer values to represent actions taken; users must convert each `MAMEToolkit.emulator.Action` into an integer
For example, to do so for Pac-Man:
```python
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
```

### Writing a data retrieval/update function

A user must provide model data as a `numpy` array containing `numpy.float32` values. This may involve engineering features from values retrieved from memory. The Ms. Pac-Man example contains a function that returns many values for the model, some from memory, others calculated.

### Writing an episode finish check function

A user must define when an episode ends. In the Ms. Pac-Man example, if the agent loses a life or completes a level, the episode ends.


## Examples

A Ms. Pac-Man example exists in `example`, showing how an agent is created, configured, and used.
