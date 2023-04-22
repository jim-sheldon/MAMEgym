# From https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Second-Edition/blob/master/Chapter06/02_dqn_pong.py

import copy
import os
from types import SimpleNamespace

import numpy as np
from ignite.engine import Engine
import torch
import torch.optim as optim

# from dqn import DQN
from .dqn_experience import ExperienceFirstLast, ExperienceSourceFirstLast, ExperienceReplayBuffer
from ..model.rainbow import RainbowDQN
from ..util.torch_ignite import setup_ignite


HYPERPARAMS = {
    'mspac': SimpleNamespace(**{
        'env_name':         os.environ.get("ENV_NAME"),
        'stop_reward':      int(os.environ.get("REWARD_BOUND")),
        'run_name':         os.environ.get("ENV_NAME"),
        'replay_size':      int(os.environ.get("REPLAY_SIZE")),
        'replay_initial':   int(os.environ.get("REPLAY_START_SIZE")),
        'target_net_sync':  int(os.environ.get("SYNC_TARGET_FRAMES")),
        'epsilon_frames':   int(os.environ.get("EPSILON_DECAY_LAST_FRAME")),
        'epsilon_start':    float(os.environ.get("EPSILON_START")),
        'epsilon_final':    float(os.environ.get("EPSILON_FINAL")),
        'learning_rate':    float(os.environ.get("LEARNING_RATE")),
        'gamma':            float(os.environ.get("GAMMA")),
        'batch_size':       int(os.environ.get("BATCH_SIZE")),
        'n_steps':          int(os.environ.get("N_STEPS")),
        'pr_replay_alpha':  float(os.environ.get("PRIO_REPLAY_ALPHA"))
    })
}


# replay buffer params
BETA_START = float(os.environ.get("BETA_START"))
BETA_FRAMES = int(os.environ.get("BETA_FRAMES"))


class PrioReplayBuffer:
    def __init__(self, exp_source, buf_size, prob_alpha=0.6):
        self.exp_source_iter = iter(exp_source)
        self.prob_alpha = prob_alpha
        self.capacity = buf_size
        self.pos = 0
        self.buffer = []
        self.priorities = np.zeros(
            (buf_size, ), dtype=np.float32)
        self.beta = BETA_START

    def update_beta(self, idx):
        v = BETA_START + idx * (1.0 - BETA_START) / \
            BETA_FRAMES
        self.beta = min(1.0, v)
        return self.beta

    def __len__(self):
        return len(self.buffer)

    def populate(self, count):
        max_prio = self.priorities.max() if \
            self.buffer else 1.0
        for _ in range(count):
            sample = next(self.exp_source_iter)
            if len(self.buffer) < self.capacity:
                self.buffer.append(sample)
            else:
                self.buffer[self.pos] = sample
            self.priorities[self.pos] = max_prio
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = prios ** self.prob_alpha

        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer),
                                   batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        return samples, indices, \
               np.array(weights, dtype=np.float32)

    def update_priorities(self, batch_indices,
                          batch_priorities):
        for idx, prio in zip(batch_indices,
                             batch_priorities):
            self.priorities[idx] = prio


class ActionSelector:
    """
    Abstract class which converts scores to the actions
    """
    def __call__(self, scores):
        raise NotImplementedError


class ArgmaxActionSelector(ActionSelector):
    """
    Selects actions using argmax
    """
    def __call__(self, scores):
        # assert isinstance(scores, np.ndarray)
        return np.argmax(scores, axis=1)


class BaseAgent:
    """
    Abstract Agent interface
    """
    def initial_state(self):
        """
        Should create initial empty state for the agent. It will be called for the start of the episode
        :return: Anything agent want to remember
        """
        return None

    def __call__(self, states, agent_states):
        """
        Convert observations and states into actions to take
        :param states: list of environment states to process
        :param agent_states: list of states with the same length as observations
        :return: tuple of actions, states
        """
        assert isinstance(states, list)
        assert isinstance(agent_states, list)
        assert len(agent_states) == len(states)

        raise NotImplementedError


def default_states_preprocessor(states):
    """
    Convert list of states into the form suitable for model. By default we assume Variable
    :param states: list of numpy arrays with states
    :return: Variable
    """
    if len(states) == 1:
        np_states = np.expand_dims(states[0], 0)
    else:
        np_states = np.array([np.array(s, copy=False) for s in states], copy=False)
    return torch.tensor(np_states)


def float32_preprocessor(states):
    np_states = np.array(states, dtype=np.float32)
    return torch.tensor(np_states)


class DQNAgent(BaseAgent):
    """
    DQNAgent is a memoryless DQN agent which calculates Q values
    from the observations and  converts them into the actions using action_selector
    """
    def __init__(self, dqn_model, action_selector, device="cpu", preprocessor=default_states_preprocessor):
        self.dqn_model = dqn_model
        self.action_selector = action_selector
        self.preprocessor = preprocessor
        self.device = device

    @torch.no_grad()
    def __call__(self, states, agent_states=None):
        if agent_states is None or agent_states == [None]:
            agent_states = [None, None, None, None]
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)

        q_v = self.dqn_model(states)
        q = q_v.data.cpu().numpy()

        actions = self.action_selector(q)

        return actions, agent_states


class TargetNet:
    """
    Wrapper around model which provides copy of it instead of trained weights
    """
    def __init__(self, model):
        print(f"Target network init")
        self.model = model
        self.target_model = copy.deepcopy(model)

    def sync(self):
        print(f"Target network sync")
        self.target_model.load_state_dict(self.model.state_dict())

    def alpha_sync(self, alpha):
        """
        Blend params of target net with params from the model
        :param alpha:
        """
        state = self.model.state_dict()
        tgt_state = self.target_model.state_dict()
        for k, v in state.items():
            tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
        self.target_model.load_state_dict(tgt_state)


def unpack_batch(batch: list[ExperienceFirstLast]):
    states, actions, rewards, dones, last_states = [],[],[],[],[]
    for exp in batch:
        state = np.array(exp.state)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            lstate = state  # the result will be masked anyway
        else:
            lstate = np.array(exp.last_state)
        last_states.append(lstate)
    return np.array(states, copy=False), np.array(actions), \
           np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), \
           np.array(last_states, copy=False)


def calc_loss(batch, batch_weights, net, tgt_net, gamma,
                      device="cpu", double=True):
    states, actions, rewards, dones, next_states = unpack_batch(batch)

    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)
    batch_weights_v = torch.tensor(batch_weights).to(device)

    actions_v = actions_v.unsqueeze(-1)
    state_action_values = net(states_v).gather(1, actions_v)
    state_action_values = state_action_values.squeeze(-1)
    with torch.no_grad():
        next_states_v = torch.tensor(next_states).to(device)
        if double:
            next_state_actions = net(next_states_v).max(1)[1]
            next_state_actions = next_state_actions.unsqueeze(-1)
            next_state_values = tgt_net(next_states_v).gather(
                1, next_state_actions).squeeze(-1)
        else:
            next_state_values = tgt_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        expected_state_action_values = \
            next_state_values.detach() * gamma + rewards_v
    losses_v = (state_action_values -
                expected_state_action_values) ** 2
    losses_v *= batch_weights_v
    return losses_v.mean(), (losses_v + 1e-5).data.cpu().numpy()


def calc_loss_prio(batch, batch_weights, net, tgt_net, func, gamma, device="cpu"):
    states, actions, rewards, dones, next_states = unpack_batch(batch)

    fixed_actions = func(actions)

    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(fixed_actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)
    batch_weights_v = torch.tensor(batch_weights).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_states_v = torch.tensor(next_states).to(device)
        next_state_values = tgt_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        expected_state_action_values = next_state_values.detach() * gamma + rewards_v
    losses_v = batch_weights_v * (state_action_values - expected_state_action_values) ** 2
    return losses_v.mean(), (losses_v + 1e-5).data.cpu().numpy()


def batch_generator(buffer: ExperienceReplayBuffer,
                    initial: int, batch_size: int):
    buffer.populate(initial)
    while True:
        buffer.populate(1)
        yield buffer.sample(batch_size)


def run_agent(env, device, observation_space_n, actions, functions):
    observation_space_shape = (4, observation_space_n)    

    params = HYPERPARAMS.get("mspac")

    net = RainbowDQN(observation_space_shape, len(actions)).to(device)
    tgt_net = TargetNet(net)

    selector = ArgmaxActionSelector()
    agent = DQNAgent(net, selector, device=device)

    exp_source = ExperienceSourceFirstLast(
        env, agent, actions, gamma=params.gamma, steps_count=params.n_steps
    )

    exp_source.reset = functions["reset"]
    exp_source.get_reward = functions["get_reward"]
    exp_source.get_frame_data = functions["get_frame_data"]
    exp_source.check_done = functions["check_done"]

    buffer = PrioReplayBuffer(exp_source, params.replay_size, params.pr_replay_alpha)

    optimizer = optim.Adam(net.parameters(), lr=params.learning_rate)

    def process_batch(engine, batch_data):
        batch, batch_indices, batch_weights = batch_data
        optimizer.zero_grad()
        loss_v, sample_prios = calc_loss_prio(
            batch, batch_weights, net, tgt_net.target_model, functions["fix_actions"],
            gamma=params.gamma**params.n_steps, device=device
        )
        loss_v.backward()
        optimizer.step()
        buffer.update_priorities(batch_indices, sample_prios)
        if engine.state.iteration % params.target_net_sync == 0:
            tgt_net.sync()
        return {
            "loss": loss_v.item(),
            "beta": buffer.update_beta(engine.state.iteration),
        }

    engine = Engine(process_batch)
    setup_ignite(engine, params, exp_source, params.env_name)
    engine.run(batch_generator(buffer, params.replay_start_size, params.batch_size))
