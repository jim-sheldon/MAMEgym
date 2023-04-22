# From: https://github.com/Shmuma/ptan/blob/master/ptan/ignite.py

from datetime import timedelta, datetime
import enum
import time
from types import SimpleNamespace
from typing import Optional, Iterable, Tuple, List
import warnings

from ignite.engine import Engine, State
from ignite.engine import Events, EventEnum
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import tensorboard_logger as tb_logger
from ignite.handlers.timing import Timer

from experience import ExperienceSource


class EpisodeEvents(EventEnum):
    EPISODE_COMPLETED = "episode_completed"
    BOUND_REWARD_REACHED = "bound_reward_reached"
    BEST_REWARD_REACHED = "best_reward_reached"


class EndOfEpisodeHandler:
    def __init__(self, exp_source: ExperienceSource, alpha: float = 0.98,
                 bound_avg_reward: Optional[float] = None,
                 subsample_end_of_episode: Optional[int] = None):
        """
        Construct end-of-episode event handler
        :param exp_source: experience source to use
        :param alpha: smoothing alpha param
        :param bound_avg_reward: optional boundary for average reward
        :param subsample_end_of_episode: if given, end of episode event will be subsampled by this amount
        """
        self._exp_source = exp_source
        self._alpha = alpha
        self._bound_avg_reward = bound_avg_reward
        self._best_avg_reward = None
        self._subsample_end_of_episode = subsample_end_of_episode

    def attach(self, engine: Engine):
        engine.add_event_handler(Events.ITERATION_COMPLETED, self)
        engine.register_events(*EpisodeEvents)
        State.event_to_attr[EpisodeEvents.EPISODE_COMPLETED] = "episode"
        State.event_to_attr[EpisodeEvents.BOUND_REWARD_REACHED] = "episode"
        State.event_to_attr[EpisodeEvents.BEST_REWARD_REACHED] = "episode"

    def __call__(self, engine: Engine):
        for reward, steps in self._exp_source.pop_rewards_steps():
            engine.state.episode = getattr(engine.state, "episode", 0) + 1
            engine.state.episode_reward = reward
            engine.state.episode_steps = steps
            engine.state.metrics['reward'] = reward
            engine.state.metrics['steps'] = steps
            self._update_smoothed_metrics(engine, reward, steps)
            if self._subsample_end_of_episode is None or engine.state.episode % self._subsample_end_of_episode == 0:
                engine.fire_event(EpisodeEvents.EPISODE_COMPLETED)
            if self._bound_avg_reward is not None and engine.state.metrics['avg_reward'] >= self._bound_avg_reward:
                engine.fire_event(EpisodeEvents.BOUND_REWARD_REACHED)
            if self._best_avg_reward is None:
                self._best_avg_reward = engine.state.metrics['avg_reward']
            elif self._best_avg_reward < engine.state.metrics['avg_reward']:
                engine.fire_event(EpisodeEvents.BEST_REWARD_REACHED)
                self._best_avg_reward = engine.state.metrics['avg_reward']

    def _update_smoothed_metrics(self, engine: Engine, reward: float, steps: int):
        for attr_name, val in zip(('avg_reward', 'avg_steps'), (reward, steps)):
            if attr_name not in engine.state.metrics:
                engine.state.metrics[attr_name] = val
            else:
                engine.state.metrics[attr_name] *= self._alpha
                engine.state.metrics[attr_name] += (1-self._alpha) * val


class EpisodeFPSHandler:
    FPS_METRIC = 'fps'
    AVG_FPS_METRIC = 'avg_fps'
    TIME_PASSED_METRIC = 'time_passed'

    def __init__(self, fps_mul: float = 1.0, fps_smooth_alpha: float = 0.98):
        self._timer = Timer(average=True)
        self._fps_mul = fps_mul
        self._started_ts = time.time()
        self._fps_smooth_alpha = fps_smooth_alpha

    def attach(self, engine: Engine, manual_step: bool = False):
        self._timer.attach(engine, step=None if manual_step else Events.ITERATION_COMPLETED)
        engine.add_event_handler(EpisodeEvents.EPISODE_COMPLETED, self)

    def step(self):
        """
        If manual_step=True on attach(), this method should be used every time we've communicated with environment
        to get proper FPS
        :return:
        """
        self._timer.step()

    def __call__(self, engine: Engine):
        t_val = self._timer.value()
        if engine.state.iteration > 1:
            fps = self._fps_mul / t_val
            avg_fps = engine.state.metrics.get(self.AVG_FPS_METRIC)
            if avg_fps is None:
                avg_fps = fps
            else:
                avg_fps *= self._fps_smooth_alpha
                avg_fps += (1-self._fps_smooth_alpha) * fps
            engine.state.metrics[self.AVG_FPS_METRIC] = avg_fps
            engine.state.metrics[self.FPS_METRIC] = fps
        engine.state.metrics[self.TIME_PASSED_METRIC] = time.time() - self._started_ts
        self._timer.reset()


class PeriodEvents(EventEnum):
    ITERS_10_COMPLETED = "iterations_10_completed"
    ITERS_100_COMPLETED = "iterations_100_completed"
    ITERS_1000_COMPLETED = "iterations_1000_completed"
    ITERS_10000_COMPLETED = "iterations_10000_completed"
    ITERS_100000_COMPLETED = "iterations_100000_completed"


class PeriodicEvents:
    """
    The same as CustomPeriodicEvent from ignite.contrib, but use true amount of iterations,
    which is good for TensorBoard
    """

    INTERVAL_TO_EVENT = {
        10: PeriodEvents.ITERS_10_COMPLETED,
        100: PeriodEvents.ITERS_100_COMPLETED,
        1000: PeriodEvents.ITERS_1000_COMPLETED,
        10000: PeriodEvents.ITERS_10000_COMPLETED,
        100000: PeriodEvents.ITERS_100000_COMPLETED,
    }

    def attach(self, engine: Engine):
        engine.add_event_handler(Events.ITERATION_COMPLETED, self)
        engine.register_events(*PeriodEvents)
        for e in PeriodEvents:
            State.event_to_attr[e] = "iteration"

    def __call__(self, engine: Engine):
        for period, event in self.INTERVAL_TO_EVENT.items():
            print(f"Period event")
            print(f"period: {period}, event: {event}")
            if engine.state.iteration % period == 0:
                engine.fire_event(event)


def setup_ignite(engine: Engine, params: SimpleNamespace,
                 exp_source, run_name: str,
                 extra_metrics: Iterable[str] = ()):
    # get rid of missing metrics warning
    warnings.simplefilter("ignore", category=UserWarning)

    handler = EndOfEpisodeHandler(
        exp_source, bound_avg_reward=params.stop_reward)
    handler.attach(engine)
    EpisodeFPSHandler().attach(engine)

    @engine.on(EpisodeEvents.EPISODE_COMPLETED)
    def episode_completed(trainer: Engine):
        passed = trainer.state.metrics.get('time_passed', 0)
        print("Episode %d: reward=%.0f, steps=%s, "
              "speed=%.1f f/s, elapsed=%s" % (
            trainer.state.episode, trainer.state.episode_reward,
            trainer.state.episode_steps,
            trainer.state.metrics.get('avg_fps', 0),
            timedelta(seconds=int(passed))))

    @engine.on(EpisodeEvents.BOUND_REWARD_REACHED)
    def game_solved(trainer: Engine):
        passed = trainer.state.metrics['time_passed']
        print("Game solved in %s, after %d episodes "
              "and %d iterations!" % (
            timedelta(seconds=int(passed)),
            trainer.state.episode, trainer.state.iteration))
        trainer.should_terminate = True

    now = datetime.now().isoformat(timespec='minutes').replace(':', '')
    logdir = f"runs/{now}-{params.run_name}-{run_name}"
    tb = tb_logger.TensorboardLogger(log_dir=logdir)
    run_avg = RunningAverage(output_transform=lambda v: v['loss'])
    run_avg.attach(engine, "avg_loss")

    metrics = ['reward', 'steps', 'avg_reward']
    handler = tb_logger.OutputHandler(
        tag="episodes", metric_names=metrics)
    event = EpisodeEvents.EPISODE_COMPLETED
    tb.attach(engine, log_handler=handler, event_name=event)

    # write to tensorboard every 100 iterations
    PeriodicEvents().attach(engine)
    metrics = ['avg_loss', 'avg_fps']
    metrics.extend(extra_metrics)
    handler = tb_logger.OutputHandler(
        tag="train", metric_names=metrics,
        output_transform=lambda a: a)
    event = PeriodEvents.ITERS_100_COMPLETED
    tb.attach(engine, log_handler=handler, event_name=event)

    print("Ignite setup done")
