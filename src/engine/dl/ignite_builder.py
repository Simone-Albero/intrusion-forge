from collections.abc import Callable

from ignite.engine import Engine, Events
from ignite.metrics import Metric


def build_engine(
    step_fn: Callable,
    *,
    state: dict[str, object],
    metric: tuple[str, Metric],
    handlers: list[tuple[Events, Callable[[Engine], None]]],
) -> Engine:
    """Build an Ignite engine: inject `state` on start, attach `metric`, attach `handlers`.

    State is injected on Events.STARTED rather than set directly, because `Engine.run()`
    replaces `engine.state` with a fresh one at the start of every run.
    """
    engine = Engine(step_fn)

    def _inject_state(engine: Engine) -> None:
        for key, value in state.items():
            setattr(engine.state, key, value)

    engine.add_event_handler(Events.STARTED, _inject_state)
    name, m = metric
    m.attach(engine, name)
    for event, handler in handlers:
        engine.add_event_handler(event, handler)
    return engine
