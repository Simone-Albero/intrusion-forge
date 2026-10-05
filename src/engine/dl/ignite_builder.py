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
    """Build an Ignite engine with `state` set on start, `metric` and `handlers`."""
    engine = Engine(step_fn)

    def _inject_state(engine: Engine) -> None:
        for key, value in state.items():
            setattr(engine.state, key, value)

    engine.add_event_handler(Events.STARTED, _inject_state)
    metric_name, metric_impl = metric
    metric_impl.attach(engine, metric_name)
    for event, handler in handlers:
        engine.add_event_handler(event, handler)
    return engine
