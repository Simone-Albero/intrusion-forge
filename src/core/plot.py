from dataclasses import dataclass


@dataclass
class Plot:
    """Immutable, self-contained figure payload.

    Carries the rendered bytes and their format; produced by the plotting
    helpers in `src.domain.plot` and routed to subscribers via `LogBundle`.
    """

    data: bytes
    format: str = "png"
