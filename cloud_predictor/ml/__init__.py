from .schema import Band, BreakupEvent, BreakupParameters, Impactor, Spacecraft

__all__ = [
    "Band",
    "BreakupEvent",
    "BreakupParameterEstimator",
    "BreakupParameters",
    "Impactor",
    "Spacecraft",
]


def __getattr__(name):
    # Lazy so that numpy-only consumers (engine/) don't need lightgbm installed.
    if name == "BreakupParameterEstimator":
        from .estimator import BreakupParameterEstimator
        return BreakupParameterEstimator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
