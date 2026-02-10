"""RAT selection API, algorithms, and file-based integration."""

def __getattr__(name):
    if name in ("RATSelectionAPI", "JointController"):
        from selection.api import RATSelectionAPI, JointController
        return {"RATSelectionAPI": RATSelectionAPI, "JointController": JointController}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
