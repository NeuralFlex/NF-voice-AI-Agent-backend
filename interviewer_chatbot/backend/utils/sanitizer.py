from typing import Any


def sanitize_state(obj: Any) -> Any:
    """Recursively convert all NumPy types to native Python types."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: sanitize_state(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_state(v) for v in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    return obj
