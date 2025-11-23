import sys
from typing import Any, Set

def deep_size(obj: Any, seen: Set[int] | None = None) -> int:

    # approximate total memory usage
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    size = sys.getsizeof(obj) #size of container of object

    if isinstance(obj, dict):
        for k, v in obj.items():
            size += deep_size(k, seen)
            size += deep_size(v, seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            size += deep_size(item, seen)

    return size

def format_bytes(num_bytes: int) -> str:
    
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} TB"
