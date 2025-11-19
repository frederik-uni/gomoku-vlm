import threading
import random

_thread_local = threading.local()


def get_random() -> random.Random:
    global _thread_local
    if not hasattr(_thread_local, "rng"):
        _thread_local.rng = random.Random()
    return _thread_local.rng
