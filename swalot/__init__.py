from .gpu import MemoryProtector, AutoMemoryProtector
from .time import TimeController

def gpu(remain=256, device=0):
    protector = MemoryProtector(remain, device)
    return AutoMemoryProtector(protector)

def time(speed=1.5, offset=0, random=False, seed=None):
    return TimeController(speed, random, seed)