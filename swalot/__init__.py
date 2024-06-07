from .swalot import MemoryProtector, AutoMemoryProtector

def protect(remain=256, device=0):
    protector = MemoryProtector(remain, device)
    return AutoMemoryProtector(protector)