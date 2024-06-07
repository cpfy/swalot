import torch
import time
from rich import print
from functools import partial, update_wrapper

import swalot as sw

with sw.protect():
    # This allocation will be automatically managed by our protector
    # time.sleep(60)
    a = torch.randn(1000, 1000, 600).cuda()

    # Simulate some work
    print("sleeping 5s...")
    time.sleep(5)

    # Another allocation which might trigger auto-memory management
    b = torch.randn(1000, 1000, 600).cuda()

    for _ in range(100):
        print("sleeping 5s...")
        time.sleep(5)
        c = torch.randn(1000, 1000, 100).cuda()
        c1 = torch.randn(1000, 1000, 100).cuda()
        c2 = torch.randn(1000, 1000, 100).cuda()
        c3 = torch.randn(1000, 1000, 100).cuda()
        c4 = torch.randn(1000, 1000, 100).cuda()
        c5 = torch.randn(1000, 1000, 100).cuda()
        c6 = torch.randn(1000, 1000, 100).cuda()
        c7 = torch.randn(1000, 1000, 100).cuda()
        c8 = torch.randn(1000, 1000, 100).cuda()
        c9 = torch.randn(1000, 1000, 100).cuda()
        c10 = torch.randn(1000, 1000, 100).cuda()
        c11 = torch.randn(1000, 1000, 100).cuda()
        c12 = torch.randn(1000, 1000, 100).cuda()
