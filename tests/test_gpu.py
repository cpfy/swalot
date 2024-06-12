import torch
from rich import print
import swalot as sw
import pytest
from pytest import approx
from GPUtil import getGPUs

DEBUG = True
if DEBUG:
    from icecream import ic


def test_full_torchCheck():
    with sw.gpu(remain=256):
        allocated_MB = torch.cuda.memory_allocated(0) / 1024**2
        cached_MB = torch.cuda.memory_reserved(0) / 1024**2
        total_MB = torch.cuda.get_device_properties(0).total_memory / 1024**2
        free_MB = total_MB - allocated_MB - cached_MB
        assert free_MB <= 256


def test_full_GPUtilCheck():
    with sw.gpu(remain=256):
        gpu = getGPUs()[0]
        assert gpu.memoryFree <= 256

"""
BUG: The following testcases are not working as expected. Got CUDA OOM in real test
"""

def test_nooverflow_case1():
    with sw.gpu(remain=256):
        a = torch.randn(1024, 1024, 100).to("cuda:0")


def test_nooverflow_case2():
    with sw.gpu(remain=256):
        a = torch.randn(1024, 1024, 100, device="cuda:0")


def test_nooverflow_case3():
    with sw.gpu(remain=256):
        a = torch.randn(1024, 1024, 100).cuda()


@pytest.mark.skipif(
    getGPUs()[0].memoryFree < 1024 * 4.8, reason="No enough RAM to run this test."
)
def test_nooverflow_longcase1():
    with sw.gpu(remain=256):
        # Total = 4.8 GB
        device = "cuda:0"
        c1 = torch.randn(1000, 1000, 100).to(device)
        c2 = torch.randn(1000, 1000, 100, device=device)
        c3 = torch.randn(1000, 1000, 100).cuda()
        c4 = torch.randn(1000, 1000, 100).to(device)
        c5 = torch.randn(1000, 1000, 100, device=device)
        c6 = torch.randn(1000, 1000, 100).cuda()
        c7 = torch.randn(1000, 1000, 100).to(device)
        c8 = torch.randn(1000, 1000, 100, device=device)
        c9 = torch.randn(1000, 1000, 100).cuda()
        c10 = torch.randn(1000, 1000, 100).to(device)
        c11 = torch.randn(1000, 1000, 100, device=device)
        c12 = torch.randn(1000, 1000, 100).cuda()


@pytest.mark.skipif(
    getGPUs()[0].memoryFree < 1024 * 18, reason="No enough RAM to run this test."
)
def test_nooverflow_longcase2():
    with sw.gpu(remain=256):
        # Total = 18 GB
        device = "cuda:0"
        c1 = torch.randn(1000, 1000, 100).to(device)
        c2 = torch.randn(1000, 1000, 100, device=device)
        c3 = torch.randn(1000, 1000, 100).cuda()
        c4 = torch.randn(1000, 1000, 200).to(device)
        c5 = torch.randn(1000, 1000, 200, device=device)
        c6 = torch.randn(1000, 1000, 200).cuda()
        c7 = torch.randn(1000, 1000, 400).to(device)
        c8 = torch.randn(1000, 1000, 400, device=device)
        c9 = torch.randn(1000, 1000, 400).cuda()
        c10 = torch.randn(1000, 1000, 800).to(device)
        c11 = torch.randn(1000, 1000, 800, device=device)
        c12 = torch.randn(1000, 1000, 800).cuda()


# test_nooverflow_case1()
# test_nooverflow_case2()
# test_nooverflow_case3()
# test_nooverflow_longcase1()
# test_nooverflow_longcase2()


