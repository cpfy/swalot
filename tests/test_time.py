import time
import swalot as sw
from pytest import approx


DEBUG = True
if DEBUG:
    from icecream import ic


def test_normal_time():
    t0 = time.time()
    for _ in range(3):
        start_time = time.time()
        time.sleep(0.3)
        end_time = time.time()
        ic(start_time, end_time, end_time - start_time)

        assert end_time - start_time == approx(0.3, rel=1e-1)

    t_end = time.time()
    assert t_end - t0 == approx(0.3 * 3, rel=2e-1)


def test_scale_case1():
    scale = 0.1
    with sw.time(scale):
        for _ in range(3):
            start_time = time.time()
            time.sleep(1)
            end_time = time.time()
            ic(start_time, end_time, end_time - start_time)

            assert end_time - start_time == approx(1 / scale, rel=1e-1)

    ic(time.time())
    assert time.time() < end_time


def test_decorator():
    @sw.time(speed=5)
    def time_in_func():
        for _ in range(3):
            start_time = time.time()
            time.sleep(1)
            end_time = time.time()
            ic(start_time, end_time, end_time - start_time)

            assert end_time - start_time == approx(1 / 5, rel=1e-1)

    time_in_func()
