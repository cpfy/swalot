import time
import random
from rich import print

DEBUG = True
if DEBUG:
    from icecream import ic

class TimeController():
    def __init__(self, speed=1.0, rand=False, seed=None):
        self.speed = speed
        # self.offset = offset
        self.original_time = None

        self.rand = rand

        if seed:
            assert seed.isdigit(), "Invalid Format! Seed must be integers with 10 length!"
            self.seed = seed
        else:
            self.seed = self._random_seed()

        if(rand):
            random.seed(self.seed)
            print(print("[bold cyan][Info][/bold cyan][bold magenta] | swat:[/bold magenta] Set Timer seed: [bold green]{} [/bold green]".format(self.seed)))
            self.speed = random.uniform(0.66, 1.5)
            if DEBUG: print(self.speed)

    def __enter__(self):
        self.original_time = time.time
        time.time = self.modified_time
        self.start_time = self.original_time()

    def modified_time(self):
        return self.start_time + (self.original_time() - self.start_time) / self.speed

    def __exit__(self, exc_type, exc_val, exc_tb):
        time.time = self.original_time

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper

    def _random_seed(self, length=10):
        random.seed()
        min = 10**(length-1)
        max = 9*min + (min-1)
        return random.randint(min, max)

