import time

class Timer:
    def __init__(self):
        self.start_time = time.time()
        self.end_time = 0
        self.steps = 0

    def step(self):
        self.steps += 1
        self.end_time = time.time()

    def get_speed(self):
        speed = 1.0 * self.steps / (self.end_time - self.start_time)
        return speed



