import numpy as np

class AngleSource:
    def get_angle_deg(self) -> float:
        raise NotImplementedError

class GeneratedAngleSource(AngleSource):
    def __init__(self, angles):
        self.angles = np.asarray(angles, dtype=float)
        self.i = 0

    def get_angle_deg(self) -> float:
        angle = self.angles[self.i % len(self.angles)]
        self.i += 1
        return float(angle)

class LiveAngleSource(AngleSource):
    def __init__(self, initial_angle_deg: float = 0.0):
        self.current_angle_deg = float(initial_angle_deg)

    def update_angle(self, angle_deg: float):
        self.current_angle_deg = float(angle_deg)

    def get_angle_deg(self) -> float:
        return self.current_angle_deg
    
def generate_motion(angle_max_deg=90.0, ang_speed_deg_per_sec=60.0, cycles=1, fps=60):
    cycle_time = 2.0 * angle_max_deg / ang_speed_deg_per_sec
    frames_per_cycle = int(np.ceil(cycle_time * fps))

    i = np.arange(frames_per_cycle)
    phase = i / max(frames_per_cycle - 1, 1)
    tri = 1.0 - np.abs(2.0 * phase - 1.0)   # 0 -> 1 -> 0
    angles = np.tile(tri * angle_max_deg, cycles)
    return angles