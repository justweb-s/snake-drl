from __future__ import annotations


class CurriculumManager:
    def __init__(self, grid_min: int = 10, grid_max: int = 30, threshold_scores=(5, 10, 15, 20)):
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.threshold_scores = tuple(threshold_scores)
        self.current_level = 0
        self.max_level = len(self.threshold_scores)

    def should_advance(self, avg_score: float) -> bool:
        return self.current_level < self.max_level and avg_score >= self.threshold_scores[self.current_level]

    def advance_level(self) -> bool:
        if self.current_level < self.max_level:
            self.current_level += 1
            return True
        return False

    def get_grid_size(self) -> tuple[int, int]:
        progress = min(1.0, self.current_level / self.max_level) if self.max_level else 1.0
        grid_size = int(self.grid_min + progress * (self.grid_max - self.grid_min))
        return grid_size, grid_size

    def get_obstacle_setting(self) -> bool:
        return self.current_level >= 1
