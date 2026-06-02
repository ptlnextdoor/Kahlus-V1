from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from neurotwin.data.prepared_tasks import SupervisedWindowTask


BaselineRunner = Callable[[SupervisedWindowTask, int, int], np.ndarray]
TaskAvailability = Callable[[SupervisedWindowTask], bool]


@dataclass(frozen=True)
class ExecutableBaselineRunner:
    model_id: str
    runner: BaselineRunner
    available_for_task: TaskAvailability = lambda task: True

    def supports(self, task: SupervisedWindowTask) -> bool:
        return self.available_for_task(task)

    def predict(self, task: SupervisedWindowTask, seed: int, train_steps: int) -> np.ndarray:
        return self.runner(task, seed, train_steps)
