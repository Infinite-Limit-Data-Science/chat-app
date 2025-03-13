from typing import Tuple, Dict, Type, Any
from abc import ABC, ABCMeta, abstractmethod

TASK_REGISTRY: Dict[Tuple[float, float, str], Type["BaseTask"]] = {}


class TaskMeta(ABCMeta):
    """Metaclass to automatically register task classes"""

    def __init__(
        cls: "BaseTask", name: str, bases: Tuple[Any], clsdict: Dict[str, Any]
    ):
        super().__init__(name, bases, clsdict)
        if hasattr(cls, "temperature_range") and hasattr(cls, "task_type"):
            TASK_REGISTRY[
                (cls.temperature_range[0], cls.temperature_range[1], cls.task_type)
            ] = cls


class BaseTask(ABC, metaclass=TaskMeta):
    """The lower the temperature range, the better the algorithm and, thus, accuracy"""

    temperature_range: Tuple[float, float] = (0.0, 1.0)
    task_type: str = ""

    @abstractmethod
    def perform(self):
        """Perform work"""
        pass

    @classmethod
    def fetch(cls, temperature: float, task_type: str) -> "BaseTask":
        for (
            temp_min,
            temp_max,
            registered_task_type,
        ), task_class in TASK_REGISTRY.items():
            if (
                temp_min <= temperature <= temp_max
                and registered_task_type == task_type
            ):
                return task_class()
        raise ValueError(
            f"No task class found for temperature {temperature} and task type {task_type}"
        )
