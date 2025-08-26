from enum import Enum


class SchedulerType(Enum):
    """Enum for learning rate scheduler types"""

    COSINE = "cosine"
    EXPONENTIAL = "exponential"
    LINEAR_FLAT = "linear_flat"

    @classmethod
    def get_all_values(cls):
        """Get all scheduler type values as a list"""
        return [scheduler.value for scheduler in cls]

    @classmethod
    def is_valid(cls, value: str) -> bool:
        """Check if a scheduler type value is valid"""
        return value in cls.get_all_values()
