from enum import Enum


class HyperParams(Enum):

    BATCH_SIZE = 32
    EPSILON = 0.9
    EPSILON_MIN = 0.01
    EPSILON_DECAY = 0.99
    GAMMA = 0.99
    LEARNING_RATE = 0.001
    MEMORY_MAX = 1000

    @classmethod
    def has_value(cls, key):
        return key in cls._member_names_