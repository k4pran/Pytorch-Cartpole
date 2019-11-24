from enum import Enum


class HyperParams(Enum):

    EPSILON = 0.9
    EPSILON_MIN = 0.01
    EPSILON_DECAY = 0.99
    GAMMA = 0.9
    LEARNING_RATE = 0.001

    @classmethod
    def has_value(cls, key):
        return key in cls._member_names_