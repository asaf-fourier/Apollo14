from enum import Enum


class Interaction(Enum):
    REFLECTED = "reflected"
    TRANSMITTED = "transmitted"
    ENTERING = "entering"
    EXITING = "exiting"
    TIR = "tir"
    ABSORBED = "absorbed"
