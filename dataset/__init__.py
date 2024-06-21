from .cifar100 import SplitCifar100
from .cars import SplitCars
from .cub import SplitCUB
from .aircraft import SplitAircraft
from .birdsnap import SplitBirdsnap
from .gtsrb import SplitGTSRB

DATASET = {
    'cifar100': SplitCifar100,
    'cars': SplitCars,
    'cub': SplitCUB,
    'aircraft': SplitAircraft,
    'birdsnap': SplitBirdsnap,
    'gtsrb': SplitGTSRB
}