import numpy as np

FILL, PREDICTOR, COALITIONS = 0, 1, 2


def get_generator(sample, kind: int, *extra) -> np.random.Generator:
    return np.random.default_rng([sample.seed, kind, *extra])
