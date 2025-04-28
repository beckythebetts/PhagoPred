import numpy as np
from scipy.optimize import minimize

from PhagoPred.utils import mask_funcs

def objective(flat_image, target_features):
    image = flat_image.reshape((50, 50))
    image_features = mask_funcs.get_haralick_texture_features(image.astype(np.int16), mask=None)
    return np.linalg.norm(image_features.flatten() - target_features.flatten())


def example_texture(haralick_features):
    image = np.random.randint(0, 255, size=(50, 50))
    flat_image = image.flatten()

    optimised = minimize(objective, flat_image, args=haralick_features)

    print(optimised.success)
    return optimised.x.reshape(50, 50)

