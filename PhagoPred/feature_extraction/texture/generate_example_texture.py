import numpy as np
from scipy.optimize import minimize


from PhagoPred.utils import mask_funcs

def objective(flat_image, target_features, im_size):
    image = flat_image.reshape(im_size)
    image_features = mask_funcs.get_haralick_texture_features(image.astype(np.uint8), mask=None)
    res = np.linalg.norm(image_features.flatten() - target_features.flatten())
    print(res)
    return res


def example_texture(haralick_features, im_size, image):
    
    flat_image = image.flatten()

    optimised = minimize(objective, flat_image, args=(haralick_features, im_size), method='Nelder-Mead', bounds = [(0, 255)]*im_size[0]*im_size[1], options = {'maxiter':100})

    print('\n', optimised.success, optimised.message)
    return optimised.x.reshape(im_size)

    # return image

