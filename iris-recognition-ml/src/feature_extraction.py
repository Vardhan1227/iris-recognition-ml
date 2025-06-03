from skimage.feature import hog
import numpy as np

def extract_hog_features(images):
    hog_features = []
    for img in images:
        features = hog(img, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=False)
        hog_features.append(features)
    return np.array(hog_features)
