import numpy as np
from PIL import Image

from torchvision import transforms

# label colors + resolution

CAR = [0, 255, 0]
STREET = [255, 0, 0]
ELSE = [0, 0, 0]

RESOLUTION = 224

#############

CAR_TABLE = np.array([CAR] * RESOLUTION**2)
STREET_TABLE = np.array([STREET] * RESOLUTION**2)

CAR_LABEL = 2
STREET_LABEL = 1
ELSE_LABEL = 0 # default

TABLES = [CAR_TABLE, STREET_TABLE]
LABELS = [CAR_LABEL, STREET_LABEL] # ensure the index of this matches that of the tables

def image_to_label(img: Image):
    img = np.array(img).reshape(RESOLUTION**2, 3)
    
    t = np.zeros((RESOLUTION, RESOLUTION))
    for table, label in zip(TABLES, LABELS):
        mask = (table == img).all(axis = 1).reshape(RESOLUTION, RESOLUTION)
        t[mask] = label
    
    return np.int64(t)

def image_to_label_transform(x):
    return (np.array(x[0]), image_to_label(x[1]))