import os
import pickle
from imageio import imwrite
import numpy as np

def _make_missing_directories(file_path):
    directory_path = os.path.dirname(file_path)
    os.makedirs(directory_path, exist_ok=True)


def _save_pgm(np_array,pgm_path):
    np_array_uint8 = np.uint8(np_array)
    _make_missing_directories(pgm_path)
    imwrite(pgm_path, np_array_uint8, format='pgm')

def _upscale_numpy_array_by_factor(numpy_array, upscale_factor,array_max=255):
    numpy_array[numpy_array*upscale_factor>array_max] = array_max // upscale_factor

    numpy_array = numpy_array * upscale_factor
    return numpy_array

def _save_pickle(pickle_object, pickle_path):
    with open(pickle_path, "wb") as file:
            pickle.dump(pickle_object, file)