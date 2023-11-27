import imageio
from skimage.transform import resize
from skimage.color import rgb2gray
import scipy.ndimage
import cv2
import json
import numpy as np
from PIL import Image
import pickle

def _load_pickle(pickle_path):
    with open(pickle_path, "rb") as file:
        pickle_object = pickle.load(file)
    return pickle_object

def _load_json(json_path):
    with open(json_path, "r") as json_file:
        data = json.load(json_file)
    return data

def _load_png(png_path):
    png_as_numpy = imageio.imread(png_path)
    return png_as_numpy

def _load_disp_from_png(png_path):
    f_in = np.array(Image.open(png_path))
    d_r = f_in[:,:,0].astype('float64')
    d_g = f_in[:,:,1].astype('float64')
    d_b = f_in[:,:,2].astype('float64')

    depth = d_r * 4 + d_g / (2**6) + d_b / (2**14)
    depth = np.round(depth).astype(np.uint8)
    return depth


def _load_ppm(ppm_path):
    ppm_as_numpy = imageio.imread(ppm_path)
    return ppm_as_numpy

def _adjust_displacement_pgm(pgm_as_numpy, upscale_factor):
    pgm_as_numpy = pgm_as_numpy / upscale_factor
    return pgm_as_numpy

def _load_pgm(pgm_path):
    pgm_as_numpy = imageio.imread(pgm_path)
    return pgm_as_numpy

def _cast_to_uint8(numpy_array):
    uint8_numpy_array = numpy_array.astype(np.uint8)
    return uint8_numpy_array

def _downscale_numpy_array_by_factor(numpy_array, downscale_factor):
    numpy_array = numpy_array / downscale_factor
    return numpy_array

def _downsample_numpy_image(numpy_array, downsample_shape):
    downsampled_array = resize(numpy_array,downsample_shape)
    return downsampled_array

def _resize_numpy_displacements(numpy_array, downsample_factor):
    frame_shape = _calculate_frame_shape_from_downsample_factor(numpy_array,downsample_factor)
    downsampled_array = resize(numpy_array,frame_shape,preserve_range=True,order=0)
    return downsampled_array

def _calculate_frame_shape_from_downsample_factor(frame,downsample_factor):
    full_height, full_width = frame.shape
    height = full_height //downsample_factor
    width = full_width //downsample_factor
    return height, width


def _convert_numpy_rgb_to_gray_scale(rgb_numpy_array):
    gray_numpy_array = rgb2gray(rgb_numpy_array)
    return gray_numpy_array

def _median_filter(image, median_filter_size):
    filtered_image = scipy.ndimage.median_filter(image, size=median_filter_size)
    return filtered_image

def _bilateral_filter(image):
    bilateral_filtered_image = cv2.bilateralFilter(image, d=12, sigmaColor=75, sigmaSpace=75)
    return bilateral_filtered_image
