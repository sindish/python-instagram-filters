import cv2 
import numpy as np

def negative(image):
     return (255-image)


def edgy(image, minval=100, maxval=300):
    # output grayscale
    return cv2.Canny(image,minval,maxval)

def blur(image, aperture=5):
    assert aperture % 2 == 1  # must be odd cuz cv2 
    return cv2.GaussianBlur(image, (aperture, aperture), 0)

def sharpen(image, a, b, sigma=11):
    blurred = blur(image, sigma)
    sharper = np.clip(image * a - blurred * b, 0, 255.0)  
    return np.uint8(sharper)

def add_reds(image):
    channel = channel_adjust(image, 2, [
    0, 0.05, 0.1, 0.2, 0.3,
    0.5, 0.7, 0.8, 0.9,
    0.95, 1.0])
    reconstructed = np.stack([image[:,:,0], image[:,:,1], channel], axis = -1)
    return np.uint8(reconstructed)

def add_greens(image):
    channel = channel_adjust(image, 1, [
    0, 0.05, 0.1, 0.2, 0.3,
    0.5, 0.7, 0.8, 0.9,
    0.95, 1.0])
    reconstructed = np.stack([image[:,:,0], channel, image[:,:,2]], axis = -1)
    return np.uint8(reconstructed)


def add_blues(image):
    channel = channel_adjust(image, 0, [
    0, 0.05, 0.1, 0.2, 0.3,
    0.5, 0.7, 0.8, 0.9,
    0.95, 1.0])
    reconstructed = np.stack([channel, image[:,:,1], image[:,:,2]], axis = -1)
    return np.uint8(reconstructed)

#util 
def channel_adjust(image, channel_idx, values):
    channel = image[:, :, channel_idx]
    # flatten
    orig_size = channel.shape
    flat_channel = channel.flatten()
    float_flat = np.true_divide(flat_channel, 255.0)
    adjusted = np.interp(
        float_flat,
        np.linspace(0, 1, len(values)),
        values)
    adjusted = adjusted.reshape(orig_size)
    return np.uint8(adjusted * 255.0)