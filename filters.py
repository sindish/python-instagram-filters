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

#util 
def adjust_histogram(channel, values):
    orig_size = channel.shape
    flat_channel = channel.flatten()
    values = [x * 255.0 for x in values]
    adjusted = np.interp(np.float32(flat_channel), np.linspace(0, 255.0, len(values)), values)
    adjusted = adjusted.reshape(orig_size)
    return np.uint8(adjusted)

def add_reds(image):
    red = adjust_histogram(image[:, :, 2], [
    0, 0.05, 0.1, 0.2, 0.3,
    0.5, 0.7, 0.8, 0.9,
    0.95, 1.0])
    reconstructed = np.stack([image[:,:,0], image[:,:,1], red], axis = -1)
    return np.uint8(reconstructed)

def add_greens(image):
    green = adjust_histogram(image[:, :, 1], [
    0, 0.05, 0.1, 0.2, 0.3,
    0.5, 0.7, 0.8, 0.9,
    0.95, 1.0])
    reconstructed = np.stack([image[:,:,0], green, image[:,:,2]], axis = -1)
    return np.uint8(reconstructed)


def add_blues(image):
    blue = adjust_histogram(image[:, :, 0], [
    0, 0.05, 0.1, 0.2, 0.3,
    0.5, 0.7, 0.8, 0.9,
    0.95, 1.0])
    reconstructed = np.stack([blue, image[:,:,1], image[:,:,2]], axis = -1)
    return np.uint8(reconstructed)

def gotham(image):
    red_boost_lower = adjust_histogram(image[:, :, 2], [
        0, 0.05, 0.1, 0.2, 0.3,
        0.5, 0.7, 0.8, 0.9,
        0.95, 1.0])
    blue_add = np.uint8(np.clip(np.float32(image[:, :, 0]) + 0.03 * 255.0, 0, 255.0))
    processed = np.stack([blue_add, image[:, :, 1], red_boost_lower], axis=2)
    
    processed = sharpen(processed, 1.3, 0.3)
        
    adjusted_blues = adjust_histogram(processed[:, :, 0], [
        0, 0.047, 0.118, 0.251, 0.318,
        0.392, 0.42, 0.439, 0.475,
        0.561, 0.58, 0.627, 0.671,
        0.733, 0.847, 0.925, 1])
    
    processed = np.stack([adjusted_blues, processed[:, :, 1], processed[:, :, 2]], axis=2)
    return processed

def cinematic(image):
    red_remove_low_boost_high = adjust_histogram(image[:, :, 2], [
        0., 0., 0.165, 0.333, 0.5, 0.665, 0.833, 1.0, 1.0])
    green_remove_high_boost_low = adjust_histogram(image[:, :, 1], [
        0.15, 0.5, 0.85])
    blue_remove_high_boost_low = adjust_histogram(image[:, :, 0], [
        0.125, 0.5, 0.875])
    
    processed = np.stack([blue_remove_high_boost_low, green_remove_high_boost_low, red_remove_low_boost_high], axis=2)
    return processed

