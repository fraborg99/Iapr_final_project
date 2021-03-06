from this import d
from turtle import left
import time
import pickle

from typing import List, Union, Tuple
import os
import time
from copy import copy
import pickle

import numpy as np
from scipy.fft import fft
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from skimage.transform import rotate, resize
from skimage.morphology import binary_erosion, dilation, opening, closing, disk, binary_closing
from skimage.color import rgb2gray, rgb2hsv
from skimage.filters import median, gaussian
from skimage.measure import find_contours
from skimage.exposure import match_histograms
from IPython.display import Image, display, clear_output
import cv2
import PIL.Image
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from our_utils import deskew


def corners(lis, im):
    """
    Function to select from an array of points delimiting a shape, those corresponding to the corners
    
    Inputs:
    -------
    lis: list of point delimiting a quadratic shape.
    im: reference image.
    
    Outputs:
    --------
    The four corners of the shape
    """
    # save the reference image's dimensions
    height = im.shape[0]
    width = im.shape[1]
    n = len(lis)
    corners = []
    # To detect the corner we consider couples of consecutive points: if the absolute variation in position in two successive 
    # points is greater along the opposite axis (say x) of what was the axis with the biggest variation for the former two
    # points (say y), it means we have found a possible corner.
    if np.abs(lis[0][0][1] - lis[1][0][0]) > np.abs(lis[0][0][1] - lis[1][0][0]):
        old_axis = "x"
    else:
        old_axis = "y"
    for i in range(len(lis)+1):
        if np.abs(lis[(i+1)%n][0][0] - lis[i%n][0][0]) > np.abs(lis[(i+1)%n][0][1] - lis[i%n][0][1]):
            current_axis = "x"
        else:
            current_axis = "y"
        if current_axis != old_axis:
            old_axis = current_axis
            corners.append(lis[i%n][0][:])
            continue
    corners = np.array(corners)

    # whenever it happens that you end up with more than 4 points, keep only the two rightmost, the two leftmost, the
    # two higest and the two lowest points. In this way you end up exaclty with the 4 corners
    while len(corners)>4:

        right = list(map(lambda x: x[0] > width//2, corners))
        if len(np.array(corners)[right]) != 2:
            rem = np.argmin([x[0] for x in np.array(corners)[right]])
            corners = np.delete(corners, rem, axis = 0)

        left = list(map(lambda x: x[0] <= width//2, corners))
        if len(np.array(corners)[left]) != 2:
            rem = np.argmax([x[0] for x in np.array(corners)[left]])
            corners = np.delete(corners, rem, axis = 0)

        up = list(map(lambda x: x[1] > height//2, corners))
        if len(np.array(corners)[up]) != 2:
            rem = np.argmin([x[1] for x in np.array(corners)[up]])
            corners = np.delete(corners, rem, axis = 0)

        down = list(map(lambda x: x[1] <= height//2, corners))
        if len(np.array(corners)[down]) != 2:
            rem = np.argmax([x[1] for x in np.array(corners)[down]])
            corners = np.delete(corners, rem, axis = 0)

    return corners

def plot(im):
    """
    Function to isolate the table from the images.
    
    Inputs:
    -------
    im: reference image.
    
    Outputs:
    --------
    The convex hull, the edges and the contours of the table.
    """    
    ## add some whithe pixels above and below the make the table more recognizable in case of decentered picutres
    im = np.vstack([np.zeros((100,6000,3)), im, np.zeros((100,6000,3))])
    im = im.astype(np.uint8)

    width = im.shape[1]
    height = im.shape[0]

    # go to gray color and apply a gaussian blur filter
    img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (21, 21), 1)

    # obtain a binary image
    ret, thresh = cv2.threshold(img_gray, 170, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # define the filter that will be used for opening, dialtion and erosion
    kernelop = np.ones((80,80),np.uint8)
    kerneldil = np.ones((90,90),np.uint8)
    kernelder = np.ones((30,30),np.uint8)

    # apply opening, dialtion and erosion
    image = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernelop)
    image = cv2.dilate(image,kerneldil,iterations = 1)
    image = cv2.erode(image ,kernelder,iterations = 5)
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    # find the contours and select that one which has points that are both in the lower left and upper right corners, it
    # will be the table's contour
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        current_contour = contours[i]
        current_contour = current_contour.reshape(len(current_contour), 2)
        if (np.any([j[0]>(width*2/3) and j[1]<(height/4) for j in current_contour]) and np.any([j[0]<(width/3) and j[1]>(height*3/4) for j in current_contour])):

            max_c = i
            break

    # apply convex hull
    hull = cv2.convexHull(contours[max_c])

    return hull, edges, contours

def cyclic_intersection_pts(pts):
    """
    Sorts 4 points in clockwise direction to enable the rest of the code to stretch the correct dimensions
    
    Inputs:
    -------
    pts: points to be sorted.
    
    Outputs:
    --------
    Points sorted in clockwise direction
    """
    if pts.shape[0] != 4:
        return None

    # Calculate the center
    center = np.mean(pts, axis=0)

    # Sort the points in clockwise
    cyclic_pts = [
        pts[np.where(np.logical_and(pts[:, 0] < center[0], pts[:, 1] < center[1]))[0][0], :], # Top-left
        pts[np.where(np.logical_and(pts[:, 0] > center[0], pts[:, 1] < center[1]))[0][0], :], # Top-right
        pts[np.where(np.logical_and(pts[:, 0] > center[0], pts[:, 1] > center[1]))[0][0], :], # Bottom-Right
        pts[np.where(np.logical_and(pts[:, 0] < center[0], pts[:, 1] > center[1]))[0][0], :]  # Bottom-Left
    ]

    return np.array(cyclic_pts)

def deskew(im, calibration = False):
    """
    Cut the table portion and deskew the cropped image.
    
    Inputs:
    -------
    im: image from which to cut the table.
    calibration: Set to True if you need to calibrate the histogram using a reference train image
    
    Outputs:
    --------
    Deskewed image
    """
    # If calibration set to true, before cropping and deskewing, match the histogram of the current image with that of 
    # image train_21
    if calibration:
        file2 = os.path.join("data/train/train_21.jpg")
        im2 = skimage.io.imread(file2)
        newimage = e.match_histograms(im, im2, channel_axis=-1)
        im = newimage
    try:
        # get corners of the table
        hull, edges, contours = plot(im)
        corn = corners(hull, im) - np.tile(np.array([0,100]), (4, 1))
        intersect_pts = cyclic_intersection_pts(corn)

        width = 4000
        # List the output points in the same order as input
        # Top-left, top-right, bottom-right, bottom-left
        dstPts = [[0, 0], [width, 0], [width, width], [0, width]]
        # Get the transform
        m = cv2.getPerspectiveTransform(np.float32(intersect_pts), np.float32(dstPts))
        # Transform the image
        out = cv2.warpPerspective(im, m, (int(width), int(width))) #substitute color to im to save pics
        #return deskewd img
    except IndexError:
        try:
            # if above doesn't work, try again with the image being recalibrated
            out = deskew(im, calibration = True)
        except:
            # if still it doesn't work, cut an arbitrary portion from the center of the image
            out = im[200:3800,1200:4800,:]
    return out


def count_fiches(fiches):
    """
    Count the number of different fiches per color that are on the table.
    
    Inputs:
    -------
    fiches: Image of the portion of the table containing the fiches, obtained with the function area_partition.
    
    Outputs:
    --------
    Dictionary containing the number of fiches per color.
    """
    # First we convert the fiches image from rgb to hsv
    fiches_hsv = cv2.cvtColor(fiches, cv2.COLOR_BGR2HSV)    
    # first we load the mean saturation numpy file to map histograms to is, as you will see
    mean_saturation = np.load("mean_saturation.npy")
    # We define the thresholds values for each color, for each color, a part from red, we have two thresholds, one for
    # those images which mean saturation is close to the mean of the train set and one for those images in which the saturation
    # is farther than one standard deviation from the mean saturation. For the red color the threshold is just one since, as you
    # will see, it is processed in a slightly different manner, mapping the hostogram of high saturation pictures to the mean values.
    # All threshold have been selected with trial and error procedure
    lower_bounds = {"red": np.array([90, 100, 120]),
                   "green": [np.array([30, 60, 40]), np.array([23, 180, 60])],
                   "blue": [np.array([0, 60, 40]), np.array([0, 180, 120])],
                   "black": [np.array([80, 20, 0]), np.array([0, 80, 0])],
                   "white": [np.array([70, 30, 115]), np.array([0, 5, 230])]}
    
    upper_bounds = {"red": np.array([151, 255,200]),
                   "green": [np.array([90, 120,100]), np.array([40, 255,220])],
                   "blue": [np.array([25, 120,100]), np.array([23, 255,220])],
                   "black": [np.array([120, 80,60]), np.array([85, 180,181])],
                   "white": [np.array([130, 90, 175]), np.array([20, 60, 270])]}
    
    # In the same way we define the sizes of the closing and opening operators
    kernel_closings = {"red": (40,40),
                      "green": [(20,20), (20,20)],
                      "blue": [(20,20), (20,20)],
                      "black": [(20,20), (20,20)],
                      "white": [(20,20), (50,50)]}
    
    kernel_openings = {"red": (50,50),
                      "green": [(100,100), (100,100)],
                      "blue": [(100,100), (100,100)],
                      "black": [(100,100), (100,100)],
                      "white": [(100,100), (170,170)]}
    
    # Define a reference area of the fiche to be used to determine the number of fiches. the number slightly varies as the
    # erosive power of the mathematical morphology operations is different for each color
    fiches_areas = {"red": 57500,
                    "green": [53000, 53000],
                    "blue": [53000, 53000],
                    "black": [50000, 50000],
                    "white": [48000, 45000]}
        
    colors = ["red", "green", "blue", "black", "white"]
    count = {}
    for color in colors:
        if color == "red":
            # In the case of red color we map the histogram of high saturation images to the mean histogram
            # 61.57001 and 33.639077 are respectively mean and standard deviation of our train set's saturation
            if np.abs(np.mean(fiches_hsv[:,:,1]) - 61.57001)> 33.639077:
                current_hsv = e.match_histograms(fiches_hsv, mean_saturation, channel_axis=-1)
            else:
                current_hsv = fiches_hsv
            
            # lower bound and upper bound for red color found by trial and error
            lower_bound = lower_bounds[color] 
            upper_bound = upper_bounds[color]
            # filter for closing and opening for the thresholded image
            kernelcl = np.ones(kernel_closings[color],np.uint8)
            kernelop = np.ones(kernel_openings[color],np.uint8)
            
            ref_area = fiches_areas[color]
            
        else: # if not red
            if (np.mean(fiches_hsv[:,:,1]) - 61.57001)> 33.639077:
                # in case of non-red color we simply use different thresholds instead of mapping histograms
                lower_bound = lower_bounds[color][0]
                upper_bound = upper_bounds[color][0]
                # filter for closing and opening for the thresholded image
                kernelcl = np.ones(kernel_closings[color][0],np.uint8)
                kernelop = np.ones(kernel_openings[color][0],np.uint8)
                
                ref_area = fiches_areas[color][0]
            else:
                lower_bound = lower_bounds[color][1]
                upper_bound = upper_bounds[color][1]

                kernelcl = np.ones(kernel_closings[color][1],np.uint8)
                kernelop = np.ones(kernel_openings[color][1],np.uint8)
                
                ref_area = fiches_areas[color][1]
                
            current_hsv = fiches_hsv
                
        # find the colors within the boundaries
        mask = cv2.inRange(current_hsv, lower_bound, upper_bound)
        # closing and opening to remove unwanted noise from the mask
        mask2 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernelcl)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernelop)
        # Now we detect the edges in the mask we created and we find the contours
        edges = cv2.Canny(mask2, 10, 10, apertureSize=3)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area = 0
        # we sum up the areas of the found contours and chech how many fiches can be contained in that area, using as the
        # reference area of a fiche the areas defined above
        for c in contours:
            area+=cv2.contourArea(c)
        n = area // ref_area
        # however if we find that there are more contours than estimated fiches we use the number of contours as result
        if len(contours)>n:
            n = len(contours)
        # Finally, if in the case of red fiches, the number of fiches is too big, it means that there has been some issue,
        # so we return 0
        if color == "red":
            if n > 8:
                n = 0
        count[color] = int(n)
    return count


class CouldNotFind4PointApprox(Exception):
    pass


def memoize(func):
    cached = {}
    def wrapper(path_data):
        if path_data not in cached:            
            cached[path_data] = func(path_data)
        return cached[path_data]
    return wrapper

@memoize
def load_data(path_data: str):
    return {file.split('.jpg')[0]: PIL.Image.open(os.path.join(path_data, file))
            for file in os.listdir(path_data)
            if file.endswith('.jpg')}

def my_interp(contour: np.ndarray, L_max: int):
    """
    Interpolate a contour to have a cooredinate list of length `L_max`.
    
    Inputs:
    -------
    contour: np.array of N x 2, where N is the number of points in the
        contour, and the first and second columns represent the x- and
        y-dimensions, respectively.
    L_max: integer length of the longest contour of all digits.
    
    Outputs:
    --------
    The interpolated contour of shape L_max x 2
    """
    L = len(contour)
    new_inputs = np.linspace(0, L - 1, L_max)
    interps = [interp1d(range(L), dim) for dim in contour.T]
    return np.array([interp(new_inputs) for interp in interps]).T

def get_fourier_descriptors(contours: List[np.array], length: Union[None, int] = None):
    """
    Given a list of contours, get the fourier descriptors for each contour

    Inputs:
    -------
    contours: list of numpy array contours, each contours should have shape
        N x 2, where N is the number of points in the contour, and the first
        and second columns represent the x- and y-dimensions, respectively.
    
    Outputs:
    -------
    """
    if length:
        L_max = max([len(c) for c in contours])
    else:
        L_max = length
    interp_contours = [my_interp(c, L_max) for c in contours]
    complex_contours = [np.array([complex(*p) for p in c]) for c in interp_contours]
    return [fft(c) for c in complex_contours]

def get_closest_contour(im: np.ndarray, contours: list, verbose: bool = False):
    """
    Check that a contour encircles and the center point of an image
    
    Inputs
    ------
    im: numpy array of shape (w, h)
    contours: array or list of numpy array contours of shape (l, 1, 2)
    
    Ouputs:
    -------
    Integer index
    """
    
    center = np.array([im.shape[1]//2, im.shape[0]//2])
    min_distance = 10000000
    for i, contour in enumerate(contours):
        distances = [np.linalg.norm(point - center) for point in contour[:, 0]]
        if min(distances) < min_distance:
            min_distance = min(distances)
            min_contour_idx = i
    return min_contour_idx


def contour_fully_connected(contour: np.ndarray):
    """
    Check that a contour forms a closed loop
    
    Inputs
    ------
    contour: numpy array contour of shape (l, 1, 2)
    
    Ouputs:
    -------
    Boolean yes/no answer
    """
    
    return all([np.linalg.norm(a.ravel() - b.ravel()) < 2
                for a, b in zip(contour, np.roll(contour, 1, axis=0))])

def contour_encircles_center(im: np.ndarray, contour: np.ndarray, verbose: bool = False):
    """
    Check that a contour encircles and the center point of an image
    
    Inputs
    ------
    im: numpy array of shape (w, h)
    contour: numpy array contour of shape (l, 1, 2)
    
    Ouputs:
    -------
    Boolean yes/no answer
    """
    
    w, h = im.shape[1], im.shape[0]
    points = [(w // 2, h // 2),
              (w // 2 - w // 6, h // 2 + h // 6),
              (w // 2 - w // 6, h // 2 - h // 6),
              (w // 2 + w // 6, h // 2 + h // 6),
              (w // 2 + w // 6, h // 2 - h // 6),
              (w // 2 + w // 6, h // 2),
              (w // 2 - w // 6, h // 2),
              (w // 2,          h // 2 + h // 6),
              (w // 2,          h // 2 - h // 6)]
    polygon = Polygon([(j, i) for i, j in contour[:, 0]])
    
    if verbose:
        plt.imshow(im, cmap='gray')
        plt.plot(im.shape[1]//2, im.shape[0]//2, 'm*', ms=20)
        plt.plot([point[0] for point in points[1:]], [point[1] for point in points[1:]], 'r*')

    return sum([polygon.contains(Point(*point)) for point in points]) >= 3

def region_growing(img: np.array, thresholds: tuple, seed: tuple = (0, 0),  fill: float = 1.,
                   verbose: bool = False):
    """
    Perform a region growing.

    Inputs:
    -------
    imgs: numpy array image of shape (h, w)
    thersholds: tuple of length 2 of upper and lower thresholds (in that order)
    seed: tuple of length 2, seed point for the algorithm
    fill: the fill value to use
    verbose: whether or not to print progress

    Ouputs:
    -------
    The result of the region growing, a numpy array of same shape of `img`.
    """
    result = np.zeros_like(img)
    lower_threshold, upper_threshold = thresholds

    queue = [seed]
    result[seed] = fill

    f = np.vectorize(lambda x: 1 if (x <= upper_threshold and x >= lower_threshold) else 0)
    watermap = f(img)
    
    tracker = 0
    while len(queue) > 0:
        x, y = queue.pop(0)    
        coordinates = [(x - 1, y - 1), (x, y - 1), (x + 1, y - 1),
                       (x - 1, y),                 (x + 1, y),
                       (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)]
        for x_, y_ in coordinates:
            if not result[x_, y_] and watermap[x_, y_]:
                result[x_, y_] = fill
                queue += [(x_, y_)]
        tracker += 1
        if tracker % 500000 == 0 and verbose:
            print(f'At iteration {tracker}, queue has length {len(queue)}, there are {result.sum()} bricks.')
    return result

def cart2pol(x, y):
    """ Cartesian coordinates to polar coordinates. """
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho

def pol2cart(theta, rho):
    """ Polar coordinates to cartesian coordinates. """
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y

def get_opposite_points(points: np.ndarray, right: bool):
    """
    Given na upper and lower corner of one side of a card, find the
    upper and lower corneers of the other side of the card.
    """
    
    assert points.shape == (2, 1, 2)

    # separate upper and lower points
    upper_point = points[np.argmin(points[:, :, 0].ravel())]
    lower_point = points[np.argmax(points[:, :, 0].ravel())]

    # get the angle between the upper and lower points
    angle = np.arctan2(*(lower_point.ravel() - upper_point.ravel()))
    
    # add or subtract 90 degrees depending on whether we are working
    # with the right-hand or left-hand points
    new_angle = angle + np.pi/2 * (-1 if right else 1)

    # get the upper opposite point
    upper_opposite = np.roll(pol2cart(new_angle, 465) + np.roll(upper_point.ravel(), 1), 1)[np.newaxis, :]

    # get the lower opposite point
    lower_opposite = lower_point.ravel() - np.array([upper_point.ravel() - upper_opposite.ravel()])

    # put the points together, contenate the first point to the end of the list for plotting purposes
    card = np.array([upper_point, lower_point, lower_opposite, upper_opposite]).round().astype(np.int32)
    card_plot = np.vstack((card, card[0][np.newaxis, :]))
    
    return card, card_plot

def separate_left_and_right_points(approx):
    """
    Given a list of four points, separate the rightmost and leftmost points.
    """
    
    # get the left-hand points
    left_points = []
    approx_copy = approx.copy()

    for i in range(2):

        # find the index of the leftmost point
        index_of_leftmost_point = np.argmax(approx_copy[:, :, 1])

        # add this point to the left_points list
        left_points += [approx_copy[index_of_leftmost_point]]

        # delete the point from the approximation
        approx_copy = np.delete(approx_copy, index_of_leftmost_point, axis=0)

    left_points = np.array(left_points)    

    # the rest of the points are the right hand points
    right_points = approx_copy
    
    return right_points, left_points

def rotate_contour(cnt, angle):
    """ Rotate a contour. """
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    
    coordinates = cnt_norm[:, 0, :]
    xs, ys = coordinates[:, 0], coordinates[:, 1]
    thetas, rhos = cart2pol(xs, ys)
    
    thetas = np.rad2deg(thetas)
    thetas = (thetas + angle) % 360
    thetas = np.deg2rad(thetas)
    
    xs, ys = pol2cart(thetas, rhos)
    
    cnt_norm[:, 0, 0] = xs
    cnt_norm[:, 0, 1] = ys

    cnt_rotated = cnt_norm + [cx, cy]
    cnt_rotated = cnt_rotated.astype(np.int32)

    return cnt_rotated

def area_partition(out, default = False, reference = None):
    """ Partition the table into different segments. """
    if default:
        ref = reference
        
    else:
        ref = out
    k = 4
    
    sides_rat = 1/(k - 5/4)
    cent_rat = 1/(k - 2)
    size = ref.shape[0]
    
    half = size//2
    beg = round(sides_rat*size)
    end = round((1-sides_rat)*size)
    mid = round(cent_rat*size)
    
    pl1 = np.rot90(out[beg-150:end+50, end+beg//3-50:, :], k=3, axes=(0, 1))
    pl2 = np.rot90(out[:beg//3*2, half+beg//4:end + beg//3*2, :], k=2, axes=(0, 1))
    pl3 = np.rot90(out[:beg//3*2, beg//3:half, :], k=2, axes=(0, 1))
    pl4 = np.rot90(out[beg-50:end+100, :beg//3*2+50, :], k=1, axes=(0, 1))
    table = out[end:,:,:]
    
    sides_rat = 1/k
    cent_rat = 1/(k - 2)
    size = ref.shape[0]
    
    half = size//2
    beg = round(sides_rat*size)
    end = round((1-sides_rat)*size)
    mid = round(cent_rat*size)

    fiches = out[beg:end,beg:end,:] 
    
    return [pl1, pl2, pl3, pl4, table[:,400:-400,:], fiches]

def find_players(h_imgs: List[np.ndarray], imgs: List[np.ndarray],
                 verbose=False):
    """
    Given a list of images of cards, find those players who are not playing
    """

    # Find red cards
    flattened = [img[:2 * img[0].shape[0] // 3].ravel() for img in h_imgs[:4]]
    red_intensity = np.array([(im > 0.93).sum() / np.prod(im.shape[:2]) for im in flattened])
    red_non_players = np.argwhere(red_intensity < 0.1).ravel()

    flattened = [rgb2gray(img)[:2 * img[0].shape[0] // 3].ravel() for img in imgs[:5]]
    other_intensity = np.array([img[img > 0.5].sum() / np.prod(len(img)) for img in flattened])
    other_intensity = (other_intensity / other_intensity[4]).round(2)
    other_non_players = np.argwhere(other_intensity[:4] > 0.91)

    if verbose:
        print('Red intensity:', red_intensity)
        print('Red nonplayers:', red_non_players)
        print('Blue intensity:', other_non_players)
        print('Blue nonplayers:', other_non_players)

    return (np.sort(np.intersect1d(red_non_players, other_non_players)),
            red_intensity, red_non_players, other_intensity, other_non_players)

def get_card_contours(bin_cards: np.ndarray):
    """
    Given a binary image of cards, get the contour of the card
    """

    wrong_shape_contours = find_contours(bin_cards)

    # reshape contours and only take the longest ones
    contours = [np.array([row.T[np.newaxis, :].round() for row in contour]).astype(np.int32)
                for contour in wrong_shape_contours if (len(contour) > 2000 and len(contour) < 5000)]

    # only take contours that fully encircle the center
    fewer_contours = [contour for contour in contours
                      if contour_encircles_center(bin_cards, contour)]

    return fewer_contours

def get_approx(card_outline: np.ndarray, multiple: float):
    """
    Given a card outline and a multiple, get a polynomial
    approximation in points
    """
    card_outline_len = cv2.arcLength(card_outline, True)
    epsilon = multiple * card_outline_len
    return cv2.approxPolyDP(card_outline, epsilon, True)

class ContourFinder:
    def __init__(self, img: np.ndarray):
        self.img = img

        # save original contours and make a copy that will be worked through
        self.all_contours = [
            np.array([row.T[np.newaxis, :].round() for row in contour]).astype(np.int32)
            for contour in find_contours(self.img)
        ]
        self.contours = self.all_contours.copy()

        # get a few points aronud the center of the contour
        h, w = img.shape
        self.img_center_points = [(w // 2, h // 2),
                                  (w // 2 - w // 6, h // 2 + h // 6),
                                  (w // 2 - w // 6, h // 2 - h // 6),
                                  (w // 2 + w // 6, h // 2 + h // 6),
                                  (w // 2 + w // 6, h // 2 - h // 6),
                                  (w // 2 + w // 6, h // 2),
                                  (w // 2 - w // 6, h // 2),
                                  (w // 2,          h // 2 + h // 6),
                                  (w // 2,          h // 2 - h // 6)]
        self.h, self.w = h, w

    def get_card_outline(self, plot: bool = True, fig_sup_title: Union[None, str] = None, **kwargs):
        
        if plot:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            plt.suptitle(fig_sup_title)

        self.limit_contours_by_length(**kwargs)

        if plot:
            self.plot_contours(axes[0])
            axes[0].set_title(f'All contours (N = {len(self.contours)})')

        self.limit_contours_by_encircle(**kwargs)

        if plot:
            self.plot_contours(axes[1])
            axes[1].set_title(f'Contours that encircle center (N = {len(self.contours)})')

        self.limit_contours_by_four_point_approx(**kwargs)

        if plot:
            self.plot_contours(axes[2], approxes=True)
            axes[2].set_title(f'Contours with decent 4-point approx. (N = {len(self.contours)})')

            plt.tight_layout()
            plt.savefig('tmp_img.png')
            plt.close('all')
            display(Image(filename='tmp_img.png'))
            clear_output(wait=True)

        return self.get_best_contour_and_approx()
    
    def limit_contours_by_length(self, **kwargs):
        self.contours = [contour for contour in self.contours
                         if self.contour_has_right_length(contour, **kwargs)]
    
    def limit_contours_by_encircle(self, **kwargs):
        self.contours = [contour for contour in self.contours
                         if self.contour_encircles(contour, self.img_center_points, **kwargs)]

    def limit_contours_by_four_point_approx(self, **kwargs):
        self.get_four_point_approxes(**kwargs)
        self.contours = [contour for i, contour in enumerate(self.contours)
                         if len(self.approxes[i])]
        self.approxes = [approx for approx in self.approxes if len(approx)]
    
    def get_four_point_approxes(self, **kwargs):
        self.approxes = [self.get_four_point_approx(contour, **kwargs)
                         for contour in self.contours]

    def get_best_contour_and_approx(self):
        center = np.array([self.w // 2, self.h // 2])
        min_distances = [min([np.linalg.norm(point - center) for point in contour[:, 0]])
                         for contour in self.contours]

        assert len(self.contours) == len(self.approxes)

        try:
            return [self.contours[np.argmin(min_distances)]], [self.approxes[np.argmin(min_distances)]]
        except ValueError:
            return [], []

    @staticmethod
    def contour_has_right_length(contour: np.ndarray, upper_lim: int = 5000, lower_lim: int = 1500):
        return (len(contour) > lower_lim and len(contour) < upper_lim)
    
    def contour_encircles(self, contour: np.ndarray, points: list, num_points_to_encircle: int = 3):
        return  sum(self.bool_encircled_points(contour, points)) >= num_points_to_encircle

    @staticmethod
    def bool_encircled_points(contour: np.ndarray, points: list):
        polygon = Polygon([(j, i) for i, j in contour[:, 0]])
        return [polygon.contains(Point(*point)) for point in points]

    def get_four_point_approx(self, contour: np.ndarray, **kwargs) -> np.ndarray:

        for multiple in np.arange(0.003, 0.3, 0.001):
            
            # get approximation
            approx = get_approx(contour, multiple)

            # if desired length, check that the points are good
            if len(approx) == 4:
                if self.approx_is_correct_shape(approx, **kwargs):
                    return approx

        return np.array([])

    def approx_is_correct_shape(self, approx, check_short_side: bool = False):
        right_points, left_points = separate_left_and_right_points(approx)
        for upper_corner, lower_corner in [right_points, left_points]:
            if not abs(np.linalg.norm(upper_corner.ravel() - lower_corner.ravel()) - 625) < 70:
                if self.h - 1 in lower_corner.ravel():
                    continue
                return False
            
            if (approx[:, :, 1] == self.w - 1).any():
                return False
            
            if (approx[:, :, 1] == 0).any():
                return False

            if (approx[:, :, 0] == self.h - 1).any():
                return False

        if check_short_side:
            for right, left in zip(right_points, left_points):
                pass
            
        return True

    def plot_contours(self, ax, approxes: bool = False):
        ax.imshow(self.img, cmap='gray')

        for point in self.img_center_points:
            ax.plot(point[0], point[1], c='gray', marker='+', ms=20)

        colors = ['r', 'g', 'b', 'c', 'm', 'y']

        for i, c in enumerate(self.contours):
            j = i % len(colors)
            ax.plot(c[:, :, 1], c[:, :, 0], c=colors[j], ls='-', lw=2)
        
        if approxes:
            for i, a in enumerate(self.approxes):
                j = i % len(colors)
                right_points, left_points = separate_left_and_right_points(a)
                ax.plot(right_points[:, :, 1], right_points[:, :, 0], c=colors[j], ls='', marker='*', ms=30, mfc='None')
                ax.plot(left_points[:, :, 1], left_points[:, :, 0], c=colors[j], ls='', marker='*', ms=30, mfc='None')

def get_outline(h_img: np.ndarray, k, verbose: bool = False, plot: bool = False) -> Tuple[list, list, list]:
    """
    Get the outline of two cards lying on top of one another 
    """

    # init loop
    limit = 0.04
    try_next_limit = True

    while try_next_limit:

        # binarize cards and perform a little morphology to get a clearer outline
        bin_cards = np.where(h_img < limit, 0, 1)   

        # morphology
        hey = binary_erosion(bin_cards, disk(2))

        for i in range(1, min([int(limit * 200), 12])):
            
            contour_finder = ContourFinder(hey)

            # get card outline while plotting whats up
            try:
                card_outline, approx = contour_finder.get_card_outline(
                                            fig_sup_title=f'K = {k}, LIMIT = {limit}, AT EROSION #{i}',
                                            plot=plot
                )
            except Exception:
                continue

            # if it worked, then break out of both loops
            if len(card_outline):
                try_next_limit = False
                break

            # otherwise try more morphology
            hey = binary_erosion(hey, disk(2))
        
        limit += 0.01

        if limit > 0.15:
            return [], [], hey

    return card_outline, approx, hey

def get_four_point_approx(card_outline: np.ndarray):
    """
    Given a card outline, get the four corner points of the outline
    """
    multiple = 0.003
    approx = []
    points_are_bad = True
    while (len(approx) != 4 or points_are_bad):

        # stop loop if we've gone too far
        if multiple > 0.3:
            raise CouldNotFind4PointApprox
        
        # get approximation
        approx = get_approx(card_outline, multiple)

        # if desired length, check that the points are good
        if len(approx) == 4:
            print('found a 4-point approx!')
            points_are_bad = False if check_four_point_approx(approx) else True
            print('points bad?', points_are_bad)
        
        # increase multiple
        multiple += 0.0005

    return approx

def check_four_point_approx(approx: np.ndarray, verbose: bool = False):
    """
    Check that the four point approximation is the correct shape
    """
    right_points, left_points = separate_left_and_right_points(approx)

    if verbose:
        print('right points:', right_points)
        print('left points:', left_points)

    for a, b in [right_points, left_points]:
        if not abs(np.linalg.norm(a.ravel() - b.ravel()) - 625) < 25:
            if verbose:
                print('falsity!')
                print(a.ravel())
                print(b.ravel())
                print(np.linalg.norm(a.ravel() - b.ravel()))
            return False
    return True


def get_individual_outlines(card_outline: np.ndarray, approx: np.ndarray):
    """
    Given an outline of overlapping cards and it's four corners, extract
    the individual cards
    """

    # separate the left and the right points
    right_points, left_points = separate_left_and_right_points(approx)

    card_outlines, plot_outlines = [], []
    for i, points in enumerate([right_points, left_points]):
        outline, plot_outline = get_opposite_points(points, right={0: True, 1: False}[i])
        card_outlines += [outline]
        plot_outlines += [plot_outline]
    
    return card_outlines, plot_outlines

def segment_cards(imgs: np.ndarray, k, h_imgs: np.ndarray, make_plot: bool = True, verbose: bool = False):

    if not len(imgs):
        make_plot = False

    # cut the cards on the left-hand side
    if make_plot:
        fig, ax = plt.subplots(len(imgs), 3, figsize=(15, len(imgs) * 4))
        fig.suptitle(k)
        if len(ax.shape) == 1:
            ax = ax[np.newaxis, :]

    card_outlines, plot_outlines = [], []
    for j in range(len(imgs)):
        
        # check whether or not it is a table card
        table_card = j >= (len(imgs) - 5)

        # get card outline
        card_outline, approx, hey = get_outline(h_imgs[j], k, verbose=verbose, plot=False)

        assert isinstance(card_outline, list)
        assert isinstance(approx, list)

        if make_plot:
            ax[j, 0].imshow(hey, cmap='gray')
            ax[j, 1].imshow(hey, cmap='gray')

            # plot contour and polygon
            ax[j, 1].imshow(hey, cmap='gray')

            if len(card_outline):
                for card, app in zip(card_outline, approx):
                    ax[j, 1].plot(card[:, :, 1].ravel(), card[:, :, 0].ravel(), 'r-', lw=5)
                    ax[j, 1].plot(app[:, :, 1], app[:, :, 0], 'c*', ms=30)

        if len(card_outline):

            assert len(card_outline) == 1
            assert len(approx) == 1

            if not table_card:
                card_outlines_, _ = get_individual_outlines(*card_outline, *approx)
            else:
                card_outlines_ = card_outline

            card_outlines += card_outlines_


        else:
            card_outlines += [np.array([])] * (1 if table_card else 2)

        # plot
        if len(card_outline):
            if make_plot:
                ax[j, 2].imshow(hey, cmap='gray')
                for outline in card_outlines_:
                    ax[j, 2].plot(outline[:, :, 1], outline[:, :, 0], '-*c' if not table_card else '-c', ms=30)
                ax[j, 0].set_ylabel(f'Cards {j}')
    
    if make_plot:
        _ = ax[0, 0].set_title('Use morphology to\nharden the card outline')
        _ = ax[0, 2].set_title('Use angle between\npoints to get rectangles')
        _ = ax[0, 1].set_title('Get contour and 4-point\npolynomial approximation')

    assert len(card_outlines) == (len(imgs) - 5) * 2 + 5
    assert all([isinstance(outline, np.ndarray) for outline in card_outlines])

    return card_outlines

def get_boxes_and_rectangles(card_outlines: List[np.ndarray]):
    """
    Get a list of boxes and rectangles from a list of card outlines
    """
    boxes, rects = [], []
    for contour in card_outlines:
        if len(contour):
            new_contour = np.roll(contour, 1, axis=2)
            rects += [cv2.minAreaRect(new_contour)]
            box = cv2.boxPoints(rects[-1])
            boxes += [np.int0(box)]
        else:
            rects += [np.array([])]
            boxes += [np.array([])]
    return boxes, rects

def warp_card(rect: tuple, box: np.ndarray, img: np.ndarray):
    """
    Warp card so that it's straight up and down
    """
    # get width and height of the detected rectangle
    width, height = int(rect[1][0]), int(rect[1][1])
    
    # coordinates of the box
    src_pts = box.astype("float32")

    # coordinate of the points in box points after the rectangle has been
    # straightened
    dst_pts = np.array([[0, height-1],
                        [0, 0],
                        [width-1, 0],
                        [width-1, height-1]], dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    return cv2.warpPerspective(img, M, (width, height))

def straighten_card(warped: np.ndarray):
    """
    Check if a card is laying on its side and straighten it if nessecary
    """
    if warped.shape[0] < warped.shape[1]:
        warped = np.flip(np.stack([warped[:, :, i].T for i in range(3)], -1), 1)
    return warped

def segment_number_and_suit(warped: np.ndarray, verbose: bool = False):
    """
    Given a flat, binarized card, segment the number in the upper right corner
    """

    # make image binary and apply a median filter to crop number and suit
    bin_im = np.where(rgb2gray(warped)[:, :] > 0.8, 1, 0)
    bin_im = median(bin_im)
    
    # Get the first place coordinates where the card has a marking
    x = np.where(bin_im.T[:120, :120] == 0)[0][0]
    y = np.where(bin_im[:120, :120] == 0)[0][0]

    tally = 0
    no_error = True

    # if x is very small, we have not hit the numebr but something
    # else - so, iteratively increase x - if x gets too big, then
    # just set x to 0 
    while x < 3:
        tally += 3
        try:
            x = np.where(bin_im.T[tally:130, tally:130] == 0)[0][0]
        except IndexError:
            x = 0
            no_error = False
            break
    x += tally if no_error else 0

    tally = 0
    no_error = True

    # same with y (as above)
    while y < 3:
        tally += 3
        try:
            y = np.where(bin_im[tally:130, x:130] == 0)[0][0]
        except IndexError:
            y = 0
            no_error = False
            break
    y += tally if no_error else 0

    # find the bottom and rightmost edge of the number
    x_ = np.argmax(bin_im[y:120, x:120].sum(axis=0))
    y_ = np.argmax(bin_im[y:120, x:x + x_].sum(axis=1))

    # same pattern
    tally = 0
    while x_ < 20:
        tally += 3
        x_ = np.argmax(bin_im[y:120, x + tally:120].sum(axis=0))
    x_ += tally

    # same pattern
    tally = 0
    while y_ < 40:
        tally += 3
        y_ = np.argmax(bin_im[y + tally:120, x:x + x_].sum(axis=1))
    y_ += tally

    ys = np.where(bin_im[y + y_:200, x:x + x_] == 0)[0][0]
    # same pattern
    tally = 0
    while ys < 5:
        tally += 3
        ys = np.where(bin_im[y + y_ + tally:200, x:x + x_] == 0)[0][0]
    ys += tally

    ys_ = np.argmax(bin_im[y + y_ + ys:200, x:x + x_].sum(axis=1))
    # same pattern
    tally = 0
    while ys_ < 45:
        tally += 3
        ys_ = np.argmax(bin_im[y + y_ + ys + tally:200, x:x + x_].sum(axis=1))
    ys_ += tally

    if verbose:
        print(f"""
        {x}, {y} --- {x + x_}
         |              |
         |              |
         |              |
        {x}, {y + y_} --- {x + x_}, {y + y_}
         |              |
        {x}, {y + y_ + ys} --- {x + x_, y_ + ys}
         |
         |
        {x}, {y + y_ + ys + ys_} ---
        """)

    return x, x_, y, y_, ys, ys_

# def get_numbers_and_suits(card_outlines: List[np.ndarray], imgs: List[np.ndarray], make_plot: bool = True,
#                           verbose: bool = False, fig_sup_title: Union[str, None] = None):
#     """
#     Given a list of card outlines, semgent the numbers and suits of each outline
#     """
#     boxes, rects = get_boxes_and_rectangles(card_outlines)

#     if make_plot:
#         fig, axes = plt.subplots(5, 8, figsize=(15, 15))
#         fig.suptitle(fig_sup_title)

#     nums, types, num_signs = [], [], []
#     for i, (rect, box) in enumerate(zip(rects, boxes)):

#         warped = warp_card(rect, box, imgs[i // 2])

#         if make_plot:
#             # plot the straightened rectangles
#             axes[0, i].imshow(warped)
#             axes[0, i].set_title(f'{warped.shape[1]} X {warped.shape[0]}')
#             axes[0, i].set_xticks([])
#             axes[0, i].set_yticks([])
#             axes[0, 0].set_ylabel("WARPED CARDS\n", fontsize=12)

#         warped = straighten_card(warped)

#         # show the straightened rectangles
#         if make_plot:
#             axes[1, i].imshow(warped)
#             for j in range(1, 3):
#                 axes[j, i].set_title(f'{warped.shape[1]} X {warped.shape[0]}')
#                 # axes[j, i].set_xticks([])
#                 # axes[j, i].set_yticks([])
#             axes[1, 0].set_ylabel("WARPED AND\nFLIPPED CARDS\n", fontsize=12)

#         x, x_, y, y_, ys, ys_ = segment_number_and_suit(warped, verbose=verbose)

#         edge = min(x, y, 5)
#         number = warped[y - edge:y + y_ + edge, x - edge:x + x_ + edge, :]
#         suit = warped[y + y_ + ys - edge:y + y_ + ys + ys_ + edge, x - edge:x + x_ + edge, :]

#         num_signs += [warped]
#         nums += [number]
#         types += [suit]

#         if make_plot:

#             # plot the crop lines
#             axes[2, i].imshow(warped[:150, :150])
#             axes[2, i].axvline(x=x, c='c')
#             axes[2, i].axvline(x=x + x_, c='c')
#             axes[2, i].axhline(y=y, c='m')
#             axes[2, i].axhline(y=y + y_, c='m')
#             axes[2, i].axhline(y=y + y_ + ys, c='m')
#             axes[2, i].axhline(y=y + y_ + ys + ys_, c='m')
#             axes[2, 0].set_ylabel("CROP LINES FOR\nNUMBER AND SUIT\n", fontsize=12)

#             # crop the numbers and show them
#             axes[3, i].imshow(nums[-1])
#             axes[3, i].set_xticks([])
#             axes[3, i].set_yticks([])
#             axes[3, i].set_title(f'{nums[-1].shape[1]} X {nums[-1].shape[0]}')
#             axes[3, 0].set_ylabel("CROPPED NUMBER\n", fontsize=12)

#             # crop the suit and show them
#             axes[4, i].imshow(types[-1])
#             axes[4, i].set_xticks([])
#             axes[4, i].set_yticks([])
#             axes[4, i].set_title(f'{types[-1].shape[1]} X {types[-1].shape[0]}')
#             axes[4, 0].set_ylabel("CROPPED SUIT\n", fontsize=12)

#     return nums, types, num_signs


def process_cutouts(nums: List[np.ndarray], make_plot: bool = True):
    """
    Given a list of images of numbers or quits, process the images
    and return the results, including the contours
    """

    if make_plot:
        _, axes = plt.subplots(4, 8, figsize=(20, 12))

    all_contours = []
    bin_trans = []
    for i in range(len(nums)):
        
        if make_plot:
            # show the black and white numbers
            axes[0, i].imshow(rgb2gray(nums[i]), cmap='gray')
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])
            axes[0, 0].set_ylabel("WHITE & BLACK\nNUMBERS\n", fontsize=12)
        
        # apply a gaussian blur
        trans = gaussian(rgb2gray(nums[i]))
        if make_plot:
            axes[1, i].imshow(trans, cmap='gray')
            axes[1, i].set_xticks([])
            axes[1, i].set_yticks([])
            axes[1, 0].set_ylabel("WHITE & BLACK WITH\nGAUSS. BLUR\n", fontsize=12)
        
        # transform to binary and perform and opening
        bin_ = np.where(trans > 0.8, 1, 0)
        bin_trans += [opening(bin_)]
        if make_plot:
            for j in range(2, 4):
                axes[j, i].imshow(bin_trans[-1], cmap='gray')
                axes[j, i].set_xticks([])
                axes[j, i].set_yticks([])
        
        # pad box and get the contours and show the longest of these
        bin_trans[-1] = np.pad(bin_trans[-1], 3, constant_values=1)
        contours = find_contours(bin_trans[-1])
        # c = contours[np.argmax([len(contour) for contour in contours])]
        all_contours += [contours]
        if make_plot:
            for c in contours:
                axes[3, i].imshow(bin_trans[-1], cmap='gray')
                axes[3, i].plot(c[:, 1], c[:, 0], 'r')
                axes[2, 0].set_ylabel("BINARY\n", fontsize=12)
                axes[3, 0].set_ylabel("BINARY WITH OUTLINE\n", fontsize=12)
    
    return bin_trans, all_contours

def load_and_process_full_image(img: Union[int, np.ndarray], make_plot: bool = True):
    """
    Load and process the full image, returning the cutouts at the end
    """
    
    # get training images
    train_images = load_data(os.path.join("data", "train"))

    # load training and setup images
    if isinstance(img, int):
        img = np.array(train_images[list(train_images.keys())[img]])

    # use the first training image as a reference
    # im = np.array(train_images[list(train_images.keys())[0]])

    # match histograms with reference
    # if not (img == im).all():
    #     img = match_histograms(img, im).round().astype(np.uint8)
    
    # deskew, transform
    img = deskew(img)
    h_ = rgb2hsv(img)

    if make_plot:
        _, ax = plt.subplots(1, 2, figsize=(15, 15))
        ax[0].imshow(h_[:, :, 1])
        ax[0].set_title('"h" part of the rgb --> hsv transform')
        ax[1].imshow(img)
        ax[1].set_title('Original image')
        plt.show()

    imgs = area_partition(img)
    hsv_imgs = area_partition(h_)

    return imgs, hsv_imgs

def cut_table_into_imgs(h_table: np.ndarray, table: np.ndarray):
    w = table.shape[1]
    lines = np.arange(0, w, w // 6)

    ims = []
    h_ims = []

    for right_side, left_side in zip(lines[:5], lines[2:]):
        ims += [
            table[200:, right_side:left_side, :]
        ]
        h_ims += [
            h_table[200:, right_side:left_side]
        ]    
    return h_ims, ims

def get_players_and_imgs(hsv_imgs, all_imgs, make_plot: bool = True, verbose: bool = False,
                         axes: Union[None, np.ndarray] = None):

    hh_imgs = [im[:, :, 0] for im in hsv_imgs]
    ss_imgs = [im[:, :, 1] for im in hsv_imgs]
    
    players, _, _, _, _ = find_players(hh_imgs, all_imgs, verbose=verbose)

    # get table imgs
    s_table_cards, table_cards = cut_table_into_imgs(ss_imgs[4], all_imgs[4])

    if make_plot:
        if not isinstance(axes, np.ndarray):
            _, axes = plt.subplots(2, len(all_imgs[:4]) + len(table_cards), figsize=(15, 8))
        for ax_, im_ in zip(axes[0], all_imgs[:4] + table_cards):
            ax_.imshow(im_)
            ax_.set_yticks([])
            ax_.set_xticks([])
        for ax_, im_ in zip(axes[1], ss_imgs[:4] + s_table_cards):
            ax_.imshow(im_)
            ax_.set_yticks([])
            ax_.set_xticks([])
        axes[0, 3].set_title(f'players = {", ".join([str(p) for p in players])}')

    return (players,
            np.array(all_imgs, dtype='object')[players].tolist() + table_cards,
            np.array(ss_imgs, dtype='object')[players].tolist() + s_table_cards)

class NumberCutter:
    def __init__(self, card_img, h: int = 200, w: int = 120):

        assert len(card_img.shape) == 3
        assert card_img.shape[2] == 3

        self.card_img = card_img

        bin_im = np.where(rgb2gray(card_img) > 0.8, 1, 0)
        bin_im = median(bin_im)

        self.h = h
        self.w = w

        self.img = bin_im[:h, :w]

        self.upper_points = []
        self.lower_points = []

    def get_right_side_of_number(self, starting_point: int = 25, plot: bool = False, ax = None):
        self.get_starting_points(plot, ax, starting_point=starting_point)
        for _ in range(65):
            if self.lower_points[-1][1] == 150:
                break
            self.trace_rightwards()
        self.plot_trace(ax)
        
        arg_lowest_point = np.argmax([p[0] for p in self.lower_points])
        
        start_search_upper = self.upper_points[arg_lowest_point]
        start_search_lower = self.lower_points[arg_lowest_point]
        
        assert start_search_lower[1] == start_search_upper[1]
        
        search_xs = [start_search_lower[1]]
        search_y_uppers = [start_search_upper[0]]
        search_y_lowers = [start_search_lower[0]]
        self.slices = [self.img[search_y_uppers[-1]:search_y_lowers[-1] + 1, search_xs[-1]]]

        if plot:
            ax.plot(np.repeat(search_xs[-1], search_y_lowers[-1] - search_y_uppers[-1]  + 1),
                    np.arange(search_y_uppers[-1], search_y_lowers[-1] + 1), 'm-', lw=1, alpha=0.5)

        while self.slice_is_white(padding=3):
            search_xs +=  [search_xs[-1] - 1]
            search_y_uppers += [copy(search_y_uppers[-1])]
            search_y_lowers += [copy(search_y_lowers[-1])]
            self.slices += [self.img[search_y_uppers[-1]:search_y_lowers[-1] + 1, search_xs[-1]]]
            slice_upper_bound = int(np.argwhere(self.slices[-1])[0])
            slice_lower_bound = -int(np.argwhere(self.slices[-1][::-1])[0])
            self.slices[-1] = self.slices[-1][slice_upper_bound:(slice_lower_bound if slice_lower_bound else len(self.slices[-1]))]
            search_y_uppers[-1] += slice_upper_bound
            search_y_lowers[-1] += slice_lower_bound
            if plot:
                ax.plot(np.repeat(search_xs[-1], search_y_lowers[-1] - search_y_uppers[-1]  + 1),
                        np.arange(search_y_uppers[-1], search_y_lowers[-1] + 1), 'm-', lw=1, alpha=0.5)
        if plot:
            ax.axvline(search_xs[-1] + 1, ls='--', lw=4, c='c')
        self.right_number_edge = search_xs[-1] + 1


        while not self.slice_is_white() or self.right_number_edge - search_xs[-1] < 40:
            if search_xs[-1] == 0:
                break
            search_xs +=  [search_xs[-1] - 1]
            search_y_uppers += [copy(search_y_uppers[-1])]
            search_y_lowers += [copy(search_y_lowers[-1])]
            self.slices += [self.img[search_y_uppers[-1]:search_y_lowers[-1] + 1, search_xs[-1]]]
            slice_upper_bound = int(np.argwhere(self.slices[-1])[0])
            slice_lower_bound = -int(np.argwhere(self.slices[-1][::-1])[0])
            self.slices[-1] = self.slices[-1][slice_upper_bound:(slice_lower_bound if slice_lower_bound else len(self.slices[-1]))]
            search_y_uppers[-1] += slice_upper_bound
            search_y_lowers[-1] += slice_lower_bound
            if plot:
                ax.plot(np.repeat(search_xs[-1], search_y_lowers[-1] - search_y_uppers[-1]  + 1),
                        np.arange(search_y_uppers[-1], search_y_lowers[-1] + 1), 'y-', lw=1, alpha=0.5)
        if plot:
            ax.axvline(search_xs[-1] - 1, ls='--', lw=4, c='c')
        self.left_number_edge = max(0, search_xs[-1] - 1)

        self.top_of_square = max(search_y_uppers)
        self.slice_ = self.img[self.top_of_square + 1:, self.left_number_edge:self.right_number_edge + 1]

        self.top_of_number = np.argwhere(self.slice_.sum(axis=1) < self.slice_.shape[1] - 3).ravel()[0] + self.top_of_square
        if plot:
            ax.plot(np.arange(self.left_number_edge, self.right_number_edge + 1),
                    np.repeat(self.top_of_number, self.right_number_edge - self.left_number_edge + 1),
                    ls='--', lw=4, c='c') 
        
        slice_ = self.img[self.top_of_number + 1:, self.left_number_edge:self.right_number_edge + 1]

        self.number_and_suit_joined = False
        self.bottom_of_number = np.argwhere(slice_.sum(axis=1) == slice_.shape[1]).ravel()[0] + self.top_of_number

        if self.bottom_of_number > 80 + self.top_of_number:
            self.bottom_of_number = 80 + self.top_of_number
            self.number_and_suit_joined = True
        
        if plot:
            ax.plot(np.arange(self.left_number_edge, self.right_number_edge + 1),
                    np.repeat(self.bottom_of_number, self.right_number_edge - self.left_number_edge + 1),
                    ls='--', lw=4, c='c') 

        num =  np.pad(self.img[self.top_of_number:self.bottom_of_number + 1, self.left_number_edge:self.right_number_edge], 3, constant_values=1)

        if self.number_and_suit_joined:
            self.top_of_suit = self.bottom_of_number
        else:
            slice_ = self.img[self.bottom_of_number + 1:, self.left_number_edge:self.right_number_edge + 1]
            self.top_of_suit = np.argwhere(slice_.sum(axis=1) < slice_.shape[1] - 3).ravel()[0] + self.bottom_of_number

        if plot:
            ax.plot(np.arange(self.left_number_edge, self.right_number_edge + 1),
                    np.repeat(self.top_of_suit, self.right_number_edge - self.left_number_edge + 1),
                    ls='--', lw=4, c='c')
        
        slice_ = self.img[self.top_of_suit + 1:, self.left_number_edge:self.right_number_edge + 1]
        self.bottom_of_suit = np.argwhere(slice_.sum(axis=1) == slice_.shape[1]).ravel()[0] + self.top_of_suit

        if self.bottom_of_number > 50 + self.top_of_number:
            self.bottom_of_number = 50 + self.top_of_number
        
        if plot:
            ax.plot(np.arange(self.left_number_edge, self.right_number_edge + 1),
                    np.repeat(self.bottom_of_suit, self.right_number_edge - self.left_number_edge + 1),
                    ls='--', lw=4, c='c')

        
        slice_ = self.img[self.top_of_suit + 1:self.bottom_of_suit + 1, :self.right_number_edge + 1]
        self.right_suit_edge = -np.argwhere(slice_.sum(axis=0)[::-1] < slice_.shape[0] - 3).ravel()[0] + self.right_number_edge

        slice_ = self.img[self.top_of_suit + 1:self.bottom_of_suit + 1, :self.right_suit_edge + 1]
        try:
            self.left_suit_edge = -np.argwhere(slice_.sum(axis=0)[::-1] == slice_.shape[0]).ravel()[0] + self.right_suit_edge
        except IndexError:
            self.left_suit_edge = 0

        suit = np.pad(self.img[self.top_of_suit:self.bottom_of_suit + 1, self.left_suit_edge:self.right_suit_edge + 3], 3, constant_values=1)


        return num, suit

    
    def slice_is_white(self, padding: int = 0):
        return bool(sum(self.slices[-1]) >= len(self.slices[-1].ravel()) - padding)

    def get_starting_points(self, plot: bool = False, ax = None, starting_point: int = 25):
        
        assert ax if plot else True

        # get the point in the upper left corner
        upper_point = [starting_point, self.w - 1]
        lower_point = [starting_point, self.w - 1]

        while self.img[tuple(upper_point)]:
            if (upper_point[0] == 0) or (not self.img[upper_point[0] - 1, upper_point[1]]):
                break
            upper_point[0] -= 1

        while self.img[tuple(lower_point)]:
            if (lower_point[0] == 150) or (not self.img[lower_point[0] + 1, upper_point[1]]):
                break
            lower_point[0] += 1

        self.upper_points += [upper_point]
        self.lower_points += [lower_point]

        if plot:
            ax.plot(*np.roll(upper_point, 1), marker=5, ms=30, mfc='None', c='r')
            ax.plot(*np.roll(lower_point, 1), marker=5, ms=30, mfc='None', c='g')
    
    def trace_rightwards(self):

        upper_point = self.upper_points[-1]
        lower_point = self.lower_points[-1]

        next_upper_point = [upper_point[0], upper_point[1] - 1]
        next_lower_point = [lower_point[0], lower_point[1] - 1]

        while self.img[tuple(next_upper_point)]:
            if (next_upper_point[0] == 0) or (self.img[next_upper_point[0] - 1, next_upper_point[1]]):
                break
            next_upper_point[0] -= 1
        
        while not self.img[tuple(next_upper_point)]:
            if next_upper_point[0] == 25:
                break
            next_upper_point[0] += 1

        while self.img[tuple(next_lower_point)]:
            if (next_lower_point[0] == 150) or (not self.img[next_lower_point[0] + 1, next_lower_point[1]]):
                break
            next_lower_point[0] += 1
        
        while not self.img[tuple(next_lower_point)]:
            if next_lower_point[0] == next_upper_point[0]:
                break
            next_lower_point[0] -= 1

        self.upper_points += [next_upper_point]
        self.lower_points += [next_lower_point]

    def plot_trace(self, ax = None):
        ax.plot([p[1] for p in self.upper_points], [p[0] for p in self.upper_points],
                    ls='-', lw=4, c='r')
        ax.plot([p[1] for p in self.lower_points], [p[0] for p in self.lower_points],
                    ls='-', lw=4, c='g')
        arg_lowest_point = np.argmax([p[0] for p in self.lower_points])
        ax.plot(self.lower_points[arg_lowest_point][1], self.lower_points[arg_lowest_point][0],
                    marker='*', c='g', ms=30, mfc='None')
        ax.plot(self.upper_points[arg_lowest_point][1], self.upper_points[arg_lowest_point][0],
                    marker='*', c='g', ms=30, mfc='None')


def match_number_and_suit(card: np.ndarray, numbers: dict, suits: dict, make_plot: bool = False, axes = None):
    cutter = NumberCutter(card)

    if axes is not None:
        assert axes.shape == (3,)

    if make_plot:
        axes[0].imshow(cutter.img, cmap='gray')
    
    error = True
    starting_point = 25
    while error:
        try:
            number_im, suit = cutter.get_right_side_of_number(plot=True, ax=axes[0], starting_point=starting_point)
            error = False
        except Exception:
            if starting_point == 50:
                starting_point = 15
            if starting_point == 15:
                raise IndexError
            if starting_point == 25:
                starting_point = 50
    
    best_match = 0
    for k, im in numbers.items():
        resized_truth = resize(im, number_im.shape, preserve_range=True).round()

        match = (resized_truth == number_im).sum()
        if match > best_match:
            best_match = match
            best_number = k

    best_match = 0
    for k, im in suits.items():
        resized_truth = resize(im, suit.shape, preserve_range=True).round()
        
        match = (resized_truth == suit).sum()
        if match > best_match:
            best_match = match
            best_suit = k
    
    if best_number in [11, 2]:
        if (number_im[:25, :10] == 1).sum() > 25 * 10 - 10:
            best_number = 11
        else:
            best_number = 2

    if make_plot:
        axes[1].imshow(number_im, cmap='gray')
        axes[1].set_title(best_number)
        axes[2].imshow(suit, cmap='gray')
        axes[2].set_title(best_suit)
    for j in range(3):
        axes[j].set_xticks([])
        axes[j].set_yticks([])
    
    return best_number, best_suit

def get_numbers_and_suits(card_outlines: List[np.ndarray], imgs: List[np.ndarray], real_numbers: dict,
                          real_suits: dict, make_plot: bool = True):

    cards = []
    boxes, rects = get_boxes_and_rectangles(card_outlines)
    for i, (rect, box) in enumerate(zip(rects, boxes)):

        num_players = len(imgs) - 5
        table_card = i >= num_players*2

        if len(box):
            img = imgs[i - num_players] if table_card else imgs[i // 2]
            warped = warp_card(rect, box, img)
            cards += [gaussian(straighten_card(warped))]
        else:
            cards += [[]]

    numbers, suits = [], []
    for card in cards:
        if len(card):
            ax = None
            if make_plot:
                fig, ax = plt.subplots(1, 3, figsize=(5, 3))
            try:
                best_number, best_suit = match_number_and_suit(card, real_numbers, real_suits, make_plot, ax)
            except Exception:
                best_number = np.random.default_rng().integers(1, 14)
                best_suit = ['diamond', 'spade', 'heart', 'clover'][np.random.default_rng().integers(0, 4)]
        else:
            best_number = np.random.default_rng().integers(1, 14)
            best_suit = ['diamond', 'spade', 'heart', 'clover'][np.random.default_rng().integers(0, 4)]
        numbers += [best_number]
        suits += [best_suit]
        if make_plot:
            plt.tight_layout()
            plt.savefig('tmp_img.png')
            plt.close('all')
            display(Image(filename='tmp_img.png'))
            plt.pause(2)
            clear_output(wait=True)
    return numbers, suits


def get_table_contours(h_table: np.ndarray, make_plot: bool = False):

    # binarize images and perform a closing
    bin_fig = opening(np.where(h_table < 0.15, 1, 0))

    # do region growing
    new_fig = region_growing(bin_fig, (0.5, 1.))

    # get contours using cv2 instead
    cs, _ = cv2.findContours(np.where(new_fig, 0, 1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # separate the card contours
    cs1 = [c for c in cs if c.shape[0] > 1800]

    # draw the contours to check them
    # create an empty image for contours
    if make_plot:
        _, ax = plt.subplots(1, 1, figsize=(8, 4))
        img_contours = np.zeros_like(new_fig)
        cv2.drawContours(img_contours, cs1, -1, 1, 10)
        ax.imshow(img_contours)
    
    # box the contours
    boxes, rects = [], []
    for contour in cs1:
        rects += [cv2.minAreaRect(contour)]
        box = cv2.boxPoints(rects[-1])
        boxes += [np.int0(box)]
    
    # make boxes into contour
    return [np.array([row.T[np.newaxis, :].round() for row in contour]).astype(np.int32)
            for contour in boxes]



def predict(image):
    """
    Predict all the required parameters.

    Inputs:
    -------
    image: numpy array image of shape (h, w, 3)

    Ouputs:
    -------
    Dictionary with the results.
    """    
    imgs, hsv_imgs = load_and_process_full_image(image, make_plot = False)
    out = deskew(image)
    fiches_area = area_partition(deskew(image))[-1]
    fiches = count_fiches(fiches_area)

    players, card_imgs, s_card_imgs = get_players_and_imgs(hsv_imgs, imgs, make_plot=False)
    card_outlines = segment_cards(card_imgs, "", s_card_imgs, make_plot=False)
    
    with open('numbers.pickle', 'rb') as handle:
        real_numbers = pickle.load(handle)
    
    with open('suits.pickle', 'rb') as handle:
        real_suits = pickle.load(handle)

    nums, suits = get_numbers_and_suits(card_outlines, card_imgs, real_numbers, real_suits, make_plot=True)

    prediction = {}
    keys = [
        'P11', 'P12', 'P21', 'P22', 'P31', 'P32', 'P41', 'P42',
        'T1', 'T2', 'T3', 'T4', 'T5'
    ]

    assert len(nums) == len(suits)

    new_nums = nums[::-1]
    new_suits = suits[::-1]

    for k in keys:
        if k.startswith('P'):
            player = int(k[1]) - 1
            if player not in players:
                prediction[k] = 0
                continue
        n = new_nums.pop()
        s = new_suits.pop()
        prediction[k] = str(n) + s[0].upper()

    prediction["CR"] = fiches["red"]
    prediction["CG"] = fiches["green"]
    prediction["CB"] = fiches["blue"]
    prediction["CK"] = fiches["black"]
    prediction["CW"] = fiches["white"]

    return prediction

    

    




