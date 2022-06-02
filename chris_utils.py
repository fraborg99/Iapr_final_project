from turtle import left
from typing import List, Union
import os

import numpy as np
from scipy.fft import fft
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from skimage.transform import rotate
from skimage.morphology import binary_erosion, dilation, opening, closing, disk, binary_closing
from skimage.color import rgb2gray, rgb2hsv
from skimage.filters import median, gaussian
from skimage.measure import find_contours
from skimage.exposure import match_histograms
import cv2
import PIL.Image
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from our_utils import deskew


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

def my_interp(contour: np.array, L_max: int):
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

def get_fourier_descriptors(contours: List[np.array]):
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
    L_max = max([len(c) for c in contours])
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

def get_opposite_points(points: np.ndarray):
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

    # check if the points are the right hand or the left hand points
    right_or_left = -1 if all(points[:, :, 1].ravel() < 700) else 1
    
    # add or subtract 90 degrees depending on whether we are working
    # with the right-hand or left-hand points
    new_angle = angle + np.pi/2 * right_or_left

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
    
    return [pl1, pl2, pl3, pl4, table, fiches]

def find_players(h_imgs: List[np.ndarray], s_imgs: List[np.ndarray], verbose=False):
    """
    Given a list of images of cards, find those players who are not playing
    """

    # Find red cards
    flattened = [img.ravel() for img in h_imgs]
    intensity = np.array([im[im > 0.8].sum() / np.prod(im.shape[:2]) for im in flattened])
    red_non_players = np.argwhere(intensity < 0.2).ravel()
    if verbose:
        print('Red intensity:', intensity)
        print('Red nonplayers:', red_non_players)

    # Find blue cards
    flattened = [img.ravel() for img in s_imgs]
    intensity = np.array([(im > 0.8).sum() / np.prod(im.shape[:2]) for im in flattened])
    blue_non_players = np.argwhere(intensity < 0.04).ravel()
    if verbose:
        print('Blue intensity:', intensity)
        print('Blue nonplayers:', blue_non_players)

    return np.sort(np.intersect1d(red_non_players, blue_non_players))

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

def get_card_outline(h_img: np.ndarray, verbose: bool = False):
    """
    Get the outline of two cards lying on top of one another 
    """

    # init loop
    limit = 0.06
    try_next_limit = True

    while try_next_limit:

        # binarize cards and perform a little morphology to get a clearer outline
        bin_cards = np.where(h_img < limit, 0, 1)   
        
        # init loop
        fewer_contours = []
        i = 0

        # morphology
        hey = binary_erosion(bin_cards, disk(2))

        while not len(fewer_contours):

            i += 1

            if verbose:
                print('erosion:', i, 'bin limit:', limit)

            # morphology
            hey = binary_erosion(hey, disk(2))

            # get all contours
            fewer_contours = get_card_contours(hey)

            # # get the approximation points of the contours
            # approxes = []
            # for c in fewer_contours:
            #     approxes += [get_four_point_approx(c)]

            # # check the approximation points
            # fewer_contours = [c for i, c in enumerate(fewer_contours)
            #                   if check_four_point_approx(approxes[i])]

            # stop if we've tried too many times
            if i == 12:
                break

        if len(fewer_contours):
            try_next_limit = False
        
        limit += 0.01

        if limit > 0.3:
            raise Exception

    card_outline = fewer_contours[get_closest_contour(hey, fewer_contours)]

    return card_outline, hey

def get_four_point_approx(card_outline: np.ndarray):
    """
    Given a card outline, get the four corner points of the outline
    """
    multiple = 0.003
    approx = []
    while len(approx) != 4:
        if multiple > 0.1:
            raise Exception
        approx = get_approx(card_outline, multiple)
        multiple += 0.0025
    return approx

# def check_four_point_approx(approx: np.ndarray):
#     """
#     Check that the four point approximation is approximately the correct shape
#     """
#     right_points, left_points = separate_left_and_right_points(approx)

#     # print('right points:', right_points)
#     # print('left points:', left_points)

#     for a, b in zip(right_points, left_points):
#         if not abs(np.linalg.norm(a.ravel() - b.ravel()) - 800) < 100:
#             print('falsity!')
#             print(a.ravel())
#             print(b.ravel())
#             print(np.linalg.norm(a.ravel() - b.ravel()))
#             return False
#     return True


def get_individual_outlines(card_outline: np.ndarray, approx: np.ndarray):
    """
    Given an outline of overlapping cards and it's four corners, extract
    the individual cards
    """

    # separate the left and the right points
    right_points, left_points = separate_left_and_right_points(approx)

    card_outlines, plot_outlines = [], []
    for points in [right_points, left_points]:
        outline, plot_outline = get_opposite_points(points)
        card_outlines += [outline]
        plot_outlines += [plot_outline]
    
    return card_outlines, plot_outlines

def segment_cards(imgs: np.ndarray, h_imgs: np.ndarray, make_plot: bool = True, verbose: bool = False):


    # cut the cards on the left-hand side
    if make_plot:
        _, ax = plt.subplots(len(imgs), 3, figsize=(15, 12))

    card_outlines, plot_outlines = [], []
    for j in range(len(imgs)):
    
        # get card outline
        card_outline, hey = get_card_outline(h_imgs[j], verbose=verbose)

        if make_plot:
            ax[j, 0].imshow(hey, cmap='gray')
            ax[j, 1].imshow(hey, cmap='gray')
        
        # get approx
        approx = get_four_point_approx(card_outline)

        # plot contour and polygon
        ax[j, 1].imshow(hey, cmap='gray')
        ax[j, 1].plot(card_outline[:, :, 1].ravel(), card_outline[:, :, 0].ravel(), 'r-', lw=5)
        ax[j, 1].plot(approx[:, :, 1], approx[:, :, 0], 'c*', ms=30)

        # separate the left and the right points
        card_outlines_, plot_outlines = get_individual_outlines(card_outline, approx)
        card_outlines += card_outlines_

        # plot
        if make_plot:
            ax[j, 2].imshow(hey, cmap='gray')
            ax[j, 2].plot(plot_outlines[-1][:, :, 1], plot_outlines[-1][:, :, 0], '*-c', ms=30)
            ax[j, 2].plot(plot_outlines[-2][:, :, 1], plot_outlines[-2][:, :, 0], '*-c', ms=30)
            ax[j, 0].set_ylabel(f'Cards {j}')
    
    if make_plot:
        # set titles
        _ = ax[0, 0].set_title('Use morphology to\nharden the card outline')
        _ = ax[0, 2].set_title('Use angle between\npoints to get rectangles')
        _ = ax[0, 1].set_title('Get contour and 4-point\npolynomial approximation')

    return card_outlines

def get_boxes_and_rectangles(card_outlines: List[np.ndarray]):
    """
    Get a list of boxes and rectangles from a list of card outlines
    """
    boxes, rects = [], []
    new_contours = [np.roll(contour, 1, axis=2) for contour in card_outlines]
    for contour in new_contours:
        rects += [cv2.minAreaRect(contour)]
        box = cv2.boxPoints(rects[-1])
        boxes += [np.int0(box)]
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
    while x_ < 30:
        tally += 3
        x_ = np.argmax(bin_im[y:120, x + tally:120].sum(axis=0))
    x_ += tally

    # same pattern
    tally = 0
    while y_ < 50:
        tally += 3
        y_ = np.argmax(bin_im[y + tally:120, x:x + x_].sum(axis=1))
    y_ += tally

    ys = np.where(bin_im[y + y_:200, x:x + x_] == 0)[0][0]
    ys_ = np.argmax(bin_im[y + y_ + ys:200, x:x + x_].sum(axis=1))

    if verbose:
        print(f"""
        {x}, {y} --- {x + x_}
         |
         |
         |
        {x}, {y_}
        """)

    return x, x_, y, y_, ys, ys_

def get_numbers_and_suits(card_outlines: List[np.ndarray], imgs: List[np.ndarray], make_plot: bool = True,
                          verbose: bool = False):
    """
    Given a list of card outlines, semgent the numbers and suits of each outline
    """
    boxes, rects = get_boxes_and_rectangles(card_outlines)

    if make_plot:
        _, axes = plt.subplots(5, 8, figsize=(15, 15))

    nums, types, num_signs = [], [], []
    for i, (rect, box) in enumerate(zip(rects, boxes)):

        warped = warp_card(rect, box, imgs[i // 2])
        
        if make_plot:
            # plot the straightened rectangles
            axes[0, i].imshow(warped)
            axes[0, i].set_title(f'{warped.shape[1]} X {warped.shape[0]}')
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])
            axes[0, 0].set_ylabel("WARPED CARDS\n", fontsize=12)
        
        warped = straighten_card(warped)
        
        # show the straightened rectangles
        if make_plot:
            axes[1, i].imshow(warped)
            for j in range(1, 3):
                axes[j, i].set_title(f'{warped.shape[1]} X {warped.shape[0]}')
                axes[j, i].set_xticks([])
                axes[j, i].set_yticks([])
            axes[1, 0].set_ylabel("WARPED AND\nFLIPPED CARDS\n", fontsize=12)
        
        x, x_, y, y_, ys, ys_ = segment_number_and_suit(warped, verbose=verbose)

        edge = min(x, y, 5)
        number = warped[y - edge:y + y_ + edge, x - edge:x + x_ + edge, :]
        suit = warped[y + y_ + ys - edge:y + y_ + ys + ys_ + edge, x - edge:x + x_ + edge, :]

        num_signs += [warped]
        nums += [number]
        types += [suit]
        
        if make_plot:

            # plot the crop lines
            axes[2, i].imshow(warped)
            axes[2, i].axvline(x=x, c='c')
            axes[2, i].axvline(x=x + x_, c='c')
            axes[2, i].axhline(y=y, c='m')
            axes[2, i].axhline(y=y + y_, c='m')
            axes[2, i].axhline(y=y + y_ + ys, c='m')
            axes[2, i].axhline(y=y + y_ + ys + ys_, c='m')
            axes[2, 0].set_ylabel("CROP LINES FOR\nNUMBER AND SUIT\n", fontsize=12)

            # crop the numbers and show them
            axes[3, i].imshow(nums[-1])
            axes[3, i].set_xticks([])
            axes[3, i].set_yticks([])
            axes[3, i].set_title(f'{nums[-1].shape[1]} X {nums[-1].shape[0]}')
            axes[3, 0].set_ylabel("CROPPED NUMBER\n", fontsize=12)

            # crop the suit and show them
            axes[4, i].imshow(types[-1])
            axes[4, i].set_xticks([])
            axes[4, i].set_yticks([])
            axes[4, i].set_title(f'{types[-1].shape[1]} X {types[-1].shape[0]}')
            axes[4, 0].set_ylabel("CROPPED SUIT\n", fontsize=12)
        
    return nums, types, num_signs


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
    im = np.array(train_images[list(train_images.keys())[0]])

    # match histograms with reference
    if not (img == im).all():
        print(img.shape, im.shape)
        img = match_histograms(img, im).round().astype(np.uint8)
    
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

def get_players_and_imgs(hsv_imgs, all_imgs, make_plot: bool = True, verbose: bool = False):

    hh_imgs = [im[:, :, 0] for im in hsv_imgs[:4]]
    ss_imgs = [im[:, :, 1] for im in hsv_imgs[:4]]

    players = find_players(hh_imgs, ss_imgs, verbose=verbose)

    if make_plot:
        _, ax = plt.subplots(2, len(all_imgs), figsize=(15, 4))
        for i, (ax_, im_) in enumerate(zip(ax[0], all_imgs)):
            ax_.imshow(im_)
            ax_.set_yticks([])
            ax_.set_xticks([])
            ax_.set_title(f'Cards {i + 1}')
        for i, (ax_, im_) in enumerate(zip(ax[1], ss_imgs)):
            ax_.imshow(im_)
            ax_.set_yticks([])
            ax_.set_xticks([])
            ax_.set_title(f'Cards {i + 1}')

    return (players,
            np.array(all_imgs, dtype='object')[players].tolist(),
            np.array(ss_imgs, dtype='object')[players].tolist())

    



    

    