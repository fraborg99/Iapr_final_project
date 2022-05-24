import os

import PIL.Image
import skimage
import cv2
import numpy as np
import skimage.exposure as e


# function to select from an array of points delimiting a shape, those corresponding to the corners
def corners(lis, im):
    height = im.shape[0]
    width = im.shape[1]
    n = len(lis)
    corners = []
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

    # find the contours and select that one which has points that are both in the lowoer left and upper right corners, it
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
#    Sorts 4 points in clockwise direction to enable the rest of the code to stretch the correct dimensions"
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
    # open the image
#     im = skimage.io.imread(file)
#     color = cv2.imread(file, cv2.IMREAD_COLOR)
    if calibration:
        file2 = os.path.join("data/train/train_21.jpg")
        im2 = skimage.io.imread(file2)
        newimage = e.match_histograms(im2, im, channel_axis=-1)
        im = newimage
    try:
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
            out = deskew(im, calibration = True)
        except:
            out = im[200:3800,1200:4800,:]
    return out
