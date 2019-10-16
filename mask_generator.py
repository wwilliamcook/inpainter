"""Generates random masks for images.
"""

import cv2 as cv
import numpy as np
import time


def newPoint(points, w, h, mean=0, stddev=1):
    cx, cy = np.mean(points, axis=0)
    r = np.random.normal(loc=mean, scale=stddev)
    theta = np.random.random() * 2*np.pi
    x, y = cx + r*np.cos(theta), cy + r*np.sin(theta)
    x = np.clip(int(x), 0, w)
    y = np.clip(int(y), 0, h)
    return np.concatenate([points, [[x, y]]], axis=0)

def drawPoints(points, width, height, radius=3):
    canvas = np.ones([height, width, 3])
    for (x, y) in points:
        cv.circle(canvas, (x, y), radius, (0, 0, 0), -1)
    # Indicate center
    cx, cy = np.int32(np.round(np.mean(points, axis=0)))
    cv.circle(canvas, (round(cx), round(cy)), radius, (0, 0, 255), -1)
    return canvas

def imshow(img, delay=0):
    cv.imshow('image', img)
    k = cv.waitKey(delay)
    if delay == 0:
        cv.destroyWindow('image')
    return k

def orderConvexPolyPoints(points):
    center = np.mean(points, axis=0)
    pts = points - np.expand_dims(center, 0)
    thetas = np.mod(np.arctan2(pts[:,1], pts[:,0]), 2*np.pi)
    indices = np.arange(len(points))
    indices = np.array(list(zip(*sorted(zip(thetas, indices))))[1])
    return np.array(points)[indices]

def randomMask(w, h):
    """Randomly generate a mask with a polygonal covered region.
    """
    # Decide how many vertices the mask will have
    count = np.random.randint(3, 15)
    # Pick the first point, biased toward the center using noise shaping
    pts = np.random.random(size=[1, 2]) - .5
    pts = 2.8*pts**3 + .3*pts + .5
    pts = np.int32(pts*[w-1, h-1])
    # Add points near the center
    mean, stddev = min(w, h) / 8, min(w, h) / 5
    for i in range(1, count):
        pts = newPoint(pts, w, h, mean, stddev)
    # Sort points along polar theta axis
    pts = orderConvexPolyPoints(pts)
    # Draw the mask
    return cv.fillPoly(np.ones([h, w, 1]), [pts], (0,))

def buildMaskGenerator(w, h):
    def maskGenerator():
        while True:
            yield randomMask(w, h)
    return maskGenerator
