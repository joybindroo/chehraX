# import modules
import sys, os, cv2, dlib, time
import argparse
import numpy as np
import faceBlendCommon as fbc
import matplotlib.pyplot as plt
import re
import urllib

def create_arg_parser():
    """"Creates and returns the ArgumentParser object."""
    parser = argparse.ArgumentParser(description='Script to swap faces between two images.')
    parser.add_argument('from_image', help='Path to first image where you will extract the face.')
    parser.add_argument('to_image', help='Path to second image where you will swap the existing face')
    parser.add_argument('output_filename', help='Path to output image.')
    return parser

def find_url(string):
    """Find if a string contains an URL"""
    # findall() has been used
    # with valid conditions for urls in string
    url = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', string)
    return url

def url_to_image(url):
    """Convert URL to OpenCV Image"""
    # source: https://www.pyimagesearch.com/2015/03/02/convert-url-to-image-with-python-and-opencv/
    # download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
    resp = urllib.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	# return the image
    return image

def run_face_swap(from_image, to_image, output_filename):
    """Switch faces between two input images using dlib and OpenCV."""
    # Credits to https://github.com/spmallick/
    try:
        if len(find_url(from_image)) > 0:
            img1 = url_to_image(from_image)
        else:
            img1 = cv2.imread(from_image)
        if len(find_url(to_image)) > 0:
            img2 = url_to_image(to_image)
        else:
            img2 = cv2.imread(to_image)
        img1Warped = np.copy(img2)
        # Initialize the dlib facial landmark detector
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        # Read array of corresponding points
        points1 = fbc.getLandmarks(detector, predictor, img1)
        points2 = fbc.getLandmarks(detector, predictor, img2)
        # Find convex hull
        hullIndex = cv2.convexHull(np.array(points2).astype(np.int32), returnPoints=False) # add .astype(np.int32) to fix TypeError: data type = 9 not supported
        # Create convex hull lists
        hull1 = []
        hull2 = []
        for i in range(0, len(hullIndex)):
            hull1.append(points1[hullIndex[i][0]])
            hull2.append(points2[hullIndex[i][0]])
        # Calculate Mask for Seamless cloning
        hull8U = []
        for i in range(0, len(hull2)):
            hull8U.append((hull2[i][0], hull2[i][1]))
        mask = np.zeros(img2.shape, dtype=img2.dtype)
        cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))
        # Find Centroid
        m = cv2.moments(mask[:,:,1])
        center = (int(m['m10']/m['m00']), int(m['m01']/m['m00']))
        # Find Delaunay traingulation for convex hull points
        sizeImg2 = img2.shape
        rect = (0, 0, sizeImg2[1], sizeImg2[0])
        dt = fbc.calculateDelaunayTriangles(rect, hull2)
        # If no Delaunay Triangles were found, quit
        if len(dt) == 0:
            quit()
        # Apply affine transformation to Delaunay triangles
        for i in range(0, len(tris1)):
            fbc.warpTriangle(img1, img1Warped, tris1[i], tris2[i])
        # Seamless Cloning using OpenCV
        output = cv2.seamlessClone(np.uint8(img1Warped), img2, mask, center, cv2.NORMAL_CLONE)
        # Write output image
        cv2.imwrite(output_filename, output)
    except KeyError:
        raise KeyError('there was an error')

if __name__ == "__main__":
    arg_parser = create_arg_parser()
    parsed_args = arg_parser.parse_args(sys.argv[1:])
    run_face_swap(parsed_args.from_image, parsed_args.to_image, parsed_args.output_filename)
