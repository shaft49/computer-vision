from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils
from transform import four_point_transform
import os

class Scan:
    def __init__(self, folder):
        self.outputFolder = 'scanned-images'
        self.folder = folder
        self.ratio = None

    def loadAllImages(self):
        images = []
        for filename in os.listdir(self.folder):
            image = cv2.imread(os.path.join(self.folder, filename))
            if image is not None:
                images.append((image, filename))
        return images
    
    def edgeDetection(self, image):
        self.ratio = image.shape[0] / 500.0
        image = imutils.resize(image, height = 500)
        # convert the image to grayscale, blur it, and find edges
        # in the image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 75, 200)
        # show the original image and the edge detected image
        print("STEP 1: Edge Detection")
        cv2.imshow("Image", image)
        cv2.imshow("Edged", edged)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return edged

    def findContours(self, edged, image, orig):
        # find the contours in the edged image, keeping only the
        # largest ones, and initialize the screen contour
        cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
        # loop over the contours
        screenCnt = None
        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # if our approximated contour has four points, then we
            # can assume that we have found our screen
            if len(approx) == 4:
                screenCnt = approx
                break
        # show the contour (outline) of the piece of paper
        print("STEP 2: Find contours of paper")
        if screenCnt is not None:
            cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
        cv2.imshow("Outline", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
        # apply the four point transform to obtain a top-down
        # view of the original image
        if screenCnt is not None:
            warped = four_point_transform(orig, screenCnt.reshape(4, 2) * self.ratio)
            # convert the warped image to grayscale, then threshold it
            # to give it that 'black and white' paper effect
            warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            T = threshold_local(warped, 11, offset = 10, method = "gaussian")
            warped = (warped > T).astype("uint8") * 255
            # show the original and scanned images
            print("STEP 3: Apply perspective transform")
            cv2.imshow("Original", imutils.resize(orig, height = 650))
            cv2.imshow("Scanned", imutils.resize(warped, height = 650))
            cv2.waitKey(0)

    def startScanning(self):
        images = self.loadAllImages()
        if not os.path.exists(self.outputFolder):
            os.makedirs(self.outputFolder)
        
        for image, filename in images:
            orig = image.copy()
            edged = self.edgeDetection(image)
            self.findContours(edged, image, orig)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", required = True, help = "Path to the image folder to be scanned")
    args = parser.parse_args()
    scan  = Scan(args.folder)
    scan.startScanning()