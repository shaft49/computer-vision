import cv2
from imutils.video import VideoStream
import imutils
import numpy as np
import time
class InvisibleCloak:
    def __init__(self):
        pass


    def get_color_mask(self, hsv):
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        lower_mask = cv2.inRange(hsv, lower_red, upper_red)

        lower_red = np.array([170, 20, 70])
        upper_red = np.array([180, 255, 255])
        upper_mask = cv2.inRange(hsv, lower_red, upper_red)

        return lower_mask, upper_mask

    def run_web_cam(self):
        print('[INFO] Web Cam is starting')
        video_capture = VideoStream(src=0).start()
        time.sleep(3)

        background_frame = 0
        for i in range(30):
            background_frame = video_capture.read()
            # cv2.imshow('Background Frame', background_frame)
            # key = cv2.waitKey(1) & 0xFF
            # if key == ord('q'):
            #     break
        background_frame = np.flip(background_frame, axis=1)

        while True:
            frame = video_capture.read()
            frame = np.flip(frame, axis=1)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            lower_mask, upper_mask = self.get_color_mask(hsv)
            mask = lower_mask + upper_mask
            
            mask1 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
            mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3,3),np.uint8))
            mask2 = cv2.bitwise_not(mask1)
            segmented_image = cv2.bitwise_and(frame, frame, mask=mask2)
            background_segment = cv2.bitwise_and(background_frame, background_frame, mask = mask1)
            final_frame = cv2.addWeighted(segmented_image,1,background_segment,1,0)
            final_frame = imutils.resize(final_frame, width = 1600, height = 900)
            cv2.imshow('Output Frame', final_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

if __name__ == '__main__':
    cloak = InvisibleCloak()
    cloak.run_web_cam()
    print('[INFO] Enf of the file')