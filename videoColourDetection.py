import sys
import cv2 as cv
import numpy as np
import os

ballColours = {
    "red" : (5, 150, 255),
    "yellow" : (30, 125, 255),
    "green" : (65, 255, 153),
    "brown" : (15, 255, 165),
    "blue" : (110, 255, 200),
    "pink" : (165, 67, 255),
    "black" : (0, 10, 10),
    "white" : (30, 15, 255)
}

def findColour(center, radius, img):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv.circle(mask, center, int(radius*0.5), 255, -1)
    colour = cv.mean(img, mask=mask)

    return colour

def findClosestColour(colour, ballColours):
    bgrPixel = np.uint8([[colour]])
    hsvPixel = cv.cvtColor(bgrPixel, cv.COLOR_BGR2HSV)
    hsv = hsvPixel[0][0]

    bestDistance = 99999
    bestColour = None

    for name, ballColour in ballColours.items():
        dh = min(abs(int(hsv[0]) - ballColour[0]), 180 - abs(int(hsv[0]) - ballColour[0]))
        ds = abs(int(hsv[1]) - ballColour[1])
        dv = abs(int(hsv[2]) - ballColour[2])
        colourDistance = dh + ds + dv
        if colourDistance < bestDistance:
            bestDistance = colourDistance
            bestColour = name

    return bestColour

def main(argv):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    Filename = os.path.join(base_dir, "snookerVid1.mp4")
    source = cv.VideoCapture(Filename) 
    running  = True

    while running == True:
        ret, src = source.read()  
        if not ret:
            running = False
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        gray = cv.medianBlur(gray, 5)
        
        circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 20,
                                param1=180, param2=18,
                                minRadius=12, maxRadius=25)
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                radius = i[2]

                colour = findColour(center, radius, src)
                ballColour = findClosestColour(colour, ballColours)
                
                # circle outline
                cv.circle(src, center, radius, colour, -1)
                cv.putText(src, ballColour, (i[0]-25, i[1]-10), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1, cv.LINE_AA)
        cv.imshow("detected circles", src)
        cv.waitKey(1)
    
    return 0
if __name__ == "__main__":
    main(sys.argv[1:])