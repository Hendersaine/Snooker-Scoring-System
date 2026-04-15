from ultralytics import YOLO
import os
import cv2 as cv
import numpy as np
from collections import Counter

ballColours = {
    "red" : (5, 150, 255),
    "yellow" : (30, 160, 255),
    "green" : (85, 250, 222),
    "brown" : (15, 165, 165),
    "blue" : (97, 115, 204),
    "pink" : (9, 90, 236),
    "black" : (73, 102, 65),
    "white" : (44, 15, 255)
}

def findColour(center, radius, img):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv.circle(mask, center, int(radius*0.5), 255, -1)
    colour = cv.mean(img, mask=mask)
    print(center)
    return colour
def findClosestColour(colour, ballColours):
    bgrPixel = np.uint8([[colour]])
    hsvPixel = cv.cvtColor(bgrPixel, cv.COLOR_BGR2HSV)
    hsv = hsvPixel[0][0]
    print(hsv)

    bestDistance = 99999
    bestColour = None

    for name, ballColour in ballColours.items():
        dh = min(abs(int(hsv[0]) - ballColour[0]), 180 - abs(int(hsv[0]) - ballColour[0]))
        ds = abs(int(hsv[1]) - ballColour[1])
        dv = abs(int(hsv[2]) - ballColour[2])
        colourDistance = 8*dh + ds + dv
        if colourDistance < bestDistance:
            bestDistance = colourDistance
            bestColour = name
    print(bestColour)
    return bestColour

base_dir = os.path.dirname(os.path.abspath(__file__))

model = YOLO("yolo26m.yaml")
model = YOLO("yolo26m.pt")
video_path = os.path.join(base_dir, "Images", "SnookerVid1.mp4")

cap = cv.VideoCapture(video_path)
previousWhite, roundedCoordinates = [0, 0]
previousBalls = [23, 23, 23, 23, 23]
previousColours = [Counter() for x in range(6)]
stopCounter = 0
moveCounter = 0
moving, currentPlayer, whiteFound = False, False, False
positionRounding = 8
switchCounter = -1
annotation = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, device=0)  # ensure GPU
    #annotated = results[0].plot(labels=False, conf=False)
    annotated = frame.copy()
    numBalls = 0
    if not moving:     
        for i in range(5):
            previousColours[i] = previousColours[i+1]
    previousColours[5] = Counter()
    whiteFound = False
    if results[0] is not None:
        for ball in results[0].boxes:
            x, y, w, h = ball.xywh[0].tolist()
            id = model.names[int(ball.cls[0])]
            
            if id != "sports ball":
                continue
            numBalls += 1
            colour = findColour((int(x), int(y)), w/2, frame)
            ballColour = findClosestColour(colour, ballColours)
            cv.putText(annotated, ballColour, (int(x) - 10, int(y) - 20), cv.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 2)
            if ballColour == "white":
                roundedCoordinates = [
                    int(round(int(x/positionRounding))*positionRounding),
                    int(round(int(y/positionRounding))*positionRounding)
                ]
                if not moving:
                    whiteFound = True
            previousColours[5][ballColour] += 1
            x1, y1, x2, y2 = ball.xyxy[0].tolist()
            print(x1)
            annotated = cv.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (165, 0, 255))

    if whiteFound:
        previousBalls.append(numBalls)
    
    if previousWhite != roundedCoordinates:
        if moveCounter > 2:
            stopCounter = 0
            moving = True
            annotation = "Moving"
        else:
            moveCounter += 1
            moving = False
    else:
        if stopCounter < 10:
            stopCounter += 1
            moving = True
        elif stopCounter == 10:
            moveCounter = 0
            stopCounter += 1
            moving = False
            pottedBall = list((previousColours[0] - previousColours[5]).elements())
            if pottedBall:
                annotation = str(pottedBall)
            else:
                currentPlayer = not currentPlayer
                annotation = "pot miss"
                switchCounter += 1
        else:
            stopCounter += 1
            Moving = False
    previousWhite = roundedCoordinates

    #annotation = str(moving) + str(stopCounter)

    #annotation = "The number of balls potted: " + str(closeToPocket)
    annotated = cv.putText(annotated, annotation, (40, 40), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    #annotated = cv.rectangle(annotated, (950, 750), (1100, 880), (255, 0, 0), 2)

    cv.imshow("Detections", annotated)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
