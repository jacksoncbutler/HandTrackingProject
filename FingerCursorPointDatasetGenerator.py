import pygame
import random
from HandTrackingModule import HandDetector
import cv2
import time
import json 

WHITE =     (255, 255, 255)
BLUE =      (  0,   0, 255)
GREEN =     (  0, 255,   0)
RED =       (255,   0,   0)
TEXTCOLOR = (  0,   0,  0)
(width, height) = (1680, 1022)

running = True

# print(locations)

dataFile = "/Users/jackcameback/Classes/Fall2023/MachineLearning/HandTrackingProject/data/otherData.json"
data = {"0":[], "1":[], "dot":[]}


def main():
    global running, screen
    
    
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("TUFF")
    pos = (random.randint(0,width),random.randint(0,height))
    drawCircle(pos)
    
    cap = cv2.VideoCapture(1)
    prevTime = 0
    currentTime = 0
    detector = HandDetector()
    
    record = False

    locations = []
    delay = 0.25
    ptime = time.time()

    for y in range(0, height, 40):
        for x in range(0, width, 40):
            locations.append((x,y))
    
    while running:
        success, img = cap.read()
        detector.img = img
        detector.detect_hands(draw=True)
        detector.find_node_positions_of_hand(draw=False)

        if record:
            ctime = time.time()
            if (ctime - ptime) > delay:
                print("recording")
                handPos = detector.get_positions(-1)
                recordData(pos, handPos)
                ptime = ctime
        currentTime = time.time()
        fps = 1/(currentTime-prevTime)
        prevTime = currentTime
        
        cv2.putText(detector.img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        
        
        
        cv2.imshow("Image", detector.img)
        cv2.waitKey(1) 
        
        ev = pygame.event.get()

        for event in ev:
            if event.type == pygame.KEYDOWN:
                pass
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_a:
                    handPos = detector.get_positions(-1)
                    recordData(pos, handPos)
                    pos = (random.randint(0,width),random.randint(0,height))
                    drawCircle(pos)
                
                if event.key == pygame.K_n:
                    print("next")
                    if len(locations) > 0:
                        pos = locations.pop()
                        drawCircle(pos)
                        
                if event.key == pygame.K_s:
                    print('S key')
                    if record:
                        print("STOP RECORDING")
                        record = False
                    else:
                        record = True
                
                if event.key == pygame.K_ESCAPE:
                    print("SAVED")
                    with open(dataFile, "w") as file:
                        json.dump(data, file)
            if event.type == pygame.QUIT:
                running = False

def drawCircle(pos):
    screen.fill(WHITE)
    pygame.draw.circle(screen, BLUE, pos, 20)
    pygame.display.update()

def recordData(pos, handPos):
    
    data["0"].append(handPos[0])
    data["1"].append(handPos[1])
    data["dot"].append(pos)

if __name__ == '__main__':
    main()