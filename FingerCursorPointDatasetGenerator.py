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

dataFile = "/Users/jackcameback/Classes/Fall2023/MachineLearning/HandTrackingProject/data/FingerCursorData.json"
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
    
    while running:
        success, img = cap.read()
        detector.img = img
        detector.detect_hands(draw=False)
        detector.find_node_positions_of_hand(draw=False)
        
        
        # currentTime = time.time()
        # fps = 1/(currentTime-prevTime)
        # prevTime = currentTime
        
        # cv2.putText(detector.img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        
        
        
        # cv2.imshow("Image", detector.img)
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
                if event.key == pygame.K_ESCAPE:
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