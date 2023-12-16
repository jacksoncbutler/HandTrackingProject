import pygame
import random
from HandTrackingModule import HandDetector
import keras
import cv2
import time
import json
import pandas as pd
import os
import pyautogui
import tensorflow as tf
import timeit
import cProfile
from pynput.mouse import Controller
# import pyautogui


"""IDEA
Have model predict location and place dot
have dots pop up randomly as targets

have a button that signifies that you are aiming at target

feed this dot location into model as a form of supervised learning
Kind of like giving it real world data

GEESTURES
Train a geauster model

SCROLL:
Index and middle out
train a scroll model with sequence data


"""

WHITE =     (255, 255, 255)
BLUE =      (  0,   0, 255)
GREEN =     (  0, 255,   0)
RED =       (255,   0,   0)
TEXTCOLOR = (  0,   0,  0)
(width, height) = (1680, 1022)



    

class Screen:
    def __init__(self, draw=True):
        self.mouse = Controller()
        
        self.webcam = 0
        self.smoothing = 3
        self.screen = pygame.display.set_mode((width, height))
        self.running = True
        self.data = {"0":[], "1":[], "dot":[]}
        self.delay = 0.05
        self.record = False
        self.gestures = ["idle", "lclick", "rclick"]
        self.modelDir = "gesture"
        self.name = "GestureData"
        self.dataFile = os.path.abspath(f"data/{self.modelDir}/{self.name}.json")

        
        
        self.draw = draw
        
    
    def drawCircle(self, pos):
        self.cPosX = self.pPosX + (pos[0] - self.pPosX)/self.smoothing
        self.cPosY = self.pPosY + (pos[1] - self.pPosY)/self.smoothing
        # pyautogui.PAUSE=0
        # pyautogui.MINIMUM_SLEEP = 0
        # pyautogui.moveTo(self.cPosX, self.cPosY, duration=0.02, _pause=False) 
        mousePos = self.mouse.position
        self.mouse.move(self.cPosX-mousePos[0],self.cPosY-mousePos[1])
        self.pPosX = self.cPosX
        self.pPosY = self.cPosY

    def recordData(self, pos, handPos):

        self.data["0"].append(handPos[0])
        self.data["1"].append(handPos[1])
        self.data["dot"].append(pos)
    



def main():
    Display = Screen(draw=True)
    # pos = (random.randint(0,width),random.randint(0,height))
    pos = ""
    # Display.drawCircle(pos)
    
    cap = cv2.VideoCapture(Display.webcam)
    detector = HandDetector()
    ptime = time.time()
    
    if Display.draw:
        pygame.init()
        
        pygame.display.set_caption("TUFF")
    
    while Display.running:
        stime = time.time()
        success, img = cap.read()
        detector.img = img
        detector.detect_hands(0, draw=Display.draw)
        detector.find_node_positions_of_hand(draw=Display.draw)

        
        cv2.imshow("Image", detector.img)
        # cv2.waitKey(1) 
        
        if Display.record:
            ctime = time.time()
            if (ctime - ptime) > Display.delay:
                print("recording")
                handPos = detector.get_positions(-1)
                Display.recordData(pos, handPos)
                ptime = ctime

        
        if Display.draw:
            ev = pygame.event.get()
            for event in ev:
                if event.type == pygame.KEYDOWN:
                    pass
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_n:
                        print("next")
                        if len(Display.gestures) > 0:
                            pos = Display.gestures.pop(0)

                    if event.key == pygame.K_s:
                        print('S key')
                        if Display.record:
                            print("STOP RECORDING")
                            Display.record = False
                        else:
                            Display.record = True
                            
                    if event.key == pygame.K_ESCAPE:
                        print("SAVED")
                        print(Display.data)
                        with open(Display.dataFile, "w") as file:
                            json.dump(Display.data, file)
                if event.type == pygame.QUIT:
                    Display.running = False


    

if __name__ == '__main__':
    main()