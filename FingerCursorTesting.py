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


"""IDEA
Have model predict location and place dot
have dots pop up randomly as targets

have a button that signifies that you are aiming at target

feed this dot location into model as a form of supervised learning
Kind of like giving it real world data
"""

WHITE =     (255, 255, 255)
BLUE =      (  0,   0, 255)
GREEN =     (  0, 255,   0)
RED =       (255,   0,   0)
TEXTCOLOR = (  0,   0,  0)
(width, height) = (1680, 1022)

name="small_v6_noShuffle-05"
modelFiles = "models"
model = keras.models.load_model(os.path.abspath(f"{modelFiles}/{name}.keras"))


class Screen:
    def __init__(self, draw=True):
        
        self.webcam = 1
        self.smoothing = 5
        self.screen = pygame.display.set_mode((width, height))
        self.running = True
        self.data = {"0":[], "dot":[]}
        
        self.pPosX = 0
        self.pPosY = 0
        self.cPosX = 0
        self.cPosY = 0
        
        self.draw = draw
    
    def drawCircle(self, pos):
        self.cPosX = self.pPosX + (pos[0] - self.pPosX)/self.smoothing
        self.cPosY = self.pPosY + (pos[1] - self.pPosY)/self.smoothing
        pyautogui.moveTo(self.cPosX, self.cPosY, duration=0)
        # print(pos, (self.cPosX, self.cPosY),'\n')
        
        if self.draw:
            self.screen.fill(WHITE)
            pygame.draw.circle(self.screen, BLUE, (self.cPosX, self.cPosY), 20)
            pygame.display.update()
        
        self.pPosX = self.cPosX
        self.pPosY = self.cPosY

    def predictLocation(self, pos, handPos):
        
        self.data["0"] = [handPos[0]]
        self.data["dot"] = [pos]
        
        df = pd.DataFrame.from_dict(self.data["0"])

        x_result_df, y_result_df = self.split_coordinates(df)
        
        combined_df = pd.concat([x_result_df, y_result_df], axis=1)
        # print("COMBINED",combined_df)
        feature_tensor = combined_df.to_numpy()
        
        val = model.predict(feature_tensor, verbose=False)
        self.drawCircle(val[0])
        
        
    def split_coordinates(self, df):
        x_data = {}
        y_data = {} 
        for col in df.columns:
            x_data[col] = df[col].apply(lambda x: x[2])  # Extract x-coordinate
            y_data[col] = df[col].apply(lambda x: x[3])  # Extract y-coordinate
        x_df = pd.DataFrame(x_data)
        y_df = pd.DataFrame(y_data)
        # print("split x_data:",x_df)

        return x_df, y_df
        



def main():
    Display = Screen(draw=False)
    pos = (random.randint(0,width),random.randint(0,height))
    Display.drawCircle(pos)
    
    cap = cv2.VideoCapture(Display.webcam)
    prevTime = 0
    currentTime = 0
    detector = HandDetector()
    
    if Display.draw:
        pygame.init()
        
        pygame.display.set_caption("TUFF")
    
    while Display.running:
        stime = time.time()
        success, img = cap.read()
        detector.img = img
        detector.detect_hands(draw=False)
        detector.find_node_positions_of_hand(draw=False)
        ptime = time.time()
        print("Detector took:", round(float(ptime-stime),2),"seconds")
        
        # currentTime = time.time()
        # fps = 1/(currentTime-prevTime)
        # prevTime = currentTime
        # cv2.putText(detector.img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        # cv2.imshow("Image", detector.img)
        
        stime = time.time()
        handPos = detector.get_positions(-1)    
        Display.predictLocation(pos, handPos)
        ptime = time.time()
        print("Predictor took:", round(float(ptime-stime),2),"seconds")
        # cv2.waitKey(1) 
        
        if Display.draw:
            ev = pygame.event.get()
            for event in ev:
                if event.type == pygame.KEYDOWN:
                    pass
                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_a:
                        handPos = detector.get_positions(-1)    
                        Display.predictLocation(pos, handPos)
                    if event.key == pygame.K_s:
                        pos = (random.randint(0,width),random.randint(0,height))
                        Display.drawCircle(pos)
                if event.type == pygame.QUIT:
                    Display.running = False


    

if __name__ == '__main__':
    main()