import pygame
import random
from HandTrackingModule import HandDetector
import keras
import cv2
import time
import json
import pandas as pd
import os
import tensorflow as tf
import timeit
import cProfile
from pynput.mouse import Button,Controller
import numpy as np
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
        self.threshold = 0.8
        
        self.screen = pygame.display.set_mode((width, height))
        self.running = True
        self.data = {"0":[], "1":[], "dot":[]}
        self.targetMap = {0:'idle', 1:'lclick', 2:'rclick'}

        
        self.pPosX = 0
        self.pPosY = 0
        self.cPosX = 0
        self.cPosY = 0
        
        self.draw = draw
        
        
        self.cursorModelName="small_v7_noVal_lite"
        self.cursorModelType = "tflite"
        self.cursorModelDir = "tflite"

        if self.cursorModelType == "keras":
            self.model = keras.models.load_model(os.path.abspath(f"models/{self.cursorModelDir}/{self.cursorModelName}.keras"))
            self.model.summary()
        if self.cursorModelType == "concrete":
            self.loaded_module = tf.saved_model.load(os.path.abspath(f"models/{self.cursorModelDir}/{self.cursorModelName}"))
            self.concrete_function = self.loaded_module.func
        if self.cursorModelType == "tflite":
            self.interpreter = tf.lite.Interpreter(os.path.abspath(f"models/{self.cursorModelDir}/{self.cursorModelName}.tflite"))
            # tflite_model = interpreter.get_signature_runner()
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
        self.gestureModelName = "small_v2_gesture"
        # self.gestureModelType = "keras"
        self.gestureModelDir = "gesture"
        self.gestureModel = keras.models.load_model(os.path.abspath(f"models/{self.gestureModelDir}/{self.gestureModelName}.keras"))

            
        
    
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

    def predictLocation(self, pos, handPos):        
        df = pd.DataFrame.from_dict(self.data["0"])
        # print(df.head())

        x_result_df, y_result_df = self.split_coordinates(df)
        
        combined_df = pd.concat([x_result_df, y_result_df], axis=1)
        # print("COMBINED",combined_df)
        feature_tensor = combined_df.to_numpy()

        if self.cursorModelType == "keras":
            pos = self.model.predict(tf.constant(feature_tensor), verbose=False)
            # print("shape:", tf.constant(feature_tensor).shape()
        if self.cursorModelType == "concrete":
            output = self.concrete_function(tf.constant(feature_tensor))
            x = output[0][0][0]
            y = output[0][0][1]
        # print(x,y)
        if self.cursorModelType == "tflite":
            self.interpreter.set_tensor(self.input_details[0]['index'], feature_tensor)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            # print(output)
            x = output[0][0]
            y = output[0][1]


        # cProfile.runctx("self.drawCircle((x,y))", globals(), locals())
        self.drawCircle((x,y))
        
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
    
    def checkGesture(self):
        df = pd.DataFrame.from_dict(self.data["1"])
        x_result_df, y_result_df = self.split_coordinates(df)
        
        combined_df = pd.concat([x_result_df, y_result_df], axis=1)
        # print("COMBINED",combined_df)
        feature_tensor = combined_df.to_numpy()
        result = self.gestureModel.predict(tf.constant(feature_tensor), verbose=False)
        print("RESULTS TYPE:",result[0])
        # valid_idx = np.where(result >= self.threshold)[0]
        # print("num valid incidiecs:", len(valid_idx))
        gestureIndex = np.argmax(result[0])
        if result[0][gestureIndex] >= self.threshold:
            gesture = self.targetMap[gestureIndex]
            print("Gesture Prediction:",self.targetMap[gestureIndex])
            if gesture == 'lclick':
                self.mouse.click(Button.left)
            elif gesture == 'rclick':
                self.mouse.click(Button.right)
        else:
            print("Gesture Prediction: idle")
        # gesture = valid_idx[np.argmax(result[valid_idx])]
        # print(gesture)
        # gesture = self.targetMap[np.argmax(result)]
        # print("Gesture Prediction:",gesture)
       
        

        



def main():
    Display = Screen(draw=False)
    pos = (random.randint(0,width),random.randint(0,height))
    # Display.drawCircle(pos)
    
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
        # print("Predictor took:", round(float(time.time()-stime),2),"seconds")
        detector.img = img
        # print("Predictor took:", round(float(time.time()-stime),2),"seconds")
        detector.detect_hands(0, draw=False)
        # cProfile.runctx("detector.detect_hands(draw=False)", globals(), locals())

        
        detector.find_node_positions_of_hand(draw=False)
        # cProfile.runctx("detector.find_node_positions_of_hand(draw=False)", globals(), locals())

        
        handPos = detector.get_positions(-1)    
        # cProfile.runctx("detector.get_positions(-1)", globals(), locals())
        Display.data["0"] = [handPos[0]]
        Display.data["1"] = [handPos[1]]
        Display.data["dot"] = [pos]
        

        # cProfile.runctx("Display.predictLocation(pos, handPos)", globals(), locals())
        Display.checkGesture()
        Display.predictLocation(pos, handPos)   
        
        # print("GetTime:", round(getTime-stimei,2))

        
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