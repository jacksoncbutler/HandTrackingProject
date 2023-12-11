import pygame
import random
from HandTrackingModule import HandDetector
import keras
import cv2
import time
import json 
import pandas as pd
"""IDEA
Have model predict location andplace dot
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

running = True
model = keras.models.load_model(f'/Users/jackcameback/Classes/Fall2023/MachineLearning/HandTrackingProject/models/{name}.keras')


data = {"0":[], "dot":[]}

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
        handPos = detector.get_positions(-1)    
        predictLocation(pos, handPos)
        cv2.waitKey(1) 
        
        ev = pygame.event.get()

        
        for event in ev:
            if event.type == pygame.KEYDOWN:
                pass
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_a:
                    handPos = detector.get_positions(-1)    
                    predictLocation(pos, handPos)
                if event.key == pygame.K_s:
                    pos = (random.randint(0,width),random.randint(0,height))
                    drawCircle(pos)
            if event.type == pygame.QUIT:
                running = False

def drawCircle(pos):
    screen.fill(WHITE)
    pygame.draw.circle(screen, BLUE, pos, 20)
    pygame.display.update()

def predictLocation(pos, handPos):
    
    data["0"] = [handPos[0]]
    data["dot"] = [pos]
    print("data",len(data["0"]), len(data["dot"]))
    df = pd.DataFrame.from_dict(data)
    tempDict = {}
    
    for i in range(len(df['0'])):

        for layer in df["0"][i]:
            try:
                tempDict[layer[1]].append((layer[2],layer[3]))
            except:
                tempDict[layer[1]] = [(layer[2],layer[3])]
        try:
            tempDict["target"].append(df["dot"][i])
        except:
            tempDict["target"] = [df["dot"][i]]
        
    dataFrame = pd.DataFrame.from_dict(tempDict)
    x_result_df, y_result_df = split_coordinates(dataFrame)
    
    print("X DataFrame:")
    print(x_result_df)
    print("\nY DataFrame:")
    print(y_result_df)
    x_target = pd.Series(x_result_df["target"])
    x_result_df.drop(["target"], axis=1, inplace=True)
    print(x_target)
    y_target = pd.Series(y_result_df["target"])
    y_result_df.drop(["target"], axis=1, inplace=True)
    print(y_target)
    
    combined_df = pd.concat([x_result_df, y_result_df], axis=1)
    print(combined_df.shape)
    feature_tensor = combined_df.to_numpy()
    
    val = model.predict(feature_tensor)
    print(val)
    # print(x,y)
    drawCircle(val[0])
    
    
def split_coordinates(df):
    x_data = {}
    y_data = {}
    for col in df.columns:
        x_data[col] = df[col].apply(lambda x: x[0])  # Extract x-coordinate
        y_data[col] = df[col].apply(lambda x: x[1])  # Extract y-coordinate
    x_df = pd.DataFrame(x_data)
    y_df = pd.DataFrame(y_data)

    return x_df, y_df
    
    

if __name__ == '__main__':
    main()