import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(1)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

prevTime = 0
currentTime = 0

while True:
    success, img = cap.read()
    # print(type(img))
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            for id, landmark in enumerate(hand.landmark):
                # print(id,landmark)
                height, width, channels = img.shape
                centerX, centerY = int(landmark.x*width), int(landmark.y*height)
                print(id, centerX, centerY)
                
                if id == 8:
                    cv2.circle(img, (centerX, centerY), 15, (255,0,255), cv2.FILLED)
            
            mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)
            
    currentTime = time.time()
    fps = 1/(currentTime-prevTime)
    prevTime = currentTime
    
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    
    
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)