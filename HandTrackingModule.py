import cv2
import mediapipe as mp
import time

class HandDetector:
    def __init__(self, mode=False, maxHands=2, minDetectionConfidence=0.5, mintrackConfidence=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.minDetectionConfidence = minDetectionConfidence
        self.mintrackConfidence = mintrackConfidence
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands, 
                                        min_detection_confidence=self.minDetectionConfidence, 
                                        min_tracking_confidence=self.mintrackConfidence)
        self.mpDraw = mp.solutions.drawing_utils
        
        self._img = None
        self._img_process_result = None
        self._positions = {}
        for hand in range(maxHands):
            tempList = []
            for i in range(21):
                tempList.append((hand, i, -1, -1))
            self.positions = (hand, tempList)
        
    @property
    def img(self):
        return self._img
    
    @property
    def img_process_result(self):
        return self._img_process_result
    
    @property
    def positions(self):
        return self._positions

    
    @img.setter
    def img(self, newImage):
        self._img = newImage
        self.img_process_result = self.img

    @img_process_result.setter
    def img_process_result(self, newImage):
        imgRGB = cv2.cvtColor(newImage, cv2.COLOR_BGR2RGB)
        self._img_process_result = self.hands.process(imgRGB)
    
    @positions.setter
    def positions(self, info):
        hand, values = info

        self._positions[hand] = values
        # print(self._positions)
    
    def get_positions(self, hand:int, indicies:list=(range(21))):
        returnLyst = {}
        if hand < 0:
            for handIndex in range(len(self.positions.keys())):
                currentHand = self.positions[handIndex]
                returnLyst[handIndex] = []
                for index in indicies:
                    returnLyst[handIndex].append(currentHand[index])
        else:
            # print(self.positions)
            currentHand = self.positions[hand]
            returnLyst[hand] = []
            for index in indicies:
                returnLyst[hand].append(currentHand[index])
        return returnLyst
    
    
    def detect_hands(self, handVal=-1, draw=True):
        if self.img_process_result.multi_hand_landmarks:
            if handVal==-1:
                for hand in self.img_process_result.multi_hand_landmarks:
                    if draw:
                        self.mpDraw.draw_landmarks(self.img, hand, self.mpHands.HAND_CONNECTIONS)
            if handVal==0:

                handInfo = self.img_process_result.multi_hand_landmarks[0]

                if draw:
                    self.mpDraw.draw_landmarks(self.img, handInfo, self.mpHands.HAND_CONNECTIONS)
    
    
    def find_node_positions_of_hand(self, draw=True):
        
        if self.img_process_result.multi_hand_landmarks:
            handCount = 0
            for hand in self.img_process_result.multi_hand_landmarks:
                tempPositions = []
                for id, landmark in enumerate(hand.landmark):
                    height, width, channels = self.img.shape
                    centerX, centerY = int(landmark.x*width), int(landmark.y*height)
                    tempPositions.append((handCount, id, centerX, centerY))
                self.positions = (handCount, tempPositions)
                handCount += 1
            
        
            

                    

            
# for id, landmark in enumerate(hand.landmark):
# # print(id,landmark)
# height, width, channels = img.shape
# centerX, centerY = int(landmark.x*width), int(landmark.y*height)
# print(id, centerX, centerY)

# if id == 8:
#     cv2.circle(img, (centerX, centerY), 15, (255,0,255), cv2.FILLED)


    

def main():
    cap = cv2.VideoCapture(1)
    prevTime = 0
    currentTime = 0
    detector = HandDetector()

    while True:
        success, img = cap.read()
        detector.img = img
        detector.detect_hands()
        detector.find_node_positions_of_hand()
        print(detector.get_positions(0, (8,7,6,5)))
        
        currentTime = time.time()
        fps = 1/(currentTime-prevTime)
        prevTime = currentTime
        
        cv2.putText(detector.img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        
        
        
        cv2.imshow("Image", detector.img)
        cv2.waitKey(1) 
    
    
if __name__ == "__main__":
    main()