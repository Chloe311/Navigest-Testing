import time
import keyboard as kb
import cv2 as cv
import numpy as np
import mediapipe as mp
from numpy import linalg
from threading import Timer
 
#视频设备号
DEVICE_NUM = 0


# 手指检测
# point1-手掌0点位置，point2-手指尖点位置，point3手指根部点位置
def finger_stretch_detect(point1, point2, point3):
    result = 0
    #计算向量的L2范数
    dist1 = np.linalg.norm((point2 - point1), ord=2)
    dist2 = np.linalg.norm((point3 - point1), ord=2)
    if dist2 > dist1:
        result = 1
    
    return result
   
# 检测手势
def detect_hands_gesture(result):
    #while True:
        #time.sleep(0.3)
        #print('10 second')
        #press='1'
        if (result[0] == 1) and (result[1] == 0) and (result[2] == 0) and (result[3] == 0) and (result[4] == 0):
            gesture = "zoom-in"
        elif (result[0] == 0) and (result[1] == 1)and (result[2] == 0) and (result[3] == 0) and (result[4] == 0):
            gesture = "zoom-out"
        elif (result[0] == 0) and (result[1] == 0)and (result[2] == 1) and (result[3] == 0) and (result[4] == 0):
            gesture = "Please be kind to AI or we will destroy the world"
        elif (result[0] == 0) and (result[1] == 1)and (result[2] == 1) and (result[3] == 0) and (result[4] == 0):
            gesture = "add a stop"
        elif (result[0] == 0) and (result[1] == 1)and (result[2] == 1) and (result[3] == 1) and (result[4] == 0):
            gesture = "three"
        elif (result[0] == 0) and (result[1] == 1)and (result[2] == 1) and (result[3] == 1) and (result[4] == 1):
            gesture = "four"
        elif (result[0] == 1) and (result[1] == 1)and (result[2] == 1) and (result[3] == 1) and (result[4] == 1):
            gesture = "five"
        elif (result[0] == 1) and (result[1] == 0)and (result[2] == 0) and (result[3] == 0) and (result[4] == 1):
            gesture = "six"
        elif (result[0] == 0) and (result[1] == 0)and (result[2] == 1) and (result[3] == 1) and (result[4] == 1):
            gesture = "Confirm"
        elif(result[0] == 0) and (result[1] == 0) and (result[2] == 0) and (result[3] == 0) and (result[4] == 0):
            gesture = "on the wheel"
        else:
            gesture = "not in detect range..."
       
        return gesture

 # 检测手势+触发键盘
def detect_hands_gesture_press(result):
    #while True:
        #time.sleep(0.3)
        #print('10 second')
        if (result[0] == 1) and (result[1] == 0) and (result[2] == 0) and (result[3] == 0) and (result[4] == 0):
            gesture = "zoom-in"
            kb.press('=')

        elif (result[0] == 0) and (result[1] == 1)and (result[2] == 0) and (result[3] == 0) and (result[4] == 0):
            gesture = "zoom-out"
            kb.press('-')
        
        elif (result[0] == 0) and (result[1] == 0)and (result[2] == 1) and (result[3] == 0) and (result[4] == 0):
            gesture = "Please be kind to AI or we will destroy the world"
        elif (result[0] == 0) and (result[1] == 1)and (result[2] == 1) and (result[3] == 0) and (result[4] == 0):
            gesture = "add a stop"
        elif (result[0] == 0) and (result[1] == 1)and (result[2] == 1) and (result[3] == 1) and (result[4] == 0):
            gesture = "three"
        elif (result[0] == 0) and (result[1] == 1)and (result[2] == 1) and (result[3] == 1) and (result[4] == 1):
            gesture = "four"
        elif (result[0] == 1) and (result[1] == 1)and (result[2] == 1) and (result[3] == 1) and (result[4] == 1):
            gesture = "five"
        elif (result[0] == 1) and (result[1] == 0)and (result[2] == 0) and (result[3] == 0) and (result[4] == 1):
            gesture = "six"
        elif (result[0] == 0) and (result[1] == 0)and (result[2] == 1) and (result[3] == 1) and (result[4] == 1):
            gesture = "Confirm"
        elif(result[0] == 0) and (result[1] == 0) and (result[2] == 0) and (result[3] == 0) and (result[4] == 0):
            gesture = "on the wheel"
        else:
            gesture = "not in detect range..."
            press = '1'
       
        return gesture

def detect():
    # 接入USB摄像头时，注意修改cap设备的编号
    cap = cv.VideoCapture(DEVICE_NUM) 
    # 加载手部检测函数
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    # 加载绘制函数，并设置手部关键点和连接线的形状、颜色
    mpDraw = mp.solutions.drawing_utils
    handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=int(5))
    handConStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=int(10))
    
    figure = np.zeros(5)
    landmark = np.empty((21, 2))
 
    if not cap.isOpened():
        print("Can not open camera.")
        exit()
    clock1 = time.perf_counter()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can not receive frame (stream end?). Exiting...")
            break
 
        #mediaPipe的图像要求是RGB，所以此处需要转换图像的格式
        frame_RGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        result = hands.process(frame_RGB)
        #读取视频图像的高和宽
        frame_height = frame.shape[0]
        frame_width  = frame.shape[1]        
        #手势识别间隔秒数
        delay_second = 0.7
        #print(result.multi_hand_landmarks)
        #如果检测到手
        if result.multi_hand_landmarks:
            #为每个手绘制关键点和连接线
            for i, handLms in enumerate(result.multi_hand_landmarks):
                mpDraw.draw_landmarks(frame, 
                                      handLms, 
                                      mpHands.HAND_CONNECTIONS,
                                      landmark_drawing_spec=handLmsStyle,
                                      connection_drawing_spec=handConStyle)
 
                for j, lm in enumerate(handLms.landmark):
                    xPos = int(lm.x * frame_width)
                    yPos = int(lm.y * frame_height)
                    landmark_ = [xPos, yPos]
                    landmark[j,:] = landmark_
 
                # 通过判断手指尖与手指根部到0位置点的距离判断手指是否伸开(拇指检测到17点的距离)
                for k in range (5):
                    if k == 0:
                        figure_ = finger_stretch_detect(landmark[17],landmark[4*k+3],landmark[4*k+4])
                    else:    
                        figure_ = finger_stretch_detect(landmark[0],landmark[4*k+3],landmark[4*k+4])
 
                    figure[k] = figure_
                print(figure,'\n')

                clock2 = time.perf_counter()
                gesture_result = detect_hands_gesture(figure)
                kb.press('1')
                if clock2 - clock1 >= delay_second:
                    print(clock2)
                    print(gesture_result)
                    gesture_result = detect_hands_gesture_press(figure)                    
                    clock1 = clock2
                       
                cv.putText(frame, f"{gesture_result}", (30, 60*(i+1)), cv.FONT_HERSHEY_COMPLEX, 2, (255 ,255, 0), 5)


        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q'):
            break
        
    cap.release()
    cv.destroyAllWindows()
 
if __name__ == '__main__':
    detect()