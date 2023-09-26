import numpy as np
import pyautogui
import imutils
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

finger_tips =[8, 12, 16, 20]
thumb_tip= 4

while True:
    ret,img = cap.read()
    img = cv2.flip(img, 1)
    h,w,c = img.shape
    results = hands.process(img)


    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            #acessando os pontos de referência pela sua posição
            lm_list=[]
            for id ,lm in enumerate(hand_landmark.landmark):
                lm_list.append(lm)

            #array para manter verdadeiro ou falso se o dedo estiver dobrado    
            finger_fold_status =[]
            for tip in finger_tips:
                #obtendo a posição da ponta do ponto de referência e desenhando o círculo azul
                x,y = int(lm_list[tip].x*w), int(lm_list[tip].y*h)
                cv2.circle(img, (x,y), 15, (255, 0, 0), cv2.FILLED)

                #escrevendo a condição para verificar se o dedo está dobrado, ou seja, verificar se o valor inicial da ponta do dedo é menor que a posição inicial do dedo que é o marco interno para o dedo indicador    
                #se dedo estiver dobrado, mudar a cor para verde
                if lm_list[tip].x < lm_list[tip - 3].x:
                    cv2.circle(img, (x,y), 15, (0, 255, 0), cv2.FILLED)
                    finger_fold_status.append(True)
                else:
                    finger_fold_status.append(False)

            print(finger_fold_status)

             #verificando se todos os dedos estão dobrados
            if all(finger_fold_status):
                # faça uma captura de tela da tela e armazene-a na memória, depois
                # converta a imagem PIL/Pillow em um array NumPy compatível com o OpenCV
                # e, finalmente, grave a imagem no disco
                image = pyautogui.screenshot()
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                cv2.imwrite("in_memory_to_disk.png", image)

                # desta vez, faça uma captura de tela diretamente para o disco
                pyautogui.screenshot("straight_to_disk.png")

                # podemos então carregar nossa captura de tela do disco no formato OpenCV
                image = cv2.imread("straight_to_disk.png")
                cv2.imshow("Captura de Tela", imutils.resize(image, width=600))





            mp_draw.draw_landmarks(img, hand_landmark,
            mp_hands.HAND_CONNECTIONS, mp_draw.DrawingSpec((0,0,255),2,2),
            mp_draw.DrawingSpec((0,255,0),4,2))
    

    cv2.imshow("Rastreamento de Maos", img)
    cv2.waitKey(1)