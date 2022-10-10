import cv2
import numpy as np

img =cv2.imread('D:/Universidad/Decimo Semestre/Vision PC/SIFT/Gato.jpg') #Cambiar ruta de la imagen
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
Escala0 = []
Escala1 = []
Escala2 = []
Escala3 = []
Escala0.append(gray)
Termino = False
contador = 0
Diferencia0 = []
Diferencia1 = []
Diferencia2 = []
Diferencia3 = []
#APLICAR GAUSEANOS 0
for i in range(0,5):
    Escala0.append(cv2.GaussianBlur(Escala0[i],(3,3),0))
#RESTA DE GAUSEANOS 0
for i in range(0,4):
    Diferencia0.append(Escala0[i]-Escala0[i+1])
#ESCALAS 1
for i in range(0,5):
    ancho = int(len(Escala0[0][0])*0.50)
    alto = int(len(Escala0[0])*0.50)
    dim = (ancho,alto) #Ancho, Alto
    imgaux1 = cv2.resize(Escala0[i],dim)
    Escala1.append(imgaux1)
#APLICAR GAUSEANOS 1
for i in range(1,5):
    Escala1[i] = cv2.GaussianBlur(Escala1[i],(3,3),0)
#RESTA DE GAUSEANOS 1
for i in range(0,4):
    Diferencia1.append(Escala1[i]-Escala1[i+1])
    cv2.imshow("Escala1", Diferencia1[i])
    cv2.waitKey(0)
#ESCALAS 2
for i in range(0,5):
    ancho = int(len(Escala1[0][0])*0.50)
    alto = int(len(Escala1[0])*0.50)
    dim = (ancho,alto) #Ancho, Alto
    imgaux1 = cv2.resize(Escala1[i],dim)
    Escala2.append(imgaux1)
#APLICAR GAUSEANOS 2
for i in range(1,5):
    Escala2[i] = cv2.GaussianBlur(Escala2[i],(3,3),0)
#RESTA DE GAUSEANOS 2
for i in range(0,4):
    Diferencia2.append(Escala2[i]-Escala2[i+1])
    cv2.imshow("Escala2", Diferencia2[i])
    cv2.waitKey(0)
#ESCALAS 3
for i in range(0,5):
    ancho = int(len(Escala2[0][0])*0.50)
    alto = int(len(Escala2[0])*0.50)
    dim = (ancho,alto) #Ancho, Alto
    imgaux1 = cv2.resize(Escala2[i],dim)
    Escala3.append(imgaux1)
#APLICAR GAUSEANOS 3
for i in range(1,5):
    Escala3[i] = cv2.GaussianBlur(Escala3[i],(3,3),0)
#RESTA DE GAUSEANOS 3
for i in range(0,4):
    Diferencia3.append(Escala3[i]-Escala3[i+1])
    cv2.imshow("Escala3", Diferencia3[i])
    cv2.waitKey(0)

