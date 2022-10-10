import cv2
import numpy as np

def BuscarMax(j,i,imagen,imagen2,imagen3):
    auxiliar = np.zeros(9)
    auxiliar2 = np.zeros(9)
    auxiliar3 = np.zeros(9)
    x=0
    y=0
    for k in range(0,10,1):
        if k == 0:
            x=i-1;
            y=j-1;
            if x < 0 | y < 0 | y>=imagen.shape[0] | x>= imagen.shape[1]:
                auxiliar[k]=0
                auxiliar2[k]=0
                auxiliar3[k]=0
            else:
                auxiliar[k]=imagen[y][x]
                auxiliar2[k]=imagen2[y][x]
                auxiliar3[k]=imagen3[y][x]
        if k == 1:
            x=i-1;
            y=j;
            if x < 0 | y < 0 | y>=imagen.shape[0] | x>= imagen.shape[1]:
                auxiliar[k]=0
                auxiliar2[k]=0
                auxiliar3[k]=0
            else:
                auxiliar[k]=imagen[y][x]
                auxiliar2[k]=imagen2[y][x]
                auxiliar3[k]=imagen3[y][x]
        if k == 2:
          x=i-1;
          y=j+1;
          if x < 0 | y < 0 | y>=imagen.shape[0] | x>= imagen.shape[1]:
                auxiliar[k]=0
                auxiliar2[k]=0
                auxiliar3[k]=0
          else:
                auxiliar[k]=imagen[y][x]
                auxiliar2[k]=imagen2[y][x]
                auxiliar3[k]=imagen3[y][x]  
        if k == 3:
            x=i;
            y=j-1;
            if x < 0 | y < 0 | y>=imagen.shape[0] | x>= imagen.shape[1]:
                auxiliar[k]=0
                auxiliar2[k]=0
                auxiliar3[k]=0
            else:
                auxiliar[k]=imagen[y][x]
                auxiliar2[k]=imagen2[y][x]
                auxiliar3[k]=imagen3[y][x]   
        if k == 4: #Posición del Pixel central de la ventana
            x=i;
            y=j;
            if x < 0 | y < 0 | y>=imagen.shape[0] | x>= imagen.shape[1]:
                auxiliar[k]=0 #Central de la ventana de evaluacion
                auxiliar2[k]=0
                auxiliar3[k]=0
            else:
                auxiliar[k]=imagen[y][x]
                auxiliar2[k]=imagen2[y][x]
                auxiliar3[k]=imagen3[y][x]
        if k == 5:
            x=i;
            y=j+1;
            if x < 0 | y < 0 | y>=imagen.shape[0] | x>= imagen.shape[1]:
                auxiliar[k]=0
                auxiliar2[k]=0
                auxiliar3[k]=0
            else:
                auxiliar[k]=imagen[y][x]
                auxiliar2[k]=imagen2[y][x]
                auxiliar3[k]=imagen3[y][x]
        if k == 6:
            x=i+1;
            y=j-1;
            if x < 0 | y < 0 | y>=imagen.shape[0] | x>= imagen.shape[1]:
                auxiliar[k]=0
                auxiliar2[k]=0
                auxiliar3[k]=0
            else:
                auxiliar[k]=imagen[y][x]
                auxiliar2[k]=imagen2[y][x]
                auxiliar3[k]=imagen3[y][x]
        if k == 7:
            x=i+1;
            y=j;
            if x < 0 | y < 0 | y>=imagen.shape[0] | x>= imagen.shape[1]:
                auxiliar[k]=0
                auxiliar2[k]=0
                auxiliar3[k]=0
            else:
                auxiliar[k]=imagen[y][x]
                auxiliar2[k]=imagen2[y][x]
                auxiliar3[k]=imagen3[y][x] 
        if k == 8:
            x=i+1;
            y=j+1;
            if x < 0 | y < 0 | y>=imagen.shape[0] | x>= imagen.shape[1]:
                auxiliar[k]=0
                auxiliar2[k]=0
                auxiliar3[k]=0
            else:
                auxiliar[k]=imagen[y][x]
                auxiliar2[k]=imagen2[y][x]
                auxiliar3[k]=imagen3[y][x]
        if k == 9:
            MaxAux = auxiliar.max()
            MaxAux2 = auxiliar2.max()
            MaxAux3 = auxiliar3.max()
            MinAux = auxiliar.min()
            MinAux2 = auxiliar2.min()
            MinAux3 = auxiliar3.min()
            if auxiliar[4] >= MaxAux and auxiliar[4] >= MaxAux2 and auxiliar[4] >= MaxAux3:
                valor = True
            elif auxiliar[4] <= MinAux and auxiliar[4] <= MinAux2 and auxiliar[4] <= MinAux3:
                valor = False
            else:
                valor = None
    return valor,auxiliar[4]

img =cv2.imread('D:/Universidad/Decimo Semestre/Vision PC/SIFT/Gato.jpg') #Cambiar ruta de la imagen
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
Escala0 = []
Escala1 = []
Escala2 = []
Escala3 = []
gray_gauss = np.float32(cv2.GaussianBlur(gray,(3,3),0))
Escala0.append(gray_gauss)
Escala0[0] = np.float32(Escala0[0]) #De aqui en adelante todos los gauseannos se almacenan en Float32 =)
Termino = False
contador = 0
Diferencia0 = []
Diferencia1 = []
Diferencia2 = []
Diferencia3 = []
#APLICAR GAUSEANOS 0
for i in range(0,3):
    Escala0.append(cv2.GaussianBlur(Escala0[i],(3,3),0))
    Escala0[i] = np.float32(Escala0[i])
#RESTA DE GAUSEANOS 0
for i in range(0,3):
    Diferencia0.append(Escala0[i]-Escala0[i+1])
    #cv2.imshow("Escala0", Diferencia0[i])
    #cv2.waitKey(0)
print("Hay :",len(Escala0), "imagenes con Gauss")
print("Hay :",len(Diferencia0), "diferencias Gauseanas")
#ESCALAS 1
for i in range(0,4):
    ancho = int(len(Escala0[0][0])*0.50)
    alto = int(len(Escala0[0])*0.50)
    dim = (ancho,alto) #Ancho, Alto
    imgaux1 = cv2.resize(Escala0[i],dim)
    Escala1.append(imgaux1)
#APLICAR GAUSEANOS 1
for i in range(0,4):
    Escala1[i] = np.float32(Escala1[i])
    Escala1[i] = cv2.GaussianBlur(Escala1[i],(3,3),0)
#RESTA DE GAUSEANOS 1
for i in range(0,3):
    Diferencia1.append(Escala1[i]-Escala1[i+1])
    #cv2.imshow("Escala1", Diferencia1[i])
    #cv2.waitKey(0)
print("Hay :",len(Escala1), "imagenes con Gauss")
print("Hay :",len(Diferencia1), "diferencias Gauseanas")
#ESCALAS 2
for i in range(0,4):
    ancho = int(len(Escala1[0][0])*0.50)
    alto = int(len(Escala1[0])*0.50)
    dim = (ancho,alto) #Ancho, Alto
    imgaux1 = cv2.resize(Escala1[i],dim)
    Escala2.append(imgaux1)
#APLICAR GAUSEANOS 2
for i in range(0,4):
    Escala2[i] = np.float32(Escala2[i])
    Escala2[i] = cv2.GaussianBlur(Escala2[i],(3,3),0)
#RESTA DE GAUSEANOS 2
for i in range(0,3):
    Diferencia2.append(Escala2[i]-Escala2[i+1])
    #cv2.imshow("Escala2", Diferencia2[i])
    #cv2.waitKey(0)
print("Hay :",len(Escala2), "imagenes con Gauss")
print("Hay :",len(Diferencia2), "diferencias Gauseanas")
#ESCALAS 3
for i in range(0,4):
    ancho = int(len(Escala2[0][0])*0.50)
    alto = int(len(Escala2[0])*0.50)
    dim = (ancho,alto) #Ancho, Alto
    imgaux1 = cv2.resize(Escala2[i],dim)
    Escala3.append(imgaux1)
#APLICAR GAUSEANOS 3
for i in range(0,4):
    Escala3[i] = np.float32(Escala3[i])
    Escala3[i] = cv2.GaussianBlur(Escala3[i],(3,3),0)
#RESTA DE GAUSEANOS 3
for i in range(0,3):
    Diferencia3.append(Escala3[i]-Escala3[i+1])
    #cv2.imshow("Escala3", Diferencia3[i])
    #cv2.waitKey(0)
print("Hay :",len(Escala3), "imagenes con Gauss")
print("Hay :",len(Diferencia3), "diferencias Gauseanas")

#Ancho y alto Img Original
#NOTA SI SE CAMBIA LOS PMENORES ASI = NP.ONES... SALE BLANCA LA IMAGEN XD y los negativos creo que son en negro --> Se usan float32 porque son operaciones intermedias
#BUSQUEDA DE MAXIMOS Y MINIMOS CON LOS 26 VECINOS
ancho = int(len(Escala0[0][0]))
alto = int(len(Escala0[0]))
Pmayores = np.zeros([alto,ancho])
Pmayores = np.float32(Pmayores)
Pmenores = np.zeros([alto,ancho])
Pmenores = np.float32(Pmenores)
for y in range(0,alto-1):
    for x in range(0,ancho-1):
        Bandera, Pixel = BuscarMax(y,x,Diferencia0[0],Diferencia0[1],Diferencia0[2])
        if Bandera == True:
            Pmayores[y][x] = Pixel
        elif Bandera == False:
            Pmenores[y][x] = Pixel
            
#Ancho y alto Img primer octava
ancho = int(len(Escala1[0][0]))
alto = int(len(Escala1[0]))
Pmayores1 = np.zeros([alto,ancho])
Pmayores1 = np.float32(Pmayores1)
Pmenores1 = np.zeros([alto,ancho])
Pmenores1 = np.float32(Pmenores1)
for y in range(0,alto-1):
    for x in range(0,ancho-1):
        Bandera1, Pixel1 = BuscarMax(y,x,Diferencia1[0],Diferencia1[1],Diferencia1[2])
        if Bandera1 == True:
            Pmayores1[y][x] = Pixel1
        elif Bandera1 == False:
            Pmenores1[y][x] = Pixel1
            
#Ancho y alto Img segunda octava
ancho = int(len(Escala2[0][0]))
alto = int(len(Escala2[0]))
Pmayores2 = np.zeros([alto,ancho])
Pmayores2 = np.float32(Pmayores2)
Pmenores2 = np.zeros([alto,ancho])
Pmenores2 = np.float32(Pmenores2)
for y in range(0,alto-1):
    for x in range(0,ancho-1):
        Bandera2, Pixel2 = BuscarMax(y,x,Diferencia2[0],Diferencia2[1],Diferencia2[2])
        if Bandera2 == True:
            Pmayores2[y][x] = Pixel2
        elif Bandera2 == False:
            Pmenores2[y][x] = Pixel2

#Ancho y alto Img tercer octava            
ancho = int(len(Escala3[0][0]))
alto = int(len(Escala3[0]))
Pmayores3 = np.zeros([alto,ancho])
Pmayores3 = np.float32(Pmayores3)
Pmenores3 = np.zeros([alto,ancho])
Pmenores3 = np.float32(Pmenores3)
for y in range(0,alto-1):
    for x in range(0,ancho-1):
        Bandera3, Pixel3 = BuscarMax(y,x,Diferencia3[0],Diferencia3[1],Diferencia3[2])
        if Bandera3 == True:
            Pmayores3[y][x] = Pixel3
        elif Bandera3 == False:
            Pmenores3[y][x] = Pixel3

cv2.imshow("Pmayores", Pmayores)
cv2.imshow("Pmenores", Pmenores)
cv2.imshow("Pmayores1", Pmayores1)
cv2.imshow("Pmenores1", Pmenores1)
cv2.imshow("Pmayores2", Pmayores2)
cv2.imshow("Pmenores2", Pmenores2)
cv2.imshow("Pmayores3", Pmayores3)
cv2.imshow("Pmenores3", Pmenores3)
cv2.waitKey(0)

