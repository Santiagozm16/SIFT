import cv2
import numpy as np
import math

def BuscarMaxMin(i0,j0,imagen,imagen2,imagen3):
    k = 0
    valor = 0
    auxiliar = np.zeros(9)
    auxiliar2 = np.zeros(9)
    auxiliar3 = np.zeros(9)
    for i in range(-1, 2):
        for j in range(-1, 2):
            x=i0+(i)
            y=j0+(j)
            if(x < 0 or y < 0 or x >=imagen.shape[0] or y >= imagen.shape[1] ):
                auxiliar[k]=0  
            else:
                auxiliar[k] = imagen[x][y]
                auxiliar2[k] = imagen2[x][y]
                auxiliar3[k] = imagen3[x][y]
            k = k + 1    
    k = 0
    #print(auxiliar)
    for i in range(0, 3):
        for j in range(0, 3):
            #print(auxiliar[k])
            #print(filtro[i][j])
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

def Conec4(i0,j0,imagen):
    k = 0
    Gx = 0.0
    Gy = 0.0
    auxiliar = np.zeros(9)
    vecinos = np.zeros(4)
    for i in range(-1, 2):
        for j in range(-1, 2):
            x=i0+(i)
            y=j0+(j)
            if(x < 0 or y < 0 or x >=imagen.shape[0] or y >= imagen.shape[1] ):
                auxiliar[k]=0  
            else:
                auxiliar[k] = imagen[x][y]
            k = k + 1    
    k = 0
    #print(auxiliar)
    for i in range(0, 3):
        for j in range(0, 3):
            Gx = auxiliar[3] - auxiliar[5]
            Gy = auxiliar[1] - auxiliar[7]
    return Gx, Gy

#Funciones con switch / case
"""def BuscarMax(j,i,imagen,imagen2,imagen3):
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
    return valor,auxiliar[4]"""

img =cv2.imread(r"D:\Documentos\Universidad\Decimo Semestre\Vision por computadora\SIFT\SIFT\torre.jpg") #Cambiar ruta de la imagen
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Pasar imagen a escala de grises
Escala0 = []
Escala1 = []
Escala2 = []
Escala3 = []
gray_gauss = np.float32(cv2.GaussianBlur(gray,(3,3),0)) #Filtro Gaussiano primera imagen
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
print("Hay :",len(Escala2), "imagenes con Gauss")
print("Hay :",len(Diferencia2), "diferencias Gauseanas")


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
        Bandera, Pixel = BuscarMaxMin(y,x,Diferencia0[1],Diferencia0[0],Diferencia0[2])
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
        Bandera1, Pixel1 = BuscarMaxMin(y,x,Diferencia1[1],Diferencia1[0],Diferencia1[2])
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
        Bandera2, Pixel2 = BuscarMaxMin(y,x,Diferencia2[1],Diferencia2[0],Diferencia2[2])
        if Bandera2 == True:
            Pmayores2[y][x] = Pixel2
        elif Bandera2 == False:
            Pmenores2[y][x] = Pixel2

"""cv2.imshow("Pmayores", Pmayores)
cv2.imshow("Pmenores", Pmenores)
cv2.imshow("Pmayores1", Pmayores1)
cv2.imshow("Pmenores1", Pmenores1)
cv2.imshow("Pmayores2", Pmayores2)
cv2.imshow("Pmenores2", Pmenores2)
cv2.imshow("Pmayores3", Pmayores3)
cv2.imshow("Pmenores3", Pmenores3)
cv2.waitKey(0)"""

P = Pmayores + Pmenores
P1 = Pmayores1 + Pmenores1
P2 = Pmayores2 + Pmenores2

ancho = int(len(Escala0[0][0]))
alto = int(len(Escala0[0]))
dim = (ancho,alto) #Ancho, Alto
P1 = cv2.resize(P1,dim)
P2 = cv2.resize(P2,dim)

for y in range(0,alto-1):
    for x in range(0,ancho-1):
        Bandera, Pixel = BuscarMaxMin(y,x,P1,P,P2)
        if Bandera == True:
            Pmayores[y][x] = Pixel
            #print(Pixel)
        elif Bandera == False:
            Pmenores[y][x] = Pixel

PF = Pmayores + Pmenores

h = len(PF)
w = len(PF[0])

for y in range(0,h-1):
    for x in range(0,w-1):
        if PF[y][x] < 0.03:
           PF[y][x] = 0

file = open("Matriz.txt", "w")
for y in range(0,alto-1):
    for x in range(0,ancho-1):
        if PF[y][x] != 0:
            file.write("\n%s"%PF[y][x])

magnitud = []
orientacion = []

for y in range(0,h-1):
    for x in range(0,w-1):
        Gx, Gy = Conec4(y,x,PF)
        magnitud.append(np.square((Gx**2) + (Gy**2)))
        if Gx == 0:
            orientacion.append(0)
        else:
            orientacion.append(math.atan(Gy/Gx))
        #print(Gx)
        #print(Gy)
        #print(magnitud)
        #print(orientacion)

print(magnitud)
cv2.imshow(":c", PF)
cv2.waitKey(0) 


#Seleccion Keypoint - Hessian

"""Px = []
Py = []
Pxy = []
contador = 0
contador2 = 0
for y in range(0,alto-1):
    for x in range(0,ancho-1):
        if Pmayores[y][x] != 0:
            print("entro")
            Px.append(Pmayores[y][x])

for y in range(0,alto-1):
    for x in range(0,ancho-1):
        if Pmenores[y][x] != 0:
            contador = contador + 1
            if  contador < 927:
                Py.append(Pmenores[y][x])

print(len(Px))
print(len(Py))
trace = []
for i in range (926):
    Pxy.append(Py[i]*Px[i])
print(len(Pxy))
detH = Pxy - np.square(Pxy)
print(detH)
print(len(detH))
for i in range (926):
    trace.append(Px[i] + Py[i])

print(len(trace))
f = np.square(trace)/detH
f = f*-1
print(len(f))
r = 0.03

keypoint = []
for y in range(926):
        if f[y] > r:
           keypoint.append(255)
           print(keypoint)
print(len(keypoint))
PF = Pmayores + Pmenores
print(PF)
#Aqui ya es la resolución de todas las gauseanas con sus maximos y minimos, una unica imagen.
cv2.imshow("Pmenores3", PF)
cv2.waitKey(0)"""