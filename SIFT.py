"""DISEÑADO POR:
    Andrés Santiago Rodriguez Prada
    Maria Alejandra Pedraza Cardenas
    Valentina Cortes Castellar
    
UMNG 2022
VISION POR COMPUTADORA"""

import cv2
import numpy as np
import math
from matplotlib import pyplot as plt


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

img =cv2.imread(r"C:\Users\Santiago - PC\Documents\GitHub\SIFT\torre.jpg") #Cambiar ruta de la imagen
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Pasar imagen a escala de grises
Escala0 = []
Escala1 = []
Escala2 = []
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
keypoints = img.copy()
for y in range(0,h-1):
    for x in range(0,w-1):
        if PF[y][x] != 0:
           keypoints[y,x] = [0,255,0]

cv2.imshow("Deteccion de KeyPoints", keypoints)
cv2.waitKey(0)

for y in range(0,alto-1):
    for x in range(0,ancho-1):
        if PF[y][x] != 0:
            file.write("\n%s"%PF[y][x])

magnitud = []
orientacion = []
contBatch=0
Batch1OR=[]
Batch1MAG=[]

Batch2OR=[]
Batch2MAG=[]

Batch3OR=[]
Batch3MAG=[]

Batch4OR=[]
Batch4MAG=[]

Batch5OR=[]
Batch5MAG=[]

Batch6OR=[]
Batch6MAG=[]

Batch7OR=[]
Batch7MAG=[]

Batch8OR=[]
Batch8MAG=[]

"""for y in range(0,h-1):
    for x in range(0,w-1):
        Gx, Gy = Conec4(y,x,PF)
        magnitud.append(np.square((Gx**2) + (Gy**2)))
        if Gx == 0:
            orientacion.append(0)
        else:
            orientacion.append(math.atan(Gy/Gx))"""

#Primera grupo de celdas 9x1

vectores = img.copy()
batchX=0
batchY=0
cont=0
batchX=int((w-1)/8)
batchY=int((w-1)/8)

for y in range(0,h-1):
    batchY=0
    for x in range(0,w-1):
        #print(batchX)
        Gx, Gy = Conec4(y,x,PF)
        """#Recorrer el primer Batch de pixeles
        if((x<=int((w-1)/8)) and (y<=int((h-1)/8))):
            Batch1MAG.append(np.square((Gx**2) + (Gy**2)))
            if Gx == 0:                        
                Batch1OR.append(0)
            else:
                Batch1OR.append(math.atan(Gy/Gx))    
        #Púnto de origen de los vectores"""

        if((x<=35)or (y<=65)):
            Batch1MAG.append(np.square((Gx**2) + (Gy**2)))
            if Gx == 0:                        
                Batch1OR.append(0)
            else:
                Batch1OR.append(math.atan(Gy/Gx))
            
        if(((x>35)or (y>65))and((x<=70)or (y<=130))):
            Batch2MAG.append(np.square((Gx**2) + (Gy**2)))
            if Gx == 0:                        
                Batch2OR.append(0)
            else:
                Batch2OR.append(math.atan(Gy/Gx))
        if(((x>70)or (y>130))and((x<=105)or (y<=195))):
            Batch3MAG.append(np.square((Gx**2) + (Gy**2)))
            if Gx == 0:                        
                Batch3OR.append(0)
            else:
                Batch3OR.append(math.atan(Gy/Gx))
        if(((x>105)or (y>195))and((x<=140)or (y<=260))):
            Batch4MAG.append(np.square((Gx**2) + (Gy**2)))
            if Gx == 0:                        
                Batch4OR.append(0)
            else:
                Batch4OR.append(math.atan(Gy/Gx))
        if(((x>140)or (y>260))and((x<=175)or (y<=325))):
            Batch5MAG.append(np.square((Gx**2) + (Gy**2)))
            if Gx == 0:                        
                Batch5OR.append(0)
            else:
                Batch5OR.append(math.atan(Gy/Gx))
        if(((x>175)or (y>325))and((x<=210)or (y<=390))):
            Batch6MAG.append(np.square((Gx**2) + (Gy**2)))
            if Gx == 0:                        
                Batch6OR.append(0)
            else:
                Batch6OR.append(math.atan(Gy/Gx))
        if(((x>210)or (y>390))and((x<=240)or (y<=455))):
            Batch7MAG.append(np.square((Gx**2) + (Gy**2)))
            if Gx == 0:                        
                Batch7OR.append(0)
            else:
                Batch1OR.append(math.atan(Gy/Gx))
        if(((x>240)or (y>455))and((x<=w-1)or (y<=h-1))):
            Batch8MAG.append(np.square((Gx**2) + (Gy**2)))
            if Gx == 0:                        
                Batch8OR.append(0)
            else:
                Batch8OR.append(math.atan(Gy/Gx))
   




plt.hist(Batch1MAG,bins=9)
plt.hist(Batch1OR,bins=9)


plt.hist(Batch2MAG,bins=9)
plt.hist(Batch2OR,bins=9)


plt.hist(Batch3MAG,bins=9)
plt.hist(Batch3OR,bins=9)


plt.hist(Batch4MAG,bins=9)
plt.hist(Batch1OR,bins=9)


plt.hist(Batch4MAG,bins=9)
plt.hist(Batch1OR,bins=9)


plt.hist(Batch5MAG,bins=9)
plt.hist(Batch5OR,bins=9)

plt.hist(Batch6MAG,bins=9)
plt.hist(Batch6OR,bins=9)


plt.hist(Batch7MAG,bins=9)
plt.hist(Batch7OR,bins=9)


plt.hist(Batch8MAG,bins=9)
plt.hist(Batch8OR,bins=9)
plt.xlim(0,1.5)
plt.show()



""" ax=plt.subplot(8,8)
    #Orientacion
    ax.plot(iOr,jOr)
    ax.set_xlabel('Orientación')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Agunlo vs frecuencia')"""


    

""" #Valor de maxima frecuencia de magnitud y dirección
    valmaxOr=iOr[np.where(jOr == jOr.max())]
    valmaxMag=iMag[np.where(jMag == jMag.max())]
    #Coordenada en x y y en la imágen
    Py=int(valmaxMag*(math.sin(valmaxOr)))
    Px=int(valmaxMag*(math.cos(valmaxOr)))
    #Ajustes de la linea
    color=(0,255,0)
    thickness = 9

    linea= cv2.line(vectores,(VecX,VecY),(Px,Py),color,thickness)"""








    #la cola del vector        
    #Se haya el valor de mayor frecuencia para la magnitud y la orientacion
    #valmaxOrientacion=iOr[np.where(jOr == jOr.max())]
 
        #Al final del ciclo se le suma una variable para que las consicionales se corran un puesto 
        #print(Gx)
        #print(Gy)
        #print(magnitud)
        #print(orientacion)

#iMag,plt.show()

#Prueba
"""histogram=[1,1,1,2,2,3,3,3,3,3,3,3,3,3,3,3,3,4]
#len(y) = len(x) - 1

y,x,_=plt.hist(histogram, bins=20)
valmax=x[np.where(y == y.max())]
print(x)
print(y)
print(valmax)
plt.show()"""
#print(str(valmax)+"El valor mayor es"+str(x[int(valmax)]))


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