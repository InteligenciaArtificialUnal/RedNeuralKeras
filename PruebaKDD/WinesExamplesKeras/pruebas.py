# Import numpy
import numpy as np
pedo = [[]]

prediccion= [[  1.77939155e-03],
 [  9.71560657e-01],
 [  4.79573573e-05],
 [  1.10898036e-05],
 [  1.23733201e-07]]

x = np.array([[0.0],
 [  1.0],
 [  0.0],
 [  0.0],
 [  0.0]])


for i in range(len(prediccion)):
    prediccion[i][0]=round(prediccion[i][0])

for i in prediccion:
    print (i, round(i[0], 0))
    pedo[0].append(int(round(i[0], 0)))

prediccion = np.array(prediccion).astype(np.int32)
print (prediccion[1][0])
predicciontecho = round(prediccion[1][0], 0)
print("techo", predicciontecho)
#print("peditooo: ",(int(pedo[0])).astype(np.int32))
print (prediccion)
print("pedo: ", type(prediccion.astype(np.int32)))
