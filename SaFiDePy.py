
# # Recibe una matriz de datos por columna correspondientes a:
# #       [tiempo posXOjo posYOjo]
# # 
# # Entrega como salida un vector correspondiente a los tiempos de las
# # sacadas para cada ojo (por ahora solo el ojo derecho) el cual se calculó
# # con los siguientes parámetros:
# #       output = [matrizSacadas, matrizFijacion, matrizBlink] 
# #       
# # Laboratorio de Neurosistemas
# # Christ Devia & Samuel Madariaga
# # Febrero 2019

# # Implementación Python Camilo Espinosa
# # Mayo 2022

import numpy as np

def SaFiDePy(time, X, Y, sampling_rate = 1000,
                             pix2grad = 39.38,
                             acce1Thld = [-4000, 4000],
                             velThld = 30,
                             ampliSaccThld = 0.1,
                             lengthSaccThld = 4,
                             getPupilBlinks =False):
# PARTE 0: Calculo de los tiempos de parpadeos con script

    try:
        if getPupilBlinks:
            pass
           # blinks_data_positions = based_noise_blinks_detection(sam[:,3],sampling_rate)	# get blink positions using the noise-based approach
            #tpoBlink = reshape(blinks_data_positions',2,[])'
            #tpoBlink = index2time(tpoBlink/2,sam(:,1))
        else:
            tpoBlink = np.array([[0, 1]])
    except:
        tpoBlink = np.array([[0, 1]])
        print('WARNING: no se midió pestañeos por pupila')
    #------ PARTE 1: Calcula la velocidad y la aceleracion
    nsam = len(time)
    cal = np.zeros((nsam-2,5)) # [timestamp dist vel acc sacc?]
    try:
        pix2grad = pix2grad
    except:
        pix2grad = 1
        print('WARNING: no pix2grad')
    DTgr = np.zeros(nsam)

    vel = np.zeros(nsam)
    for i in range(nsam-1):
        # distancia total recorrida en grados visuales
        DTgr[i+1] = np.sqrt((X[i+1] - X[i])**2 + (Y[i+1] - Y[i])**2) / pix2grad 
        # velocidad en grados/segundos
        vel[i+1] = DTgr[i+1]/((time[i+1]-time[i])/sampling_rate) # *10^-3 ya que los timestamps están en milisegundos

    acc = np.zeros((len(vel)))
    for i in range(len(vel)-2):
        # aceleracion en grados/seg^2
        acc[i+1] = (vel[i+1]-vel[i])/( (time[i+1]-time[i] )/sampling_rate) 
        # Aceleracion y velocidad
        if  acc[i+1] <= acce1Thld[0] or acc[i+1] >= acce1Thld[1]  or vel[i+1] >= velThld:
            cal[i+1,4] = 1 # marca con un 1 los periodos de sacada

    ## PARTE 2: Detecta los tiempos de inicio y termino de sacada
    dife = np.diff(cal[:,4])
    T1 = np.argwhere(dife>0) # Onset of saccades
    T2 = np.argwhere(dife<0)+1 # Offset of saccades

    # Limpia las sacadas al inicio y final del registro
    if T2[1] <= T1[1]:
        T2 = T2[1:]
    T1 = T1[:len(T2)]

    # Calcula la DURACION de la sacada en samples
    Dsac = T2-T1

    # Verifica que no exitan las sacadas negativas
    if np.any(Dsac<0):
        print('Existe una sacada negativa')
        np.delete(T1,T1[Dsac<0])
        np.delete(T2,T2[Dsac<0])

    ## PARTE 3: Solo deja las sacadas con amplitud mayor a 0.1 grados visuales
    # Calcula la amplitud total por sacada
    Asac = np.zeros((len(T1),1))

    for i in range(len(T1)):
        Asac[i] = np.sum(DTgr[T1[i].item():T2[i].item()+1]) # si hay un NaN dara NaN

    ########################## Ojo 0.5 segun paper de los monkeys
    # Verifica cuales son mayores al umbral de deteccion, en este caso > 0.1
    cond = Asac > ampliSaccThld

    # Flag de que esa sacada no es un blink (por que su amplitud es NaN) y debe dejarlo
    fgbli = np.isnan(Asac)

    # Mantiene los que son NaN pues indican que esa sacada es un blink, fuerza
    # a que cond sea 0 solo cuando la amplitud es menor que 0.1

    cond2 = cond+fgbli
    aT1 = T1[cond2]
    aT2 = T2[cond2]

    # Cálculo de la velocidad peak
    velPeak = np.zeros(len(aT1))
    for i in range(len(aT1)):
        velPeak[i] = np.max(vel[aT1[i]:aT2[i]])

    # Se generan los tiempos de las sacadas
    tpoSac = np.array([np.array(time[aT1]), 
                        np.array(time[aT2]), 
                        np.array(Asac[cond2>0]), 
                        np.array(time[aT2]) - np.array(time[aT1]), 
                        np.array(velPeak),
                        np.array(X[aT1]), 
                        np.array(Y[aT1]), 
                        np.array(X[aT2]), 
                        np.array(Y[aT2]), 
                        np.zeros(len(Y[aT2]))]).T

    ## PARTE 4: Con los tiempos de los pestañeos elimina las sacadas insertas entre estos
    contB = 0
    aux = np.zeros(len(tpoSac[:,0]))
    for i in range(1,len(tpoSac)):
        # Actualiza el blink
        try:
            if tpoSac[i,0] > tpoBlink[contB,1] and contB < len(tpoBlink):
                contB = contB + 1
        except:
            pass
        # marca las sacadas menores al tiempo límite
        if tpoSac[i,1]-tpoSac[i,0] <= lengthSaccThld:
            aux[i] = 1
        
        # Busca las sacadas antes y despues del blink (wd = 60)
        try:
            if tpoSac[i,1] > tpoBlink[contB,1] and tpoSac[i,0] < tpoBlink[contB,1]:
                aux[i] = 1
        except:
            pass
        # Marca los overshoot
        if tpoSac[i,0]-tpoSac[i-1,1] <= 16 and tpoSac[i,2] <= 1.5: 
            tpoSac[i-1,1] = tpoSac[i,1]
            tpoSac[i-1,2] = tpoSac[i,2] + tpoSac[i-1,2]
            tpoSac[i-1,3] = tpoSac[i,3] + tpoSac[i-1,3]
            tpoSac[i-1,6] = tpoSac[i,6]
            tpoSac[i-1,7] = tpoSac[i,7] 
            tpoSac[i-1,8] = 1
            aux[i] = 1

    # Eliminina los blink de tamaño cero o negativo y los de mas de 10s
    tpoBlink = tpoBlink[(tpoBlink[:,1]-tpoBlink[:,0])>0,:]
    tpoBlink = tpoBlink[tpoBlink[:,1]-tpoBlink[:,0]<10*sampling_rate,:]

    # modifica la matriz de sacadas y los inicio y finales de sacadas
    tpoSac = tpoSac[((aux -1)**2).astype(bool),:]
    aT1 = aT1[((aux -1)**2).astype(bool)]
    aT2 = aT2[((aux -1)**2).astype(bool)]


    ## PARTE 5: Genera la matriz de fijaciones a partir de los datos de sac
    tf1 = aT2[:-1]+1 
    tf2 = aT1[1:]-1

    tpoFix = np.array([np.array(time[tf1]), 
                        np.array(time[tf2]), 
                        np.array(time[tf2]) - np.array(time[tf1]), 
                        np.array(X[tf1]), 
                        np.array(Y[tf2]), 
                        np.zeros(len(Y[tf2]))]).T


    contB = 0
    aux = np.zeros(len(tpoFix[:,0]))
    i=2
    tpoFix[i,1]-tpoFix[i,0]
    i=0
    while  i <= len(tpoFix)-1:
        
        # marca las fijaciones menores a 10 ms
        if tpoFix[i,1]-tpoFix[i,0] <= 10:
            aux[i] = 1
        
        # marca las fijaciones con un blink ente medio
        if  tpoFix[i,1] > tpoBlink[contB,0] and tpoFix[i,0] < tpoBlink[contB,1]:
            tpoFix[i,5] = 1
            tpoFix = [tpoFix[:i,:], tpoFix[i:,:]]
            tpoFix[i,1] = tpoBlink[contB,0]
            tpoFix[i+1,0] = tpoBlink[contB,1]
            
            # Actualiza el blink
            if contB < len(tpoBlink):
                contB = contB + 1
        i=i+1

    # modifica la matriz de fix
    tpoFix = tpoFix[((aux -1)**2).astype(bool),:]

    return tpoSac, tpoFix, tpoBlink, vel, acc




