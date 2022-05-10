#  Recibe una matriz de datos por columna correspondientes a:
#        [tiempo; posXOjo; posYOjo]

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

def SaFiDe_BlinkDetector(tpoSac, color_sacc='red',color_blink='blue',verbose=True):
    ## unión pseudo sacadas
    i=1
    tpo_Sac = np.copy(tpoSac)
    for c in range(1,len(tpo_Sac)):
        # une y borra la pseudo sacada
        if tpo_Sac[i,0]-tpo_Sac[i-1,1] <= 40:
            tpo_Sac[i-1,1] = tpo_Sac[i,1]
            tpo_Sac[i-1,2] = tpo_Sac[i,2] + tpo_Sac[i-1,2]
            tpo_Sac[i-1,3] = tpo_Sac[i,3] + tpo_Sac[i-1,3]
            tpo_Sac[i-1,6] = tpo_Sac[i,6]
            tpo_Sac[i-1,7] = tpo_Sac[i,7]
            tpo_Sac[i-1,8] = 1
            tpo_Sac = np.delete(tpo_Sac, i, 0)
            i = i-1
        i = i+1
    # kmeans
    kmeans = KMeans(n_clusters=2).fit(tpo_Sac[:,[2,3]])
    class_idx = kmeans.predict(tpo_Sac[:,[2,3]])
    cluster1 = kmeans.cluster_centers_[0]
    cluster2 = kmeans.cluster_centers_[1]
    sacc_cluster = np.argmin([np.linalg.norm(cluster1),np.linalg.norm(cluster2)])
    blink_cluster = np.argmax([np.linalg.norm(cluster1),np.linalg.norm(cluster2)])
    class_ = {}
    class_[sacc_cluster] = tpo_Sac[:,[2,3]][np.where(class_idx==sacc_cluster)] #saccades
    class_[blink_cluster] = tpo_Sac[:,[2,3]][np.where(class_idx==blink_cluster)] #blinks
    if verbose:
        plt.scatter(class_[sacc_cluster][:,0],class_[sacc_cluster][:,1],label='saccades',color=color_sacc,s=5)
        plt.scatter(class_[blink_cluster][:,0],class_[blink_cluster][:,1],label='blinks',color=color_blink,s=5)
        plt.xlabel('Amplitude [°]')
        plt.ylabel('Duration [ms]')
        plt.legend()
        plt.show()
    if blink_cluster==1: #blinks -> 1 | saccades-> 0: OK
        tpo_Sac[:,-1] = class_idx 
    elif blink_cluster==0: #blinks -> 0 | saccades-> 1: NOT OK
        tpo_Sac[:,-1] = (class_idx-1)**2 #blinks -> 1 | saccades-> 0:
    return tpo_Sac


