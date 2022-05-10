import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF

def SaFiDe_thresholds(time, X, Y, sampling_rate = 1000, color='blue', pix2grad = 39.38):
    # PARTE 1: Calcula la velocidad y la aceleracion
    nsam = len(time)
    cal = np.zeros((nsam-2,5)) # [timestamp dist vel acc sacc?]
    DTgr = np.zeros(nsam)
    vel = np.zeros(nsam)
    for i in range(nsam-1):
        # distancia total recorrida en grados visuales
        DTgr[i+1] = np.sqrt((X[i+1] - X[i])**2 + (Y[i+1] - Y[i])**2)  / pix2grad 
        # velocidad en grados/segundos
        vel[i+1] = DTgr[i+1]/((time[i+1]-time[i])/sampling_rate) # *10^-3 ya que los timestamps están en milisegundos
    vel[1] = vel[3]
    acc = np.zeros((len(vel)))
    for i in range(len(vel)-2):
        # aceleracion en grados/seg^2
        acc[i+1] = (vel[i+1]-vel[i])/( (time[i+1]-time[i] )/sampling_rate) 
    # PARTE 2: Cálculo de umbrales
    ecdf_v = ECDF(vel)
    ecdf_a = ECDF(acc)

    [fv,xv] = ecdf_v.y[1:-1],ecdf_v.x[1:-1]
    [fa,xa] = ecdf_a.y[1:-1],ecdf_a.x[1:-1]
    fa = (fa-.5)*2
    #ths
    ths_vel = xv[np.argwhere(fv>=.85)[0]]
    thsa_inf = xa[np.argwhere(fa<=-.9)[-1]] 
    thsa_sup = xa[np.argwhere(fa>=.9)[0]] 
    ths_acc = np.array([thsa_inf, thsa_sup])

    # PARTE 3: Histogramas
    fig, axs = plt.subplots(2,1)
    fig.suptitle('Empirical (Kaplan-Meier) cumulative distribution function')
    sns.histplot(ax=axs[0],x =xv[np.argwhere(fv<=.9)].reshape(-1,),bins=50)
    axs[0].axvline(x=ths_vel, color='black', ls='--', label=f'vel threshold: {ths_vel.item():.1f}')
    axs[0].set_xlabel('Velocity [u.a./s]')
    axs[0].set_ylabel('Frequency')
    axs[0].legend()
    sns.histplot(ax=axs[1], x = xa[np.argwhere((fa<=.93) & (fa>=-.93) )].reshape(-1,),bins=50)
    axs[1].axvline(x=thsa_sup, color='black', ls='--', label=f'upper acc thres: {thsa_sup.item():.1f}') 
    axs[1].axvline(x=thsa_inf, color='black', ls='--', label=f'lower acc thres: {thsa_inf.item():.1f}') 
    axs[1].set_xlabel('Acceleration [u.a./s^2]')
    axs[1].set_ylabel('Frequency')
    axs[1].legend()
    plt.tight_layout()
    plt.show()

    return  ths_vel, thsa_inf, thsa_sup
 
