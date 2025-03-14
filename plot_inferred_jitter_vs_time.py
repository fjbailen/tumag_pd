"""
This function plots the jitterinferred for diffferent cadences
"""
from PIL import Image as im
import numpy as np
import sys
sys.path.append('./functions')
import pd_functions_v22 as pdf
np.set_printoptions(precision=4,suppress=True)
from matplotlib import pyplot as plt
import general_func as gf
from scipy.fftpack import fft2,ifft2
import shift_func as sf
from tqdm import tqdm
import plots_func2 as pf
plt.rcParams['figure.constrained_layout.use'] = True #For the layout to be as tight as possible



#Parameters of input data
ymax=0.0175#0.035 #Maximum limit for the plot
SNR=100 #Signal-to-noise ratio. 0 if no noise is to be applied
N=256 #Number of pixels (Cadence simulatin)
plate_scale=0.055 #Plate scale of the simulations in arcseconds (arcsec/pixel)
Ntime=40 #Number of seconds of the simulation (max: 40)
sigma=[0,0.05,0.10,0.15]
sigmax=sigma #RMS of jitter along X in arcsec
sigmay=sigma #RMS of jitter along Y in arcsec
Nacc=1000 #Number of accumulated images
aberr=np.zeros(4) #Aberrations of the instrument


"""
Load npy file and plot
"""
folder='./Flight/Jitter/Simulations'
color=['b','r','k','orange']
i=-1

fig,axs=plt.subplots()
for sigma_i in sigma:
    i+=1
    fname="jitter_error_vs_time_rms_%g_arcsec_Njit_%g_SNR_%g.npy"%(sigmax[i],Ntime,SNR)
    data=np.load(folder+'/'+fname)


    #Arrays for plot
    time=data[:,0] #Time in seconds
    rms_error=data[:,1] #Norm of the difference between the true and inferred sigma_x and sigma_y
    rms_error_std=data[:,2] #STD of the rms_error for Nrepeat


    #Change some rms_error values because ima_shift is not representative enough
    #  of jitter and the algorithm fails. Solution: accumulate more images (e.g. 1e4)
    if sigma_i==0.05:
        rms_error[17]=0.00429
        rms_error_std[17]=0.00043
        rms_error[22]=0.00658
        rms_error_std[22]=0.00030
    elif sigma_i==0.10:
        rms_error[38]=0.01574
        rms_error_std[38]=0.00057
    #Plot
    xlabel='Time (s)'
    ylabel=r'$\Delta\sigma$ [arcsec]'

    axs.scatter(time,rms_error,color=color[i],label=r'$\sigma_{x}=\sigma_{y}=%g$ arcsec'%sigmax[i])
    axs.fill_between(time, rms_error-rms_error_std, rms_error+rms_error_std,
                      color=color[i], alpha=0.2)
    axs.set_ylim([-0.0001,ymax])
    axs.set_xlabel(xlabel)
    axs.set_ylabel(ylabel)
    #axs.set_title(r"$\sigma=%g$"%sigmax[i])

plt.legend()
plt.show()



