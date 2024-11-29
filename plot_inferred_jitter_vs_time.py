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
ymax=0.035 #Maximum limit for the plot
SNR=100 #Signal-to-noise ratio. 0 if no noise is to be applied
N=256 #Number of pixels (Cadence simulatin)
plate_scale=0.055 #Plate scale of the simulations in arcseconds (arcsec/pixel)
Ntime=40 #Number of seconds of the simulation (max: 40)
sigma=0.15
sigmax=sigma #RMS of jitter along X in arcsec
sigmay=sigma #RMS of jitter along Y in arcsec
Nacc=1000 #Number of accumulated images
aberr=np.zeros(4) #Aberrations of the instrument




"""
Load npy file and plot
"""
folder='./Flight/Jitter/Simulations'
fname="jitter_error_vs_time_rms_%g_arcsec_Njit_%g_SNR_%g.npy"%(sigmax,Ntime,SNR)
data=np.load(folder+'/'+fname)


#Arrays for plot
time=data[:,0]
rms_error=data[:,1]

#Plot
xlabel='Time (s)'
ylabel=r'$\Delta\sigma$ [arcsec]'

fig,axs=plt.subplots()
axs.scatter(time,rms_error)
axs.set_ylim([0,ymax])
axs.set_xlabel(xlabel)
axs.set_ylabel(ylabel)
axs.set_title(r"$\sigma=%g$"%sigmax)
plt.show()



