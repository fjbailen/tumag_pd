"""
This function reads an MHD simulation and infers the jitter
for different rms values of the induced jitter.
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
Nsigma=1000
SNR=0
file="true_jitter_vs_inferred_Njit_%g_SNR_%g.npy"%(Nsigma,SNR)
rms=np.load("true_jitter_vs_inferred_Njit_%g_SNR_%g.npy"%(Nsigma,SNR))
rms_true=rms[:,0]
rms_inferred=rms[:,1]

#Plot
xlabel=r'$\sigma_{\rm true}$ [arcsec]'
ylabel=r'$\sigma_{\rm inferred}$ [arcsec]'
pf.plot_scatter_density(rms_true,rms_inferred,xlabel,ylabel)



