"""
This function saves noisy focused and defosused images
for parallel_subpatches
"""
from PIL import Image as im
import numpy as np
import os
import time
import sys
sys.path.append('./functions')
import pd_functions_v16_cut as pdf
import math_func2 as mf
import zernike as zk
import multiprocessing as mtp
np.set_printoptions(precision=4,suppress=True)
from matplotlib import pyplot as plt


#Parameters of input data
Jmax_ab=33 #33 or 21
NSR=1e-2
N0=512 #420 #Size of the input synthetic data to be subframed


"""
Defocus, induced aberrations and initial guess for PD
"""
#Defocus introduced by PD plate
a_d=(-1)*np.pi/(np.sqrt(3)) #Defocus coefficient [rads]
aberr_file='IMaX_aberr.txt'
IMaX_aberr=np.loadtxt(aberr_file)
a_aberr=IMaX_aberr[:(Jmax_ab-1),1] #Aberrations induced in the image [rads]
norm_aberr=np.linalg.norm(a_aberr)
def_lambda=round(a_d*np.sqrt(3)/np.pi,1)

"""
Read image
"""
file='../images/DatosFran/continuo'
output='./results/'
ext='.sav'
I=pdf.read_image(file,ext,N=N0)

R0=N0/pdf.N*pdf.R
RHO,THETA=pdf.sampling2(N=N0,R=R0)
ap=pdf.aperture(N0,R0)
of0=pdf.convPSF(I,a_aberr,0,RHO,THETA,ap,norm=True) #Focused image
od0=pdf.convPSF(I,a_aberr,a_d,RHO,THETA,ap,norm=True) #Defocused image

#Noisy images
of0=pdf.gaussian_noise(NSR,of0)
od0=pdf.gaussian_noise(NSR,od0)

#Save focused noisy image as FITS file
SNR=int(1/NSR)
exitname='_Jmax_%g_SNR_%g_defocus_%g_wave'%(Jmax_ab,SNR,def_lambda)
pdf.save_image(of0,'of'+exitname,folder='txt')
pdf.save_image(od0,'od'+exitname,folder='txt')


quit()

#Restoration of the image and comparison with original MHD image
o_plot,susf,noise_filt=pdf.object_estimate(of0,od0,a_aberr,a_d,wind=True)
cut=39

fig, axs=plt.subplots(1,2)
vmax=np.max(I)
vmin=np.min(I)
axs[0].imshow(I[cut:-cut,cut:-cut],vmin=vmin,vmax=vmax,cmap='gray')
axs[1].imshow(o_plot[cut:-cut,cut:-cut],vmin=vmin,vmax=vmax,cmap='gray')
plt.show()
plt.close()

plt.imshow((I-o_plot)[cut:-cut,cut:-cut],cmap='seismic')
#plt.imshow(noise_filt)
plt.colorbar()
plt.show()
plt.close()
