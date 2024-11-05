"""
This function read an MHD simulation and simulates the presence of jitter
of a given amplitude.
"""
from PIL import Image as im
import numpy as np
import sys
sys.path.append('./functions')
import pd_functions_v22 as pdf
np.set_printoptions(precision=4,suppress=True)
from matplotlib import pyplot as plt
import general_func as gf
from scipy.fftpack import fftshift


#Parameters of input data
SNR=0
N0=200 #Number of pixels of the FOV of the image
sigmax=0.5 #RMS of jitter along X direction (arcsec)
sigmay=0.5 #RMS of jitter along X direction (arcsec)



"""
Read image
"""
file='./continuo'
output='./results/'
ext='.sav'
ima=pdf.read_image(file,ext)

"""
Simulate the image affected by jitter with sigmax and sigmay
"""
#Apply subpixel shift folowing a normal distribution with sigmax and sigmay
IMA=fftshift(ima)


quit()

R0=N0/pdf.N*pdf.R
RHO,THETA=pdf.sampling2(N=N0,R=R0)
ap=pdf.aperture(N0,R0)

#Stray-light phase error term
sl=pdf.sl_noise(sigma,RHO,THETA,ap,Jmax=Jmax_ab) #Stray light phase error term
np.save('./fits/sl_noise_sigma_%g_SNR_%g'%(sigma,SNR), sl)


#Name of FITs image to be saved
exitname='./fits/Jmax_%g_SNR_%g_K_%g.fits'%(Jmax_ab,SNR,K)

#Array where the focused and defocused images will be stored
matrix_ofod=np.zeros((I.shape[0],I.shape[1],K))
of0=pdf.convPSF(I,a_aberr,0,RHO,THETA,ap,sl=sl,norm=True) #Focused image
if SNR>0:
    NSR=1/SNR
    of0=pdf.gaussian_noise(NSR,of0)
matrix_ofod[:,:,0]=of0



for i in range(1,K):
    od0=pdf.convPSF(I,a_aberr,a_d[i],RHO,THETA,ap,sl=sl,
                    norm=True)#Defocused image
    if SNR>0:
        od0=pdf.gaussian_noise(NSR,od0)

    #Save images in FITS file
    matrix_ofod[:,:,i]=od0

fig,axs=plt.subplots(1,5)

for i in range(5):
    axs[i].imshow(matrix_ofod[:,:,i],cmap='gray')

gf.save_fits2(matrix_ofod,a_d,exitname)

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
