"""
This program checks the flats for the two
cameras along the spectral line.
"""   
import sys
sys.path.append('./functions')
import numpy as np
from matplotlib import colors, pyplot as plt
from astropy.io import fits
from glob import glob
from scipy.optimize import minimize
from scipy.ndimage import shift
from sklearn import linear_model
import pandas as pd
import pd_functions_v22 as pdf
plt.rcParams['figure.constrained_layout.use'] = True

# Read image
ext='.fits' #Format of the images to be opened (FITS)
dir_folder='./' #Path of the folder containing the FITS file
ffolder='Flight/COMM_1_10_7_13_14' #Name of the folder containing th FITS file
fname='ff_OM_2.02_D10-4340-5619'
path=dir_folder+ffolder+'/'+fname #Path of the image
I=fits.open(path+ext)
data = I[0].data

print('Data shape:',data.shape)
#Separate cameras
cam1=data[0,:]
cam2=data[1,:]

#Mean level for each camera and modulation
def mean_profiles(data):
    avI=np.mean(data[:,0,:,:],axis=(1,2))
    avQ=np.mean(data[:,1,:,:],axis=(1,2))
    avU=np.mean(data[:,2,:,:],axis=(1,2))
    avV=np.mean(data[:,3,:,:],axis=(1,2))
    return avI,avQ,avU,avV


Icam1,Qcam1,Ucam1,Vcam1=mean_profiles(cam1)
Icam2,Qcam2,Ucam2,Vcam2=mean_profiles(cam2)


#Plot of mean profiles
fig, axs = plt.subplots(2, 2)
axs[0,0].plot(Icam1,label='cam1')
axs[0,0].plot(Icam2,label='cam2')
axs[0,0].legend()
axs[0,1].plot(Qcam1)
axs[0,1].plot(Qcam2)
axs[1,0].plot(Ucam1)
axs[1,0].plot(Ucam2)
axs[1,1].plot(Vcam1)
axs[1,1].plot(Vcam2)
axs[0,0].set_ylabel('Mod 1')
axs[0,1].set_ylabel('Mod 2')
axs[1,0].set_ylabel('Mod 3')
axs[1,1].set_ylabel('Mod 4')
for i in range(2):
    axs[0,i].set_xlabel('Wavelength index')
    axs[1,i].set_xlabel('Wavelength index')
fig.suptitle('Mean signal')



#Normalization of cam 1 and 2 separately to correct the different signal levels
norm1=np.median(cam1[0,0,:,:])
norm2=np.median(cam2[0,0,:,:])
#cam1=cam1/norm1
#cam2=cam2/norm2


#Compute noise
Nx=cam1.shape[-1] #Number of pixels
Nwvl=cam1.shape[0]  #Number of wavelengths
wvl,fnum,Deltax=pdf.tumag_params(pref='52502')
nuc,_=pdf.compute_nuc(Nx,wvl,fnum,Deltax)
gamma=np.zeros((4,Nwvl))
noise1=np.zeros((4,Nwvl))
noise2=np.zeros((4,Nwvl))
for i in range(4):
    for j in range(Nwvl):
        noise1[i,j]=pdf.noise_power(cam1[j,i,:,:],nuc,Nx)
        noise2[i,j]=pdf.noise_power(cam2[j,i,:,:],nuc,Nx)
        gamma[i,j]=noise1[i,j]/noise2[i,j]

fig,axs=plt.subplots(2,2) #Noise level ratio
fig2,axs2=plt.subplots(2,2) #Noise level for each cam
fig3,axs3=plt.subplots(2,2) #Scatter plot
for i in range(4):
    axs2[i//2,i%2].plot(noise1[i,:],label='cam1')
    axs2[i//2,i%2].plot(noise2[i,:],label='cam2')
    axs[i//2,i%2].plot(gamma[i,:])
    axs[i//2,i%2].set_xlabel('Wavelength index')
    axs[i//2,i%2].set_ylabel(r'$\gamma_{%g}$'%i)
    axs2[i//2,i%2].set_xlabel('Wavelength index')
    axs2[i//2,i%2].set_ylabel(r'$\sigma_{%g}$'%i)
fig.suptitle('Noise level ratio between cameras')
fig2.suptitle('Noise level')
axs2[0,0].legend()


#Scatter plot
axs3[0,0].scatter(Icam1/norm1,gamma[0,:])
axs3[0,0].set_xlabel('Mod 1 signal')
axs3[0,0].set_ylabel(r'$\gamma_0$')
axs3[0,1].scatter(Qcam1/norm1,gamma[1,:])
axs3[0,1].set_xlabel('Mod 2 signal')
axs3[0,1].set_ylabel(r'$\gamma_1$')
axs3[1,0].scatter(Ucam1/norm1,gamma[2,:])
axs3[1,0].set_xlabel('Mod 3 signal')
axs3[1,0].set_ylabel(r'$\gamma_2$')
axs3[1,1].scatter(Vcam1/norm1,gamma[3,:])
axs3[1,1].set_xlabel('Mod 4 signal')
fig3.suptitle('Scatter plots')
plt.show()


quit()

#Plot the four modulations at the first wavelength
fig, axs = plt.subplots(2, 2)
im0 = axs[0,0].imshow(cam1[0,0,:,:]-cam2[0,0,:,:], cmap='gray')
fig.colorbar(im0, ax=axs[0,0])
im1 = axs[0,1].imshow(cam1[0,1,:,:]-cam2[0,1,:,:], cmap='gray')
fig.colorbar(im1, ax=axs[0,1])
im2 = axs[1,0].imshow(cam1[0,2,:,:]-cam2[0,2,:,:], cmap='gray')
fig.colorbar(im2, ax=axs[1,0])
im3 = axs[1,1].imshow(cam1[0,3,:,:]-cam2[0,3,:,:], cmap='gray')
fig.colorbar(im3, ax=axs[1,1])
plt.show()
