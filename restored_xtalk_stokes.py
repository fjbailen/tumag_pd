"""
This program infers the residual crosstalk from I to Q, U and V
after demodulation of the intensity images and corrects for it
using the method described in Jaeggli et al. 2022.
(https://doi.org/10.3847/1538-4357/ac6506)

https://github.com/sajaeggli/adhoc_xtalk/blob/main/Jaeggli_etal_2022ApJ_AdHoc_Xtalk.ipynb

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
import jaeggli as jg
import pd_functions_v22 as pdf
plt.rcParams['figure.constrained_layout.use'] = True



#Parameters
wvli=-1 #Wavelength index
wvlc=0 #Continuum wavelength index

# Read image
ext='.fits' #Format of the images to be opened (FITS)
dir_folder='./' #Path of the folder containing the FITS file
ffolder='Flight/COMM_1_10_7_13_14' #Name of the folder containing th FITS file
fname='stokes_COMM1_517_demod_ffielded_avgflat'
fnamerest='xtalk_corrected_stokes_COMM1_517_demod_ffielded_avgflat.npy'
#fnamerest='xtalk_corrected_stokes_COMM1_517_demod_ffielded_avgflat_all_wvls.npy'
path=dir_folder+ffolder+'/'+fname #Path of the image
data=pdf.read_image(path,ext,norma='no')
datarest=np.load(fnamerest)



# Reorder axis to convert into dimensions: [x,y,wavelength,stokes]
data=np.moveaxis(data,0,-1)
data=np.moveaxis(data,0,-1)
Nwvl=data.shape[2]

if datarest.shape[1]==4:
    datarest=np.moveaxis(datarest,0,-1)
    datarest=np.moveaxis(datarest,0,-1)


# Normalization of data
norm=np.median(data[:,:,wvlc,0])
data=data/norm

# Crop data to avoid edge effects arising from alignment/rotation
data=data[200:1200,200:1200,:,:]
datarest=datarest[200:1200,200:1200,:,:]

# Increase the length of the dimension 2 in datarest to match that of data
if datarest.shape[2] < Nwvl:
    pad_width = Nwvl- datarest.shape[2]
    datarest = np.pad(datarest, ((0, 0), (0, 0), (0, pad_width), (0, 0)), mode='constant', constant_values=0)




"""
Estimate the residual crosstalk from I to Q, U and V
"""
#Classical crosstalk correction of the original data
ItoQ=np.mean(data[:,:,wvli,1]/data[:,:,wvli,0])
ItoU=np.mean(data[:,:,wvli,2]/data[:,:,wvli,0])
ItoV=np.mean(data[:,:,wvli,3]/data[:,:,wvli,0])
print('-----------------')
print('ORIGINAL DATA')
print('-----------------')
print('Classical crosstalk:')
print('I to Q:',ItoQ)
print('I to U:',ItoU)
print('I to V:',ItoV)

#Linear fitting for the crosstalk correction of the restored data
linearQ=np.polyfit(data[:,:,wvli,0].flatten(),data[:,:,wvli,1].flatten(),1)
linearU=np.polyfit(data[:,:,wvli,0].flatten(),data[:,:,wvli,2].flatten(),1)
linearV=np.polyfit(data[:,:,wvli,0].flatten(),data[:,:,wvli,3].flatten(),1)

print('Linear fit for I to Q:',linearQ)
print('Linear fit for I to U:',linearU)
print('Linear fit for I to V:',linearV)


#Linear fitting for the crosstalk correction of the restored data
linearQrest=np.polyfit(datarest[:,:,wvli,0].flatten(),datarest[:,:,wvli,1].flatten(),1)
linearUrest=np.polyfit(datarest[:,:,wvli,0].flatten(),datarest[:,:,wvli,2].flatten(),1)
linearVrest=np.polyfit(datarest[:,:,wvli,0].flatten(),datarest[:,:,wvli,3].flatten(),1)

print('-----------------')
print('RESTORED DATA')
print('-----------------')
print('Linear fit for I to Q:',linearQrest)
print('Linear fit for I to U:',linearUrest)
print('Linear fit for I to V:',linearVrest)


"""
Correct residual crosstalk not corrected by the Mueller matrix using the linear fitting
"""
#Correct residual crosstalk from I to Q
Qcorr=datarest[:,:,wvli,1]-linearQrest[0]*datarest[:,:,wvli,0]-linearQ[1]


print('Mean value of Q before cross-talk correction:',np.mean(datarest[:,:,wvli,1]))
print('Mean value of Q after cross-talk correction:',np.mean(Qcorr))
print('RMS value of Q before cross-talk correction:',np.std(datarest[:,:,wvli,1]))
print('RMS value of Q after cross-talk correction:',np.std(Qcorr))

fig,axs=plt.subplots(1,2,figsize=(10,5))
im2=axs[1].imshow(Qcorr,cmap='seismic',vmin=-0.01,vmax=0.01)
fig.colorbar(im2,ax=axs[1])
axs[1].set_title('Cross-talk corrected Q')
im1=axs[0].imshow(datarest[:,:,wvli,1],cmap='seismic',vmin=-0.01,vmax=0.01)
fig.colorbar(im1,ax=axs[0])
axs[0].set_title('Restored Q')
fig.suptitle('Correct residual crosstalk from I to Q')



"""
Correct further (2nd order) residual crosstalk from V to Q
"""
linearVQ=np.polyfit(datarest[:,:,wvli,3].flatten(),Qcorr.flatten(),1)
Qcorr2=Qcorr-linearVQ[0]*datarest[:,:,wvli,3]-linearVQ[1]

fig,axs=plt.subplots(1,2,figsize=(10,5))
im2=axs[1].imshow(Qcorr2,cmap='seismic',vmin=-0.01,vmax=0.01)
fig.colorbar(im2,ax=axs[1])
axs[1].set_title('V-to-Q corrected ')
im1=axs[0].imshow(Qcorr,cmap='seismic',vmin=-0.01,vmax=0.01)
fig.colorbar(im1,ax=axs[0])
axs[0].set_title('I-to-Q Q')
fig.suptitle('Correct residual crosstalk from V to Q')



"""
Average Stokes along wavelength befor and after restoration with the fitted
Mueller matrix
"""
#Average original Stokes along wavelength
avI=np.mean(data[:,:,:,0],axis=(0,1))
avQ=np.mean(data[:,:,:,1],axis=(0,1))
avU=np.mean(data[:,:,:,2],axis=(0,1))
avV=np.mean(data[:,:,:,3],axis=(0,1))

#Average restored Stokes along wavelength
avIrest=np.mean(datarest[:,:,:,0],axis=(0,1))
avQrest=np.mean(datarest[:,:,:,1],axis=(0,1))
avUrest=np.mean(datarest[:,:,:,2],axis=(0,1))
avVrest=np.mean(datarest[:,:,:,3],axis=(0,1))



"""
Plots 
"""
#Plots of the original and restored data
fig, axs = plt.subplots(2, 2)
Imin=np.min([np.min(data[:,:,wvli,0]),np.min(datarest[:,:,wvli,0])])
Imax=np.max([np.max(data[:,:,wvli,0]),np.max(datarest[:,:,wvli,0])])
im0 = axs[0,0].imshow(data[:,:,wvli,0], cmap='gray',
                       vmin=Imin, vmax=Imax)
axs[0,0].set_title(r'Stokes $I$')
fig.colorbar(im0, ax=axs[0,0])
im1 = axs[0,1].imshow(data[:,:,wvli,1], cmap='seismic',
                       vmin=-0.01,vmax=0.01)
axs[0,1].set_title(r'Stokes $Q$')
fig.colorbar(im1, ax=axs[0,1])
im2 = axs[1,0].imshow(data[:,:,wvli,2], cmap='seismic',
                       vmin=-0.01,vmax=0.01)
axs[1,0].set_title(r'Stokes $U$')
fig.colorbar(im2, ax=axs[1,0])
im3 = axs[1,1].imshow(data[:,:,wvli,3], cmap='seismic',
                       vmin=-0.005,vmax=0.005)
axs[1,1].set_title(r'Stokes $V$')
fig.colorbar(im3, ax=axs[1,1])


fig, axs = plt.subplots(2, 2)
im0 = axs[0,0].imshow(datarest[:,:,wvli,0], cmap='gray',
                       vmin=Imin, vmax=Imax)
fig.colorbar(im0, ax=axs[0,0])
im1 = axs[0,1].imshow(datarest[:,:,wvli,1], cmap='seismic',
                       vmin=-0.01, vmax=0.01)
fig.colorbar(im1, ax=axs[0,1])
im2 = axs[1,0].imshow(datarest[:,:,wvli,2], cmap='seismic',
                       vmin=-0.01, vmax=0.01)
fig.colorbar(im2, ax=axs[1,0])
im3 = axs[1,1].imshow(datarest[:,:,wvli,3], cmap='seismic', 
                      vmin=-0.005, vmax=0.005)
fig.colorbar(im3, ax=axs[1,1])
axs[0,0].set_title(r'Stokes $I$')
axs[0,1].set_title(r'Stokes $Q$')
axs[1,0].set_title(r'Stokes $U$')
axs[1,1].set_title(r'Stokes $V$')
fig.suptitle('Restored data')


#Scatter plots for the original data
polyQ=np.poly1d(linearQ)
Qfitted=polyQ(data[:,:,wvli,0])

polyQrest=np.poly1d(linearQrest)
Qfittedrest=polyQrest(datarest[:,:,wvli,0])

fig,axs=plt.subplots(1,4,figsize=(15,5))
axs[0].scatter(data[:,:,wvli,0].flatten(),data[:,:,wvli,1].flatten(),s=0.1)
axs[0].scatter(data[:,:,wvli,0].flatten(),Qfitted,s=0.1)
axs[1].scatter(data[:,:,wvli,0].flatten(),data[:,:,wvli,2].flatten(),s=0.1)
axs[2].scatter(data[:,:,wvli,0].flatten(),data[:,:,wvli,3].flatten(),s=0.1)
axs[3].scatter(data[:,:,wvli,1].flatten(),data[:,:,wvli,3].flatten(),s=0.1)
axs[0].set_ylabel(r'$Q$')
axs[1].set_ylabel(r'$U$')
axs[2].set_ylabel(r'$V$')
axs[3].set_ylabel(r'$V$')
axs[3].set_xlabel(r'$Q$')
for i in range(3):
    axs[i].set_xlabel(r'$I$')
fig.suptitle('Original data')


#Scatter plots for the restored data
fig,axs=plt.subplots(1,4,figsize=(15,5))
axs[0].scatter(datarest[:,:,wvli,0].flatten(),datarest[:,:,wvli,1].flatten(),s=0.1)
axs[0].scatter(datarest[:,:,wvli,0].flatten(),Qfittedrest,s=0.1)
axs[1].scatter(datarest[:,:,wvli,0].flatten(),datarest[:,:,wvli,2].flatten(),s=0.1)
axs[2].scatter(datarest[:,:,wvli,0].flatten(),datarest[:,:,wvli,3].flatten(),s=0.1)
axs[3].scatter(datarest[:,:,wvli,1].flatten(),datarest[:,:,wvli,3].flatten(),s=0.1)
axs[0].set_ylabel(r'$Q$')
axs[1].set_ylabel(r'$U$')
axs[2].set_ylabel(r'$V$')
axs[3].set_ylabel(r'$V$')
axs[3].set_xlabel(r'$Q$')
for i in range(3):
    axs[i].set_xlabel(r'$I$')
fig.suptitle('Restored data')



#Scatter plots of restored Q at wvli vs I at different wavelengths
def sort_ax_indices(i,Nwvl):
    if i%2==0:
        k=i//2
    else:    
        k=i//2+Nwvl//2
    return k



fig,axs=plt.subplots(2,Nwvl//2,figsize=(15,5))
for i in range(Nwvl):
    k=sort_ax_indices(i,Nwvl)
    if Nwvl%2==1 and k==9:
        break
    print(i%2,i//2,k)
    im=axs[i%2,i//2].scatter(datarest[:,:,k,0].flatten(),datarest[:,:,wvli,1].flatten(),s=0.1)
    axs[i%2,i//2].set_xlabel(r'$I_%g$'%k)
    axs[i%2,i//2].set_ylabel(r'$Q$')
fig.suptitle('Q vs I at different wavelengths')


#Maps for the restored Stokes Q against wavelength
fig,axs=plt.subplots(2,Nwvl//2,figsize=(15,5))
for i in range(Nwvl):
    k=sort_ax_indices(i,Nwvl)
    if Nwvl%2==1 and k==9:
        break
    im=axs[i%2,i//2].imshow(datarest[:,:,k,1],cmap='seismic',vmin=-0.01,vmax=0.01)
    axs[i%2,i//2].set_title(r'$Q_%g$'%k)
    fig.colorbar(im,ax=axs[i%2,i//2])
fig.suptitle('Restored Stokes Q')


#Maps for the restored Stokes V against wavelength
fig,axs=plt.subplots(2,Nwvl//2,figsize=(15,5))
for i in range(Nwvl):
    k=sort_ax_indices(i,Nwvl)
    if Nwvl%2==1 and k==9:
        break
    im=axs[i%2,i//2].imshow(datarest[:,:,k,3],cmap='seismic',vmin=-0.01,vmax=0.01)
    axs[i%2,i//2].set_title(r'$V_%g$'%k)
    fig.colorbar(im,ax=axs[i%2,i//2])
fig.suptitle('Restored Stokes V')


#Stokes profiles along wavelength
fig,axs=plt.subplots(2,2)
axs[0,0].plot(avI,label='Original')
axs[0,0].plot(avIrest,label='Restored')
axs[0,0].legend()
axs[0,0].set_ylabel(r'$I$')
axs[0,1].plot(avQ)
axs[0,1].plot(avQrest)
axs[0,1].set_ylabel(r'$Q$')
axs[1,0].plot(avU)
axs[1,0].plot(avUrest)
axs[1,0].set_ylabel(r'$U$')
axs[1,1].plot(avV)
axs[1,1].plot(avVrest)
axs[1,1].set_ylabel(r'$V$')
for i in range(2):
    axs[0,i].set_xlabel('Wavelength index')
    axs[1,i].set_xlabel('Wavelength index')



plt.show()
quit()


