"""
This program plots the images with highest and lowest contrast along
the series for the following scenarios:
    1. Before reconstruction
    2. After WFE correction
    3. After WFE + jitter correction.
"""


import sys
sys.path.append('./functions')
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import datetime as dt
import numpy as np
import math_func2 as mf
import pd_functions_v22 as pdf
import plots_func2 as pf
from scipy.fftpack import fftshift, ifftshift, fft2, ifft2
import shift_func as sf
from tqdm import tqdm
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams["image.interpolation"] = 'none'

"""
Imports and plots the set of Zernike coefficients and
the wavefront map over the different subfields.
"""
crop=False #If True, it crops the image using x0, xf, y0 and yf as limits
cam=0 #0 or 1. Camera index
ind1=1 #First index of the series#
ind2=70 #70#15#10 #Last index of the series
wave=0 #From 0 to 10. Wavelength index
modul=0 #From 0 to 4. Modulation index
k_max=9 #Number of txt files employed to average the wavefront
low_f=0.2 #Cutoff of the Wiener filter
reg1=0.05 #0.02 #Regularization factor used when computing Q
reg2=1 #Regularization factor used when computing Q
cobs=32.4 #18.5 (MPS) 32.4 (Sunrise) #Diameter of central obscuration as a percentage of the aperture
Jmax=22# 16 or 22.Maximum index of the zernike polynomials
Jmin=4 #Minimum index of the zernike polynomials
magnif=2.47 #Magnification factor of TuMag
plate_scale= 0.0378 #Plate scale in arcseconds (arcsec/pixel)


#Region to be subframed for plotting purposes
x0=400#600 #Initial pixel of the subframe in X direction
xf=1200#1100 #Final pixel of the subframe in X direction
y0=200#300 #Initial pixel of the subframe in Y direction
yf=1000#800 #FInal pixel of the subframe in Y direction

#Path and name of the FITS file containing the focused and defocused images
fsigma='./Jitter estimations/sigma_D14-45403-50000_cam%g'%cam
fname='D14-45403-50000_cam%g'%cam#Data file
ext='.fits' #Format of the images to be opened (FITS)
dir_folder='./' #Path of the folder containing the FITS file
ffolder='Flight/Jitter/FS_1_2024_07_14'#'Flight/COMM_1_10_7_13_14' #Name of the folder containing th FITS file
pref='52502' #Prefilter employed
ext='.fits' #Extension of the data file
txtname='PD_14_7_13_07_cam_0_52502_ima_1'#'zeros' if no WFE correction
txtfolder=dir_folder +'txt/'+txtname #Path of the txt files
sigma=np.load(fsigma+'.npy')


#Compute the umber of frames per second based on the number of images
if (ind2-ind1)<30:
    fps=3
elif (ind2-ind1)<100:
    fps=5 
elif (ind2-ind1)>=100:
    fps=8

"""
Read image
"""
path=dir_folder+ffolder+'/'+fname #Path of the image
ima=pdf.read_crop_reorder(path,ext,cam,wave,modul,crop=crop,
                      crop_region=[x0,xf,y0,yf])


#Plot jitter vs contrast
contrast=np.zeros(ind2-ind1)
i=-1
for ind in range(ind1,ind2):
    i+=1
    contrast[i]=100*np.std(ima[:,:,ind])/np.mean(ima[:,:,ind])


ind_vec=np.arange(ind1,ind2)
fig,ax1=plt.subplots()
ax1.plot(ind_vec,contrast,marker='o',color='b')
ax1.set_ylabel(r'Contrast ($\%$)')
#plt.show()
plt.close()

#Select lowest and highest contrast images
ind_low=ind1+np.argmin(contrast)
ind_high=ind1+np.argmax(contrast)
ima=ima[:,:,(ind_low,ind_high)]
sigma=sigma[(ind_low,ind_high),:]
print(sigma)
print('Lowest contrast frame:',ind_low)
print('Highest contrast frame:',ind_high)



"""
Restoration with average Zernike coefficients 
"""
if txtname == 'zeros':
    a_aver=np.array([0,0,0,0])
else:
    a_aver=pdf.retrieve_aberr(k_max,Jmax,txtfolder)
a_d=0 #For pdf.object_estimate to restore a single image

#Image padding
ima_pad,pad_width=pdf.padding(ima)
cut=pad_width#To later select the non-padded region


#Plot radial MTFs

MTF_diff_radial,nuc=pdf.radial_MTF(1e-10*a_aver,0,0*sigma[0,:],plate_scale,
                         cobs=0,inst='tumag')
MTF_ideal_radial,nuc=pdf.radial_MTF(1e-10*a_aver,0,0*sigma[0,:],plate_scale,
                         cobs=cobs,inst='tumag')
MTF_jitter_radial,nuc=pdf.radial_MTF(0*a_aver,0,sigma[0,:],plate_scale,
                         cobs=cobs,inst='tumag')
MTF_wave_radial,nuc=pdf.radial_MTF(a_aver,0,0*sigma[0,:],plate_scale,
                         cobs=cobs,inst='tumag')
MTF_effective_radial,nuc=pdf.radial_MTF(a_aver,0,sigma[0,:],plate_scale,
                         cobs=cobs,inst='tumag')
radius=np.arange(MTF_effective_radial.shape[0])/nuc
plt.plot(radius,MTF_diff_radial,label='Ideal')
plt.plot(radius,MTF_ideal_radial,label='Diffraction')
plt.plot(radius,MTF_jitter_radial,label='Jitter')
plt.plot(radius,MTF_wave_radial,label='WFE')
plt.plot(radius,MTF_effective_radial,label='Effective')
plt.legend()
plt.xlabel(r'$\nu/\nu_c$')
plt.ylabel('MTF')
plt.xlim([0,1])
plt.show()
plt.close()

#Restore only WFE
ima_series=np.zeros((ima.shape[0],ima.shape[1],2))
contrast_rest=np.zeros(2)
for i in range(2):
    o_plot,_,_=pdf.object_estimate_jitter(ima_pad[:,:,i],
                    0*sigma[i,:],a_aver,a_d,cobs=cobs,low_f=low_f,
                    wind=True,reg1=reg1,reg2=reg2)
    ima_series[:,:,i]=o_plot[cut:-cut,cut:-cut] 
    contrast_rest[i]=100*np.std(ima_series[:,:,i])/np.mean(ima_series[:,:,i])


#Plots
cmap='gray'#'gist_heat'
vmin=np.min(ima_series[:,:,1])
vmax=np.max(ima_series[:,:,1])
fig,axs=plt.subplots(3,2)
fig2,axs2=plt.subplots(3,2)


# Function to remove tick labels from imshow plots
def remove_tick_labels(axs):
    for ax in axs.flatten():
        ax.set_xticklabels([])
        ax.set_yticklabels([])

#Plot only the sunspot
axs[0,0].imshow(ima[x0:xf,y0:yf,0],cmap=cmap,vmin=vmin,vmax=vmax)
axs[0,0].set_title('Lowest contrast')
axs[0,0].set_ylabel('Before restoration')
axs[1,0].imshow(ima_series[x0:xf,y0:yf,0],cmap=cmap,vmin=vmin,vmax=vmax)
axs[1,0].set_ylabel('WFE correction')
axs[0,1].imshow(ima[x0:xf,y0:yf,1],cmap=cmap,vmin=vmin,vmax=vmax)
axs[0,1].set_title('Highest contrast')
axs[1,1].imshow(ima_series[x0:xf,y0:yf,1],cmap=cmap,vmin=vmin,vmax=vmax)
remove_tick_labels(axs)

#Plot the full image
axs2[0,0].imshow(ima[:,:,0],cmap=cmap,vmin=vmin,vmax=vmax)
axs2[0,0].set_title('Lowest contrast')
axs2[0,0].set_ylabel('Before restoration')
axs2[1,0].imshow(ima_series[:,:,0],cmap=cmap,vmin=vmin,vmax=vmax)
axs2[1,0].set_ylabel('WFE correction')
axs2[0,1].imshow(ima[:,:,1],cmap=cmap,vmin=vmin,vmax=vmax)
axs2[0,1].set_title('Highest contrast')
axs2[1,1].imshow(ima_series[:,:,1],cmap=cmap,vmin=vmin,vmax=vmax)
remove_tick_labels(axs2)

#Add text labels with the contrast values
for i in range(2):
    for j in range(2):
         tx=axs2[i,j].text(100, 100, '', fontsize=15, va='top',color='white')
         if i==0 and j==0:
             cont1=contrast[ind_low]
         elif i==0 and j==1:
             cont1=contrast[ind_high]
         else:
            cont1=contrast_rest[j]
         tx.set_text('%g'%np.round(cont1,1)+r'$\,\%$')



"""
Restore WFE + jitter
"""
#Reconstruct images
for i in range(2):
    o_plot,_,_=pdf.object_estimate_jitter(ima_pad[:,:,i],
                    sigma[i,:],a_aver,a_d,cobs=cobs,low_f=low_f,
                    wind=True,reg1=reg1,reg2=reg2)
    ima_series[:,:,i]=o_plot[cut:-cut,cut:-cut] 
    contrast_rest[i]=100*np.std(ima_series[:,:,i])/np.mean(ima_series[:,:,i])

#Plot
axs[2,0].imshow(ima_series[x0:xf,y0:yf,0],cmap=cmap,vmin=vmin,vmax=vmax)
axs[2,1].imshow(ima_series[x0:xf,y0:yf,1],cmap=cmap,vmin=vmin,vmax=vmax)
axs[2,0].set_ylabel('WFE + jitter correction')

axs2[2,0].imshow(ima_series[:,:,0],cmap=cmap,vmin=vmin,vmax=vmax)
axs2[2,1].imshow(ima_series[:,:,1],cmap=cmap,vmin=vmin,vmax=vmax)
axs2[2,0].set_ylabel('WFE + jitter correction')

for j in range(2):
    tx=axs2[2,j].text(100, 100, '', fontsize=15, va='top',color='white')
    cont1=contrast_rest[j]
    tx.set_text('%g'%np.round(cont1,1)+r'$\,\%$')
plt.show()
