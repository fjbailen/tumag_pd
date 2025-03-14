"""
This program is a second version of infer_jitter_series.
It computes jitter along a series of images that
are already restored from their WFE. It finds
the image with the highest contrast and infers the jitter
for the previous and next ones. It uses the inferred jitter 
to correct the previous and next images to those and
repeats this process until the whole series is corrected
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
plt.rcParams['figure.constrained_layout.use'] = True


"""
Imports and plots the set of Zernike coefficients and
the wavefront map over the different subfields.
"""
wave=0 #From 0 to 10. Wavelength index
modul=0 #From 0 to 4. Modulation index
cam=0 #0 or 1. Camera index
ind1=0 #First index of the series#
ind2=73 #180 (full series),70 (35 min seres) 15 (short series) #Last index of the series
k_max=9 #Number of txt files employed to average the wavefront
low_f=0.2 #Cutoff of the Wiener filter
reg1=0.05 #0.02 #Regularization factor used when computing Q
reg2=1 #Regularization factor used when computing Q
cobs=32.4 #18.5 (MPS) 32.4 (Sunrise) #Diameter of central obscuration as a percentage of the aperture
Jmax=22# 16 or 22.Maximum index of the zernike polynomials
Jmin=4 #Minimum index of the zernike polynomials
magnif=2.47 #Magnification factor of TuMag
plate_scale= 0.0378 #Plate scale in arcseconds (arcsec/pixel)
fps=3 #Number of frames per second for the movie

#Region to be subframed
crop=False #If True, it crops the image using x0, xf, y0 and yf as limits
x0=200 #200 or 400 #Initial pixel of the subframe in X direction
xf=x0+1600 #900 or 1600 #Final pixel of the subframe in X direction
y0=x0 #Initial pixel of the subframe in Y direction
yf=xf  #FInal pixel of the subframe in Y direction

#Path and name of the FITS file containing the focused and defocused images
ext='.fits' #Format of the images to be opened (FITS)
dir_folder='./' #Path of the folder containing the FITS file
ffolder='Flight/QS8_11_7_9_05' #Name of the folder containing th FITS file
pref='517'#'52502' #Prefilter employed
fname='D11-30968-41605_cam%g_mod%g'%(cam,modul) #Data file
ext='.fits'#'.npy' #Extention of the data file
txtname='PD_11_7_11_00_cam_0_52502_ima_1'#'zeros'
txtfolder=dir_folder +'txt/'+txtname #Path of the txt files
moviename='Icont_jitter_'+fname+'.mp4'#Name of the movie to be saved




"""
Read image
"""
path=dir_folder+ffolder+'/'+fname #Path of the image
ima=pdf.read_crop_reorder(path,ext,cam,wave,0,crop=crop,
                      crop_region=[x0,xf,y0,yf])
print('Data shape:',ima.shape)


#Plot of the first image of the series
plt.imshow(ima[:,:,0],cmap='gray')
plt.colorbar()
plt.close()


"""
Find image with highest contrast
"""
contrast=np.zeros(ind2-ind1)
for i in range(ind1,ind2):
    contrast[i]=np.std(ima[:,:,i])/np.mean(ima[:,:,i])

i_max=np.argmax(contrast)
print('Image with highest contrast:',i_max)

#Plot of contrast along the series
plt.plot(100*contrast,marker='o')
plt.xlabel('Image index')
plt.ylabel('Contrast (%)')
plt.close()



#Colormap limits for other plots
vmin=np.min(ima[:,:,i_max])
vmax=np.max(ima[:,:,i_max])


"""
Restoration with average Zernike coefficients
"""
#a_aver=np.array([0,0,0,0])
a_aver=pdf.retrieve_aberr(k_max,Jmax,txtfolder)
a_d=0 #For pdf.object_estimate to restore a single image


#System parameters
N=ima.shape[0]
wvl,fnum,Delta_x=pdf.tumag_params(pref=pref)
nuc,R=pdf.compute_nuc(N,wvl,fnum,Delta_x)


#Image padding for restoring purposes
ima_pad,pad_width=pdf.padding(ima)
cut=pad_width#To later select the non-padded region


# Correct from jitter if several images of the series is analyzed
if ind2>ind1:
    sigma_vec=np.zeros((ind2-ind1,3))
    ima_series=ima.copy()
    ima_series[:,:,i_max]=ima[:,:,i_max]

    #Correct jitter from i_max to ind2 
    for i in range(i_max,ind2-1):
        print('-----------------')
        print('Image index %g'%i)
        print('-----------------')
        ima_jitt=ima_series[:,:,(i,i+1)]
           
    
        #Realign the chosen pair of images
        kappa=20
        F0=fft2(ima_jitt[:,:,0])
        print('Re-aligning pair of images (%g,%g)'%(i,i+1))  
        F_comp=fft2(ima_jitt[:,:,1])
        error,row_shift,col_shift,Gshift=sf.dftreg(F0,F_comp,kappa)
        deltax=int(np.round(row_shift))
        deltay=int(np.round(col_shift))
        ima_jitt[:,:,1]=np.roll(ima_jitt[:,:,1],(deltax,deltay),axis=(0,1))
        print('Delta x, Delta y:',row_shift,col_shift)    
        
    
        #Compute jitter 
        Ok,gamma,wind,susf=pdf.prepare_PD(ima_jitt,nuc,N)
        sigma0=sigma_vec[i,:]    
        sigma=pdf.minimization_jitter2(Ok,gamma,plate_scale,nuc,N,sigma0,
                                          cut=int(0.15*N))  
        sigma_vec[i+1,:]=sigma
        print('Sigma:',sigma)  

    #Correct jitter from i_max to ind1    
    for i in range(i_max,ind1,-1):
        print('-----------------')
        print('Image index %g'%i)
        print('-----------------')
        ima_jitt=ima_series[:,:,(i,i-1)]
           
        #Realign the chosen pair of images
        kappa=20
        F0=fft2(ima_jitt[:,:,0])
        print('Re-aligning pair of images (%g,%g)'%(i,i-1))  
        F_comp=fft2(ima_jitt[:,:,1])
        error,row_shift,col_shift,Gshift=sf.dftreg(F0,F_comp,kappa)
        deltax=int(np.round(row_shift))
        deltay=int(np.round(col_shift))
        ima_jitt[:,:,1]=np.roll(ima_jitt[:,:,1],(deltax,deltay),axis=(0,1))
        print('Delta x, Delta y:',row_shift,col_shift)    
        
    
        #Compute jitter 
        Ok,gamma,wind,susf=pdf.prepare_PD(ima_jitt,nuc,N)

        sigma0=sigma_vec[i,:]    
        sigma=pdf.minimization_jitter2(Ok,gamma,plate_scale,nuc,N,sigma0,
                                          cut=int(0.15*N))   
        sigma_vec[i-1,:]=sigma
        print('Sigma0:',sigma0)
        print('Sigma:',sigma)  


print(np.round(sigma_vec,3))
np.save('sigma_'+fname+'.npy',sigma_vec)

#Restore images from jitter and aberrations
for i in range(ind1,ind2): 
    o_plot,_,noise_filt=pdf.object_estimate_jitter(ima_pad[:,:,i],
                    sigma_vec[i,:],a_aver,a_d,cobs=cobs,low_f=low_f,wind=True,reg1=reg1,reg2=reg2)
    ima_series[:,:,i]=o_plot[cut:-cut,cut:-cut]

#Restore images only from aberrations
ima_wfe=0*ima_series
for i in range(ind1,ind2): 
    o_plot2,_,noise_filt2=pdf.object_estimate_jitter(ima_pad[:,:,i],
                    0*sigma_vec[i,:],a_aver,a_d,cobs=cobs,low_f=low_f,wind=True,reg1=reg1,reg2=reg2)
    ima_wfe[:,:,i]=o_plot2[cut:-cut,cut:-cut]

            
#Save movies
pf.movie3(ima_wfe,ima_series,moviename,axis=2,fps=fps,
            title=['WFE','WFE+jitter'])

pf.movie13(ima,ima_wfe,ima_series,'original_vs_wfe_jitter_corrected_mod%g.mp4'%modul,
           axis=2,fps=fps,title=['Original','WFE','WFE+jitter'])