"""
This program restores a series of jittered images with 
the jitter inferred in "infer_jitter_series.py".
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


"""
Imports and plots the set of Zernike coefficients and
the wavefront map over the different subfields.
"""
#File names and paths
fsigma='./Jitter estimations/sigma_D11-30968-41605_cam0'
fname='D11-30968-41605_cam0_mod0'#Data file
txtname='PD_11_7_11_00_cam_0_52502_ima_1'#'zeros' if no WFE correction
ffolder='Flight/QS8_11_7_9_05'# #Name of the folder containing th FITS file

#Settings
pref='52502' #Prefilter employed
region_contrast='full' #Region to compute the contrast. 'full' or 'corner'.
wfe_corrected_comparison=True #Restore the WFE of the jittered image?
ind1=0 #First index of the series#
ind2=73 #70#15#10 #Last index of the series
k_max=9 #Number of txt files employed to average the wavefront
low_f=0.2 #Cutoff of the Wiener filter
reg1=0.05 #0.02 #Regularization factor used when computing Q
reg2=1 #Regularization factor used when computing Q
cobs=32.4 #18.5 (MPS) 32.4 (Sunrise) #Diameter of central obscuration as a percentage of the aperture
Jmax=22# 16 or 22.Maximum index of the zernike polynomials
Jmin=4 #Minimum index of the zernike polynomials
magnif=2.47 #Magnification factor of TuMag
plate_scale= 0.0378 #Plate scale in arcseconds (arcsec/pixel)


#Region to be subframed
crop=False #If True, it crops the image using x0, xf, y0 and yf as limits
x0=200 #200 or 400 #Initial pixel of the subframe in X direction
xf=x0+1600 #900 or 1600 #Final pixel of the subframe in X direction
y0=x0 #Initial pixel of the subframe in Y direction
yf=xf  #FInal pixel of the subframe in Y direction


#Path and name of the FITS file containing the focused and defocused images
ext='.fits' #Format of the images to be opened (FITS)
dir_folder='./' #Path of the folder containing the FITS file
ext='.fits' #Extention of the data file
txtfolder=dir_folder +'txt/'+txtname #Path of the txt files
sigma=np.load(fsigma+'.npy')
cam=0 #0 or 1. Camera index
wave=0 #From 0 to 10. Wavelength index
modul=0 #From 0 to 4. Modulation index

#Compute the number of frames per second based on the number of images
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
ima=pdf.read_crop_reorder(path,ext,cam,wave,modul,crop=False,
                      crop_region=[x0,xf,y0,yf])


#Plot jitter vs contrast for the original image
contrast=pf.contrast_along_series(ima,ind1,ind2,region_contrast)
sigma_rms=np.sqrt(sigma[ind1:ind2,0]**2+sigma[ind1:ind2,1]**2)
mean_contrast1=np.round(np.mean(contrast),3)
std_contrast1=np.round(np.std(contrast),3)    
print('Mean contrast for original series:',mean_contrast1)
print('STD contrast for original series:',std_contrast1)


ind_vec=np.arange(ind1,ind2)
fig,ax1=plt.subplots()
plot1,=ax1.plot(ind_vec,contrast,marker='o',color='b',linestyle='solid')
ax1.set_ylabel(r'Contrast ($\%$)')
ax2 = ax1.twinx()  
plot2,=ax2.plot(ind_vec,sigma_rms,marker='o',color='r',linestyle='solid')
ax2.set_ylabel(r'$\sigma$ (arcsec)')
ax1.set_xlabel('Frame index')
ax2.legend([plot1, plot2], ['Contrast', r'$\sigma$'],loc='upper left')
#fig.legend(loc='upper left')
plt.savefig(fname+'_sigma_and_contrast_vs_index.png')
#plt.savefig(fname+'_sigma_and_contrast_vs_index.eps')
plt.close()


fig,axs=plt.subplots()
axs.plot(contrast,sigma_rms,'o')
axs.set_xlabel(r'Contrast $(\%)$')
axs.set_ylabel(r'$\sigma$ (arcsec)')
axs.set_ylim([0.9*np.sort(sigma_rms)[1],1.05*np.max(sigma_rms)])
plt.savefig(fname+'_sigma_vs_contrast.png')
#plt.savefig(fname+'_sigma_vs_contrast.eps')
plt.close()


"""
Restoration with average Zernike coefficients if only one image is selected
"""
if txtname == 'zeros':
    a_aver=np.array([0,0,0,0])
else:
    a_aver=pdf.retrieve_aberr(k_max,Jmax,txtfolder)
a_d=0 #For pdf.object_estimate to restore a single image

#Image padding
ima_pad,pad_width=pdf.padding(ima)
cut=pad_width#To later select the non-padded region

   
"""
Correct from jitter if several images of the series is analyzed
"""
if ind2>ind1:
    ima_series=np.zeros((ima.shape[0],ima.shape[1],ind2-ind1))
    flag=np.zeros(ind2-ind1) #Zero if not corrected, one if already corrected
    j=-1
    for i in tqdm(range(ind1,ind2)):
        j+=1

        # Restore the image
        o_plot,_,noise_filt=pdf.object_estimate_jitter(ima_pad[:,:,i],
                sigma[i,:],a_aver,a_d,cobs=cobs,low_f=low_f,wind=True,
                reg1=reg1,reg2=reg2)
        ima_series[:,:,j]=o_plot[cut:-cut,cut:-cut] 

        if wfe_corrected_comparison is True:
            #Restore the image without jitter correction
            o_plot,_,noise_filt=pdf.object_estimate_jitter(ima_pad[:,:,i],
                    0*sigma[i,:],a_aver,a_d,cobs=cobs,low_f=low_f,wind=True,
                    reg1=reg1,reg2=reg2)
            ima[:,:,i]=o_plot[cut:-cut,cut:-cut] 
            contrast=pf.contrast_along_series(ima,ind1,ind2,region_contrast)
      
        if j==0:
            vmin=np.min(ima_series[:,:,j])
            vmax=np.max(ima_series[:,:,j])
        
        #Plot first pair of images before and after jitter correction
        if j==0:
            fig,axs=plt.subplots(1,3)
            axs[0].imshow(ima_pad[cut:-cut,cut:-cut,i],cmap='gray',vmin=vmin,vmax=vmax)
            axs[0].set_title('Original')
            axs[1].imshow(ima[:,:,ind1],cmap='gray',vmin=vmin,vmax=vmax)
            axs[1].set_title('WFE corrected')
            axs[2].imshow(ima_series[:,:,0],cmap='gray',vmin=vmin,vmax=vmax)
            axs[2].set_title('WFE + jitter corrected')
            pf.remove_tick_labels(axs)
            #plt.show()
            plt.close()


      
    
    #Compute contrast for restored image and plot contrasts
    contrast2=pf.contrast_along_series(ima_series,0,ind2-ind1,region_contrast)
    mean_contrast2=np.round(np.mean(contrast2),3)
    std_contrast2=np.round(np.std(contrast2),3)    
    print('Mean contrast for original series:',mean_contrast2)
    print('STD contrast for original series:',std_contrast2)
    fig,ax1=plt.subplots()
    ax1.plot(ind_vec,contrast,marker='o',color='b',label='WFE corrected')
    ax1.plot(ind_vec,contrast2,marker='o',color='r',label='WFE+jitter corrected')
    ax1.set_ylabel(r'Contrast ($\%$)')
    ax1.set_xlabel(r'Frame #')
    ax1.legend(loc='upper right')
    plt.savefig(fname+'_contrast_vs_index.png')
    plt.close()
  


    pf.movie13(ima_pad[cut:-cut,cut:-cut,ind1:ind2],ima[:,:,ind1:ind2],
                ima_series,'Movie_paper_'+fname+'.mp4',
                axis=2,fps=fps,title=['Original','WFE corrected',
                                      'WFE + jitter corrected'],
                contrast=region_contrast)
    #pf.movie3(ima[:,:,ind1:ind2],ima_series,'Icont_comparison_'+fname+'.mp4',
    #          axis=2,fps=fps,title=['Jittered','Jitter free'],
    #          contrast=region_contrast)


#Print jitter
#np.save(fname+'_wfe_corrected.npy',ima_series)
print('Sigma:')
print(np.round(sigma,3))

