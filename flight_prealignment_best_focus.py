"""
This program re-alignes with pixel accuracy the series of focused
and defocused images for one camera (cam1 or cam2)

Independent re-alignment of the series of focused and defocused images with
+/- 0.5 pixel accuracy (to avoid subpixel interpolation) is carried out.

The ROI of the images must be larger than the maximum shift. The algorithms
assumes also that the illuminated area is smaller than the ROI of the image
(i.e., the image is contained within a smaller square region)

"""
import numpy as np
import os, sys
sys.path.append('./functions')
import shift_func as sf
import general_func as gf
import utils_flight as ut
from matplotlib import pyplot as plt
from scipy.fftpack import fft2

"""
Input parameters
"""


"""
WARNING: 'cam' PARAMETER NOT EMPLOYED TO SELECT IMAGES FROM ONE CAM1 OR CAM2
"""


#Filter_pos is obtained after running prealignment_series once
Nseries=1 #Number of images recorded for the focused and defocused position
cam=1 #Cam 1 or 2
dir='./Flight/Flare'
dir_focus='PD_calibration/227'
dir_defocus='PD_calibration/226'
dir_darks='Darks' #Folder containing dark images
dir_flats='Flats'
N_acc_dark=50 #Number of accumulations of dark
N_acc_flat=16 #Number of accumulations of flat
pol_state=0 #0,1,2,3Polarization state of LCVRs
dimx=2016#2048 #Dimensions of the images to be read
dimy=dimx
xmin=200 #Minimum X coordinate when saving the image
xmax=1800 #Maximum X coordinate when saving the image


#Name of the saved FITs file with the aligned focused and defocused image
fname='cam_%g_focus'%(cam)+'.fits'


"""
Search for the images
"""
nm_foc = dir+'/'+dir_focus
nm_def = dir+'/'+dir_defocus

#List of images
focus_path=os.listdir(dir+'/'+dir_focus)
defocus_path=os.listdir(dir+'/'+dir_defocus)
darks_path=os.listdir(dir+'/'+dir_darks)
flats_path=os.listdir(dir+'/'+dir_flats)



#Initialization of arrays
of0=np.zeros((dimx,dimy,Nseries))
od0=np.zeros((dimx,dimy,Nseries))
darks_ima=np.zeros((dimx,dimy,Nseries))
flats_ima=np.zeros((dimx,dimy,Nseries))



#Open the images
for i in range(Nseries):
    of0[:,:,i],header1=ut.read_Tumag(dir+'/'+dir_focus+'/'+focus_path[i])
    #print(header)
    od0[:,:,i],header2=ut.read_Tumag(dir+'/'+dir_defocus+'/'+defocus_path[i])
    darks_ima[:,:,i],header3=ut.read_Tumag(dir+'/'+dir_darks+'/'+darks_path[i])


    if dir_flats=='':
        print('Flat fielding not carried out')
    else:
        try:
            flats_ima[:,:,i],header4=ut.read_Tumag(dir+'/'+dir_flats+'/'+flats_path[i])
        except:
            print('Correct flat  was not imported. We use a "fake" flat')
            flats_ima[:,:,i]=np.load(dir+'/'+dir_flats+'/'+'flat_cam1_smooth.npy')
    
  
    
    #To combine images from both cameras
    #if i % 2 !=0:
    #    of0[:,:,i]=np.flip(of0[:,:,i],axis=1)
    #    od0[:,:,i]=np.flip(od0[:,:,i],axis=1)
    #    darks_ima[:,:,i]=np.flip(darks_ima[:,:,i],axis=1)
    #    flats_ima[:,:,i]=np.flip(flats_ima[:,:,i],axis=1)


#Float 64
of0=of0.astype(np.float64)
od0=od0.astype(np.float64)

#Dark and flat image
dark_average=np.mean(darks_ima,axis=2)/N_acc_dark
flat_average=np.mean(flats_ima,axis=2)/N_acc_flat



#Correct images from dark and flat
for i in range(Nseries):
    if dir_flats=='':
        of0[:,:,i]=(of0[:,:,i]-dark_average)
        od0[:,:,i]=(od0[:,:,i]-dark_average)
        of0[:,:,i]=of0[:,:,i]/np.mean(of0[:,:,i])
        od0[:,:,i]=od0[:,:,i]/np.mean(od0[:,:,i])
    else:
        of0[:,:,i]=(of0[:,:,i]-dark_average)/(flat_average-dark_average)
        od0[:,:,i]=(od0[:,:,i]-dark_average)/(flat_average-dark_average)
        of0[:,:,i]=of0[:,:,i]/np.mean(of0[500:1500,500:1500,i])
        od0[:,:,i]=od0[:,:,i]/np.mean(od0[500:1500,500:1500,i])





#Function to replace NaN values by the mean value
def nan_to_mean(of0):
    of_nan=np.argwhere(np.isnan(of0))
    for i in range(of_nan.shape[0]):
        x_ind=of_nan[i][0]
        y_ind=of_nan[i][1]
        z_ind=of_nan[i][2]
        try:
            of0[x_ind,y_ind,z_ind]=(of0[x_ind-1,y_ind-1,z_ind]+of0[x_ind-1,y_ind+1,z_ind]+\
            of0[x_ind+1,y_ind-1,z_ind]+of0[x_ind+1,y_ind+1,z_ind])/4
        except IndexError:
            of0[x_ind,y_ind,z_ind]=np.mean(of0)
    return of0

#Correct Nan values
of0=nan_to_mean(of0)
od0=nan_to_mean(od0)





"""
Shift found using the absolute difference technique with with pixel accuracy
"""
#Accumulate focused and defocused images independently
of=np.sum(of0,axis=2)/Nseries
od=np.sum(od0,axis=2)/Nseries

#Plot
fig,axs=plt.subplots(1,2)
if dir_flats=='':
    axs[0].imshow(of0[xmin:xmax,xmin:xmax,0],cmap='gray')
    axs[1].imshow(od0[xmin:xmax,xmin:xmax,0],cmap='gray')
else:
    axs[0].imshow(of0[xmin:xmax,xmin:xmax,0],cmap='gray')
    axs[1].imshow(od0[xmin:xmax,xmin:xmax,0],cmap='gray')


#Align focused and defocused accumulated images
kappa=20
Of=fft2(of[512:1024,512:1024])
Od=fft2(od[512:1024,512:1024])
error,row_shift,col_shift,Gshift=sf.dftreg(Of,Od,kappa)
deltax=int(np.round(row_shift))
deltay=int(np.round(col_shift))
od=np.roll(od,(deltax,deltay),axis=(0,1))

print('Delta x, Delta y (pixels):',deltax,deltay)

fig2,axs2=plt.subplots(1,2)
axs2[0].imshow(of[xmin:xmax,xmin:xmax],cmap='gray')
axs2[1].imshow(od[xmin:xmax,xmin:xmax],cmap='gray')
plt.show()

#Save images in FITS file
matrix_ofod=np.zeros((of.shape[0],of.shape[1],2))
matrix_ofod[:,:,0]=of
matrix_ofod[:,:,1]=od


gf.save_fits(matrix_ofod[xmin:xmax,xmin:xmax,:],fname)
