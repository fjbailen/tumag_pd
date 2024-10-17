"""
This program perfomes a PD analysis of the TuMag data
"""
import numpy as np
import os
import sys
sys.path.append('./functions')
import pd_functions_v21 as pdf
import multiprocessing as mtp
np.set_printoptions(precision=4,suppress=True)
from matplotlib import pyplot as plt
from scipy.fftpack import fftshift, ifftshift, fft2, ifft2
import shift_func as sf
import plots_func2 as pf



"""
Input parameters
"""
check_image=False
realign=False #Realign focused-defocused image with pixel accuracy?
cobs=32.4 #18.5 (MPS) 32.4 (Sunrise) #Diameter of central obscuration as a percentage of the aperture
n_cores=16 #Number of cores of the PC to be employed for parallelization

#Path and name of the FITS file containing the focused and defocused images
cam=0 #Cam 0 or cam 1
Nima=1 #Number of images in the series
dir_folder='./' #Path of the folder containing the FITS file
ffolder='Flight/15_7_20_02' #Name of the folder containing th FITS file
fname='PD_15_7_20_02_cam_%g_ima_%g'%(cam,Nima) #Name of the FITS file
ext='.fits' #Format of the images to be opened (FITS)
output=fname+'/' #Name of output folder to save txt files
optimization='linear' #'linear' (SVD) or 'lbfgs'
K=2 #Number of PD positions

#Zernike reconstruction settings
Jmin=4 #2 #Minimum index of the zernike to be corrected (4 corresponds to defocus)
Jmax=22 #16,22 Index of the maximum Zernike to be retrieved

#Convergence criteria and regularization parameters
tol=0.05#0.01 #Tolerance criterium for stopping iterations (Bonet's powerpoint)
maxnorm=2#0.8 #Maximum norm of the solution at each iteration [rads]
maxit=30 #Maximum number of iterations
w_cut=0.06#0.075#0.02#0.08 #Cut-off for singular values (fraction of the maximum)
cut=int(0.15*pdf.N)#29#int(0.1*pdf.N) #None#Subframing crop to avoid edges
svd_meth='svd' #'svd' or 'lstsq'

#Region to be subframed
crop=True #If True, it crops the image using x0, xf, y0 and yf as limits
x0=400 #Initial pixel of the subframe in X direction
xf=x0+900 #Final pixel of the subframe in X direction
y0=400 #Initial pixel of the subframe in Y direction
yf=y0+900 #FInal pixel of the subframe in Y direction

"""
Defocus and initial guess for PD
"""
#Defocus introduced by PD plate
focus_pos=0
defocus_pos=(32.02-28.33)*1e-3 #Defocus introduced by PD plate
magnif=2.47
defocus_length=magnif**2*np.abs(focus_pos-defocus_pos)
a_d=-np.pi*defocus_length/(8*np.sqrt(3)*pdf.wvl*(pdf.fnum)**2)
a_d=np.array([0,a_d])
defoc=-defocus_length/(8*pdf.wvl*(pdf.fnum)**2) #Peak-to-peak defocus (wvl units)
print('Defocus:',round(defoc,3),' lambda')


#Initial guess for PD optimization [rads]
if Jmin==2:
    a0=np.zeros((2*K+Jmax-3,1)) #Offset + 2K tip/tilt terms + (Jmax-3) terms >=4
else:
    a0=np.zeros((Jmax-1,1))


#Other aberrations induced by PD plate
#a_d=[0,0,0,a_d,0,0,0]


"""
Read image
"""
ima=pdf.read_image(dir_folder+ffolder+'/'+fname,ext,norma='no')
ima=np.nan_to_num(ima, nan=0, posinf=0, neginf=0)
ima=ima.astype("float32") #Change to float 32


#Crop the image to use only a part of the pattern
if crop==True:
    if ima.ndim==4: 
        ima=ima[:,:,x0:xf,y0:yf] #Select the region of interest
    elif ima.ndim==3:
        ima=ima[x0:xf,y0:yf,:]

#Re-order axis and normalize the image [# of image,dimx,dimy,F4/PD]
if ima.ndim==4:
    ima=np.moveaxis(ima,0,-1)
    ima=ima[:Nima,:] #Select Nima pair of images
    ima=np.flip(ima,axis=-1) #Invert the focused and defocused index
    for i in range(Nima):
        ima[i,:]=ima[i,:]/np.mean(ima[:,0]) #Normalize by the mean of the focused image

    
#print(ima.shape)
#pf.movie(ima[:,:200,:200,1],'PD_small_fov.mp4',fps=10,resol=720,axis=0)
#pf.movie(ima[:,:,:,1],'PD_large_fov.mp4',fps=10,resol=720,axis=0)


#Realign images of the series
if realign is True:
    kappa=20
    for j in range(2):
        print('Re-aligning images with index %g'%j)
        F_focused=fft2(ima[0,:,:,j])
        for i in range(1,ima.shape[0]):
            F_comp=fft2(ima[i,:,:,j])
            error,row_shift,col_shift,Gshift=sf.dftreg(F_focused,F_comp,kappa)
            deltax=int(np.round(row_shift))
            deltay=int(np.round(col_shift))
            ima[i,:,:,j]=np.roll(ima[i,:,:,j],(deltax,deltay),axis=(0,1))
            print('Delta x, Delta y (pixels):',row_shift,col_shift)
elif realign is False:
    ima=np.mean(ima,axis=0) #Sum all images over the series




if check_image is True:
    fig,axs=plt.subplots(1,2,layout="constrained")
    if ima.ndim==4:
        ima_foc=ima[0,:,:,0]
        ima_defoc=ima[0,:,:,1]

    elif ima.ndim==3:
        ima_foc=ima[:,:,0]
        ima_defoc=ima[:,:,1]  
    contrast_foc=100*np.std(ima_foc)/np.mean(ima_foc)
    contrast_defoc=100*np.std(ima_defoc)/np.mean(ima_defoc) 
    print('Contrast focused:',np.round(contrast_foc,2),'%')
    print('Contrast defocused:',np.round(contrast_defoc,2),'%')

    axs[0].imshow(ima_foc,cmap='gray',interpolation='none')
    axs[1].imshow(ima_defoc,cmap='gray',interpolation='none')
    axs[0].set_title('Focused')
    axs[1].set_title('Defocused')
    plt.show()
    quit()


if pdf.N != 128:
    print('N in pd_functions is not 128.')
    #quit()

#if ima.shape[0]==1:
#    ima=ima[0,:]


"""
Preparation of PD
"""
RHO,THETA=pdf.sampling2()
ap=pdf.aperture(pdf.N,pdf.R,cobs=cobs)
ima_array=pdf.scanning(ima,Lsiz=pdf.N,cut=cut)
k_vec=np.arange(0,ima_array.shape[0])


"""
num_subplots=3
fig,axs=plt.subplots(num_subplots,num_subplots,layout='constrained')
k=-1
for i in range(num_subplots):
    for j in range(num_subplots):
        k+=1
        axs[i,j].imshow(ima_array[k,:,:,0],cmap='gray',interpolation='none')
        axs[i,j].set_yticklabels([],visible=False)
        axs[i,j].set_xticklabels([],visible=False)
plt.show()
quit()
"""


try:
    os.mkdir('./txt/'+output)
except FileExistsError:
    print('./txt/'+output+' already created')

def subpatch(k):
    guess='False' #No preparado para usar como guess la solución de otra caja

    """
    PD
    """
    #Outputs of prepare_D
    if ima_array.shape[0]==0:#If no subfielding
        Ok,gamma,wind,susf=pdf.prepare_PD(ima)
    else: #If subfielding (pdf.N < data X/Y dimensions)
        Ok,gamma,wind,susf=pdf.prepare_PD(ima_array[k,:,:,:])

    #Call to optimization function
    if optimization=='linear':
        a=pdf.loop_opt(tol,Jmin,Jmax,w_cut,maxnorm,maxit,\
        a0,a_d,RHO,THETA,ap,Ok,cut=cut,method=svd_meth,gamma=gamma,K=K)  
        
    elif optimization=='lbfgs':
        a=pdf.minimization(Jmin,Jmax,a0,a_d,RHO,THETA,ap,Ok,\
        cut=cut,gamma=gamma,K=2,jac=True)

 
    """
    Save txt and images
    """
    #Aberrations and parameters txt file
    flabel=['file','output','ext','Jmin','Jmax',\
    'tol','maxnorm','maxit','w_cut','d','a_d']
    ll=-1
    if Jmin==2: #Save tip/tilt terms
        for i in range(len(a)):
            if i>(2*K):
                flabel.append('a%g'%(i+3-2*K))
            elif i>0:
                if i%2==1:
                    ll+=1
                    flabel.append('tiltx%g'%ll)
                else:
                    flabel.append('tilty%g'%ll)
            elif i==0:
                flabel.append('offset')
    else:
        for i in range(len(a)):
            flabel.append('a%g'%(i+1))
    param=np.array([fname,ffolder,ext,Jmin,Jmax,\
    tol,maxnorm,maxit,w_cut,a_d*np.sqrt(3)/np.pi,a_d],dtype=object)
    filea=np.concatenate((param.reshape(len(param),1),a))
    filea=np.column_stack((flabel,filea))
    filename='./txt/'+output+'a_optimized_Jmax_%g_k_%g.txt'%(Jmax,k)
    np.savetxt(filename,filea,delimiter='\t',fmt='%s',encoding='utf-8')

    #Process information
    proc_name = mtp.current_process().name
    print('Process', proc_name)


#Parallel computing!!!!
if __name__ == '__main__':
    p=mtp.Pool(n_cores)
    p.map(subpatch,k_vec,chunksize=1)
