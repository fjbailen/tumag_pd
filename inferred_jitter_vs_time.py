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
Nrepeat=100 #Number of repetitions for each jitter amplitude
sigmax=0 #RMS of jitter along X in arcsec
sigmay=0 #RMS of jitter along Y in arcsec
SNR=100 #Signal-to-noise ratio. 0 if no noise is to be applied
print_res=False #If True, it prints the results for each optimization
cobs=32.4 #Diameter of central obscuration as a percentage of the aperture
#N=288 #Number of pixels of the image (MHD simulation)
N=256 #Number of pixels (Cadence simulatin)
plate_scale=0.055 #Plate scale of the simulations in arcseconds (arcsec/pixel)
Ntime=40 #Number of seconds of the simulation (max: 40)
Nacc=1000 #Number of accumulated images. Increase to avoid outliers in rms_error vs time
pref='52502' #'517', '52502' or '52506'. Prefilter employed
wvl,fnum,Delta_x=pdf.tumag_params(pref=pref)
nuc,R=pdf.compute_nuc(N,wvl,fnum,Delta_x)
RHO,THETA=pdf.sampling2(N,R) 
ap=pdf.aperture(N,R,cobs=cobs)
aberr=np.zeros(4) #Aberrations of the instrument



"""
Read image
"""
#file='./continuo' #MHD simulation
file='./map_1s-cadence' #Cadence simulation
output='./results/'
#ext='.sav' #MHD simulation
ext='.npz' #Cadence simulation
data=pdf.read_image(file,ext)

#Jitter-free image
ima0=data[:N,:N,0]
ima0=ima0/np.mean(ima0[:256,:256]) #Normalize only by QS mean intensity
ima0=ima0.astype('float64')



"""
Different levels of jitter
"""
rms_error=[] #Norm of the difference between the true and inferred sigma_x and sigma_y
std_rms_error=np.zeros(Ntime) #STD of the rms_error for Nrepeat
corr_error=[] #Error of correlation of jitter between X and Y (rho_xy)

sigma_vec=np.array([sigmax,sigmay,0]) 
for i in tqdm(range(Ntime)):
    sigma_values=np.zeros((3,Nrepeat))
    sigma_error=np.zeros(Nrepeat)

    #Jittered image
    ima2=data[:N,:N,i]
    ima2=ima2/np.mean(ima2[:256,:256]) #Normalize only by QS mean intensity
    ima2=ima2.astype('float64')
    for j in range(Nrepeat):
        #Simulate the image affected by jitter with sigmax and sigmay
        if sigmax>0 or sigmay>0:
            ima_shift,rms_x,rms_y=pdf.simulate_jitter(ima2,sigmax,sigmay,
                                                plate_scale,int(Nacc))
        else:
            ima_shift=ima2
            rms_x=0
            rms_y=0   
        rms=np.sqrt(rms_x**2+rms_y**2)
        rms_vec=np.array([rms_x,rms_y,0]) #3rd dimension:correlation between x and y
        
        #Apply the telescope diffraction and noise to the jitter-free and jittered images
        ima=pdf.convPSF(ima0,aberr,0,RHO,THETA,ap,norm=True)
        ima_shift=pdf.convPSF(ima_shift,aberr,0,RHO,THETA,ap,norm=True)
        if SNR>0:
            NSR=1/SNR
            ima=pdf.gaussian_noise(NSR,ima)
            ima_shift=pdf.gaussian_noise(NSR,ima_shift)

        #Infer the jitter
        cut=int(0.1*N)
        ima_array=np.zeros((N,N,2))
        ima_array[:,:,0]=ima
        ima_array[:,:,1]=ima_shift
        Ok,gamma,wind,susf=pdf.prepare_PD(ima_array,nuc,N)
        sigma=pdf.minimization_jitter(Ok,gamma,plate_scale,nuc,
                                      N,cut=cut,print_res=print_res)
        sigma_values[:,j]=sigma
        sigma_error[j]=np.linalg.norm(sigma_values[:2,j]-rms_vec[:2])
    sigma_mean=np.mean(sigma_values[:2,:],axis=1) 
    rms_error.append(np.mean(sigma_error))
    std_rms_error[i]=np.std(sigma_error)
    corr_error.append(sigma_values[2]-sigma_vec[2])
    #print('Inferred sigma:',sigma_values[:2,:])
    #print('Mean error:',np.mean(sigma_error))
    #print('STD error:',np.std(sigma_error))

   

"""
Save in npy file and plot
"""
time=np.arange(0,Ntime)
rms_array=np.zeros((Ntime,3))
rms_array[:,0]=time
rms_array[:,1]=rms_error
rms_array[:,2]=std_rms_error
fname="jitter_error_vs_time_rms_%g_arcsec_Njit_%g_SNR_%g.npy"%(sigmax,Ntime,SNR)
np.save(fname, rms_array)
test=np.load(fname)
print(test)




#Plot norm of the error of sigma_x and sigma_y
xlabel='Time (s)'
ylabel=r'$\Delta\sigma$ [arcsec]'

fig,axs=plt.subplots()
axs.scatter(time,rms_error)
axs.set_xlabel(xlabel)
axs.set_ylabel(ylabel)
axs.set_title(r"$\sigma=%g$"%sigmax)


#Plot error of the correlation

ylabel=r'$\Delta\rho_{xy}$ [arcsec]'

fig2,axs2=plt.subplots()
axs2.scatter(time,corr_error)
axs2.set_xlabel(xlabel)
axs2.set_ylabel(ylabel)
axs2.set_title(r"$\sigma=%g$"%sigmax)
plt.show()


