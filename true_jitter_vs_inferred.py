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
SNR=0 #Signal-to-noise ratio. 0 if no noise is to be applied
cobs=32.4 #Diameter of central obscuration as a percentage of the aperture
#N=288 #Number of pixels of the image (MHD simulation)
N=256 #Number of pixels (Cadence simulatin)
plate_scale=0.055 #Plate scale of the simulations in arcseconds (arcsec/pixel)
sigma_max=0.15 #Maximum value of the jitter in arcsec
Nsigma=1000 #Number of simulations with different jitter amplitude
sigma=np.random.uniform(0,sigma_max,Nsigma) #RMS values of jitter in arcsec
Nacc=1000 #Number of accumulated images
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
ima0=pdf.read_image(file,ext)
ima0=ima0[:N,:N,0]
ima0=ima0/np.mean(ima0[:256,:256]) #Normalize only by QS mean intensity
ima0=ima0.astype('float64')

"""
Different levels of jitter
"""
rms_true=[]
rms_inferred=[]
for sig in tqdm(sigma):
    #RMS of jitter along X and Y direction (px)
    sigmax=sig 
    sigmay=sig 

    #Simulate the image affected by jitter with sigmax and sigmay
    ima_shift,rms_x,rms_y=pdf.simulate_jitter(ima0,sigmax,sigmay,
                                          plate_scale,Nacc)
    rms=np.sqrt(rms_x**2+rms_y**2)
    rms_true.append(rms)
    #Apply the telescope diffraction
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
    sigma=pdf.minimization_jitter(Ok,gamma,plate_scale,nuc,N,cut=cut)
    rms_inferred.append(np.linalg.norm(sigma))

"""
Save in npy file and plot
"""
rms_array=np.zeros((Nsigma,2))
rms_array[:,0]=rms_true
rms_array[:,1]=rms_inferred
np.save("true_jitter_vs_inferred_Njit_%g_SNR_%g.npy"%(Nsigma,SNR), rms_array)

test=np.load("true_jitter_vs_inferred_Njit_%g_SNR_%g.npy"%(Nsigma,SNR))
print(test)

#Plot
xlabel=r'$\sigma_{\rm true}$ [arcsec]'
ylabel=r'$\sigma_{\rm inferred}$ [arcsec]'
#pf.plot_scatter_density(rms_true,rms_inferred,xlabel,ylabel)


fig,axs=plt.subplots()
axs.scatter(rms_true,rms_inferred)
axs.set_xlabel(xlabel)
axs.set_ylabel(ylabel)
plt.show()


