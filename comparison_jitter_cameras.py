import numpy as np
import matplotlib.pyplot as plt

# Jitter file name for cam0 and cam1
ind1=0 #Initial frame
ind2=70 #Final frame
sigmafolder='./Jitter estimations'
fsigma='sigma_D14-45403-50000'

# Load the jitter data for cam0 and cam1
sigma_cam0 = np.load(sigmafolder+'/'+fsigma+'_cam0.npy')
sigma_cam1 = np.load(sigmafolder+'/'+fsigma+'_cam1.npy')

#Compute rms value
sigma_rms_cam0=np.sqrt(sigma_cam0[ind1:ind2,0]**2+sigma_cam0[ind1:ind2,1]**2)
sigma_rms_cam1=np.sqrt(sigma_cam1[ind1:ind2,0]**2+sigma_cam1[ind1:ind2,1]**2)

# Plot the jitter values
plt.figure(figsize=(10, 5))
plt.plot(sigma_rms_cam0, label='Cam0 Jitter',marker='o')
plt.plot(sigma_rms_cam1, label='Cam1 Jitter',marker='o')
plt.xlabel('Frame')
plt.ylabel('Jitter Value')
plt.title('Jitter Values for Cam0 and Cam1')
plt.legend()
plt.show()