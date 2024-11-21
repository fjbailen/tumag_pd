from matplotlib import pyplot as plt
import pd_functions_v21 as pdf
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from scipy.stats import gaussian_kde
plt.rcParams["image.interpolation"] = 'none'
#plt.rcParams['animation.ffmpeg_path']='C://Program Files/ffmpeg-20190312-d227ed5-win64-static/ffmpeg-20190312-d227ed5-win64-static/bin/ffmpeg'

def logplot2(I,sing=True,color=None,low=None,high=None,fourier=True):
    if fourier is True:
        N=I.shape[0]
        inc_nu=1/(N*pdf.Delta_x)
        max_nu=(N-1)*inc_nu
        nuc=pdf.nuc
        extent=[-0.5*max_nu/nuc,0.5*max_nu/nuc,0.5*max_nu/nuc,-0.5*max_nu/nuc]#
        plt.xlabel('$u/\\nu_c$')
        plt.ylabel('$v/\\nu_c$')
    else:
        extent=None
    if sing is True:
        plt.imshow(np.log10(np.abs(I)**2+1),cmap=color,vmin=low,vmax=high,\
        extent=extent)
    elif sing is False:
        plt.imshow(np.log10(np.abs(I)**2),cmap=color,vmin=low,vmax=high,\
        extent=extent)
    plt.colorbar()
    plt.show()
    plt.close()

def plot2(I,color=None,low=None,high=None):
    plt.imshow(I,cmap=color,vmin=low,vmax=high)
    plt.colorbar()
    plt.show()
    plt.close()

def plot_otf(otf):
    N=otf.shape[0]
    otfrad=otf[int(N/2),:]
    inc_nu=1/(N*pdf.Delta_x)
    plt.plot((np.arange(N)-N/2)*inc_nu/pdf.nuc,np.abs(otfrad))
    plt.xlabel('$\\nu/\\nu_c$')
    plt.show()
    plt.close()

def plot2_otf(otf):
    N=otf.shape[0]
    inc_nu=1/(N*pdf.Delta_x)
    max_nu=(N-1)*inc_nu
    nuc=pdf.nuc
    plt.imshow(np.abs(otf),\
    extent=[-0.5*max_nu/nuc,0.5*max_nu/nuc,0.5*max_nu/nuc,-0.5*max_nu/nuc])
    plt.xlabel('$u/\\nu_c$')
    plt.ylabel('$v/\\nu_c$')
    plt.colorbar()
    plt.show()
    plt.close()

def movie(image3D,filename,axis=2,fps=15,cbar='no',titles=[]):
    """
    Creates a movie from a 3D image
    """
    metadata = dict(title='Movie', artist='FJBM',
                comment='')
    writer = FFMpegWriter(fps=fps, metadata=metadata,bitrate=15000)
    n=image3D.shape[axis]
    min=np.min(image3D[:,:,:])
    max=np.max(image3D[:,:,:])
    fig,ax=plt.subplots(figsize=(8, 8))
    tx=ax.text(100, 100, '', fontsize=15, va='top',color='white')
    def animate(i):
        if axis==2:
            ax.imshow(image3D[:,:,i],cmap='gray')
        elif axis==1:
            ax.imshow(image3D[:,i,:],cmap='gray')
        elif axis==0:
            ima=image3D[i,:,:]
            cont=np.round(100*np.std(ima)/np.mean(ima),1)
            tx.set_text('%g'%cont+r'$\,\%$')
            if titles==[]:
                ax.set_title('Frame #%g'%i)
            else:
                ax.set_title(titles[i])
            ax.imshow(ima,cmap='gray',vmin=min,vmax=max)
        if cbar=='yes':
            ax.colorbar()
    ani = FuncAnimation(fig, animate, frames=n, repeat=False)
    ani.save('./'+filename, writer=writer)    

def movie2(im1,im2,filename,axis=2,fps=15,title=['',''],cmap='gray'):
    """
    Creates a movie from two 3D images
    """
    metadata = dict(title='Movie', artist='FJBM',
                comment='')
    writer = FFMpegWriter(fps=fps, metadata=metadata,bitrate=15000)
    n=im1.shape[axis]
    fig,axs=plt.subplots(1,2,layout='constrained',figsize=(15, 10))
    min1=np.min(im1[:,:,:])
    max1=np.max(im1[:,:,:])
    min2=np.min(im2[:,:,:])
    max2=np.max(im2[:,:,:])
    min=np.min((min1,min2))
    max=np.max((max1,max2))
    print('Colorbar limits:',min,max)
    #To use colorbars
    if axis==2:
        axs[0].imshow(im1[:,:,0],cmap=cmap,vmin=min,vmax=max)
        axs[1].imshow(im2[:,:,0],cmap=cmap,vmin=min,vmax=max)
    elif axis==0:
        axs[0].imshow(im1[0,:,:],cmap=cmap,vmin=min,vmax=max)
        axs[1].imshow(im2[0,:,:],cmap=cmap,vmin=min,vmax=max)
    #plt.colorbar(orientation='horizontal')
    axs[0].set_title(title[0]) 
    #plt.colorbar(orientation='horizontal')
    axs[1].set_title(title[1])
    tx1=axs[0].text(100, 100, '', fontsize=15, va='top',color='white')
    tx2=axs[1].text(100, 100, '', fontsize=15, va='top',color='white')

    #Refresh frames
    def animate(i):
        if axis==2:
            #Compute contrasts
            cont1=np.round(100*np.std(im1[:,:,i])/np.mean(im1[:,:,i]),1)
            cont2=np.round(100*np.std(im2[:,:,i])/np.mean(im2[:,:,i]),1)

            #Plots
            axs[0].imshow(im1[:,:,i],cmap=cmap,vmin=min,vmax=max)
            axs[1].imshow(im2[:,:,i],cmap=cmap,vmin=min,vmax=max)
        elif axis==0:
            #Compute contrasts
            cont1=np.round(100*np.std(im1[i,:,:])/np.mean(im1[i,:,:]),1)
            cont2=np.round(100*np.std(im2[i,:,:])/np.mean(im2[i,:,:]),1)

            #Plots
            axs[0].imshow(im1[i,:,:],cmap=cmap,vmin=min,vmax=max)
            axs[1].imshow(im2[i,:,:],cmap=cmap,vmin=min,vmax=max)
        tx1.set_text('%g'%cont1+r'$\,\%$')
        tx2.set_text('%g'%cont2+r'$\,\%$')
    ani = FuncAnimation(fig, animate, frames=n, repeat=False,blit=False)
    ani.save('./'+filename, writer=writer)
    plt.close()

def movie3(im1,im2,filename,axis=2,fps=15,title=['',''],cmap='gray'):
    """
    Creates a movie from two 3D images
    """
    metadata = dict(title='Movie', artist='FJBM',
                comment='')
    writer = FFMpegWriter(fps=fps, metadata=metadata,bitrate=15000)
    n=im1.shape[axis]
    fig = plt.figure(figsize=(15, 10), layout='constrained')
    axs = fig.subplot_mosaic([["im1", "im1", "im1", "im2", "im2", "im2"],
                              ["im1","im1", "im1", "im2", "im2", "im2"],
                              ["im1","im1", "im1", "im2", "im2", "im2"],
                              ["contrast","contrast", "contrast", "contrast","contrast","contrast"]])

    min1=np.min(im1)
    max1=np.max(im1)
    min2=np.min(im2)
    max2=np.max(im2)
    min=np.min((min1,min2))
    max=np.max((max1,max2))
    #To use colorbars
    if axis==2:
        axs["im1"].imshow(im1[:,:,0],cmap=cmap,vmin=min,vmax=max)
        #plt.colorbar(orientation='horizontal')
        axs["im1"].set_title(title[0])
        axs["im2"].imshow(im2[:,:,0],cmap=cmap,vmin=min,vmax=max)
        #plt.colorbar(orientation='horizontal')
        axs["im2"].set_title(title[1])
        tx1=axs["im1"].text(100, 100, '', fontsize=15, va='top',color='white')
        tx2=axs["im2"].text(100, 100, '', fontsize=15, va='top',color='white')

        #Contrast
        axs["contrast"].scatter([],[])
        axs["contrast"].set_xlim([-0.05,15.05])
        axs["contrast"].set_ylim([8,17])
        axs["contrast"].set_ylabel('Contrast [%]')
        axs["contrast"].set_xlabel('Frame index')

    #Refresh frames
    def animate(i):
        print(i)
        if axis==2:
            #Compute contrasts
            cont1=np.round(100*np.std(im1[:,:,i])/np.mean(im1[:,:,i]),1)
            cont2=np.round(100*np.std(im2[:,:,i])/np.mean(im2[:,:,i]),1)

            #Plots
            axs["im1"].imshow(im1[:,:,i],cmap=cmap,vmin=min,vmax=max)
            axs["im2"].imshow(im2[:,:,i],cmap=cmap,vmin=min,vmax=max)
            tx1.set_text('%g'%cont1+r'$\,\%$')
            tx2.set_text('%g'%cont2+r'$\,\%$')
            plot1=axs["contrast"].scatter(i,cont1,color='r')
            plot2=axs["contrast"].scatter(i,cont2,color='b')
            axs["contrast"].legend([plot1, plot2], [title[0], title[1]])

    ani = FuncAnimation(fig, animate, frames=n, repeat=False)
    ani.save('./'+filename, writer=writer)
    plt.close()    

def plot_scatter_density(x,y,xlabel='',ylabel=''):
    """
    Plots a scatter
    """
    # Calculate the point density
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    _, ax = plt.subplots()
    ax.scatter(x, y, c=z, s=50)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()
    return