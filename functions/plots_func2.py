from matplotlib import pyplot as plt
import pd_functions_v22 as pdf
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
    #print('Colorbar limits:',min,max)
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

def movie3(im1,im2,filename,axis=2,fps=15,title=['',''],cmap='gray',
           contrast='full'):
    """
    Creates a movie from two 3D images
        im1, im2: 3D images
        filename: name of the file for the movie
        axis: axis along the series
        fps: frames per second
        title: titles for the images
        cmap: colormap
        contrast: 'full' for computing contrast over the whole image,
                  'corner' for computing contrast over a corner
    """
    metadata = dict(title='Movie', artist='FJBM',
                comment='')
    writer = FFMpegWriter(fps=fps, metadata=metadata,bitrate=15000)
    n=im1.shape[axis] #Number of frames
    fig = plt.figure(figsize=(15, 10), layout='constrained')
    axs = fig.subplot_mosaic([["im1", "im1", "im1", "im2", "im2", "im2"],
                              ["im1","im1", "im1", "im2", "im2", "im2"],
                              ["im1","im1", "im1", "im2", "im2", "im2"],
                              ["contrast","contrast", "contrast", "contrast","contrast","contrast"]])

    min1=np.min(im1)
    max1=np.max(im1)
    min2=np.min(im2)
    max2=np.max(im2)
    min=1.05*np.min((min1,min2))
    max=0.95*np.max((max1,max2))
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

        #Contrast plot settings
        xmax=n+0.05
        dx=10 #To avoid edges when computing the contrast

        if contrast=='full':
            x0=dx
            xf=-dx
            y0=x0
            yf=xf
        elif contrast=='corner':
            x0=dx
            xf=dx+400
            y0=x0
            yf=xf   
        cont1=np.round(100*np.std(im1[x0:xf,y0:yf,0])/np.mean(im1[x0:xf,y0:yf,0]),1)
        cont2=np.round(100*np.std(im2[x0:xf,y0:yf,0])/np.mean(im2[x0:xf,y0:yf,0]),1)
        axs["contrast"].scatter([],[])
        axs["contrast"].set_xlim([-0.05,xmax])
        axs["contrast"].set_ylim([0.9*cont1,1.05*cont2])
        axs["contrast"].set_ylabel('Contrast [%]')
        axs["contrast"].set_xlabel('Frame index')

    #Refresh frames
    def animate(i):
        print('Rendering frame:',i)
        if axis==2:
            #Compute contrasts
            cont1=np.round(100*np.std(im1[x0:xf,y0:yf,i])/np.mean(im1[x0:xf,y0:yf,i]),1)
            cont2=np.round(100*np.std(im2[x0:xf,y0:yf,i])/np.mean(im2[x0:xf,y0:yf,i]),1)
            #cont1=np.round(contrast1[i],1)#np.round(100*np.std(im1[x0:xf,y0:yf,i])/np.mean(im1[x0:xf,y0:yf,i]),1)
            #cont2=np.round(contrast2[i],1)#np.round(100*np.std(im2[x0:xf,y0:yf,i])/np.mean(im2[x0:xf,y0:yf,i]),1)

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

def movie13(im0,im1,im2,filename,axis=2,fps=15,title=['',''],cmap='gray',
           contrast='full'):
    """
    Movie of three 3D images
        im1, im2: 3D images
        filename: name of the file for the movie
        axis: axis along the series
        fps: frames per second
        title: titles for the images
        cmap: colormap
        contrast: 'full' for computing contrast over the whole image,
                  'corner' for computing contrast over a corner
    """
    

    metadata = dict(title='TuMag jitter correction', artist='F.J. Bailen et al. (2025)',
                comment='')
    writer = FFMpegWriter(fps=fps, metadata=metadata,bitrate=15000)
    n=im1.shape[axis] #Number of frames
    Nx=im1.shape[0]
    fig,axs = plt.subplots(1,3,layout='constrained')
    fig.set_figheight(6)
    fig.set_figwidth(15)
   
    min1=np.min(im0)
    max1=np.max(im0)
    min2=np.min(im2)
    max2=np.max(im2)
    min=1.05*np.min((min1,min2))
    max=0.95*np.max((max1,max2))
    #To use colorbars
    if axis==2:
        axs[0].imshow(im0[:,:,0],cmap=cmap,vmin=min,vmax=max)
        axs[0].set_title(title[0])
        axs[1].imshow(im1[:,:,0],cmap=cmap,vmin=min,vmax=max)
        axs[1].set_title(title[1])
        axs[2].imshow(im2[:,:,0],cmap=cmap,vmin=min,vmax=max)
        axs[2].set_title(title[2])
        txframe=axs[0].text(50,50, '', fontsize=15, va='top',color='white')
        tx0=axs[0].text(Nx-250,50, '', fontsize=15, va='top',color='white')
        tx1=axs[1].text(Nx-250,50, '', fontsize=15, va='top',color='white')
        tx2=axs[2].text(Nx-250,50, '', fontsize=15, va='top',color='white')
        remove_tick_labels(axs)

        #Contrast plot settings
        xmax=n+0.05
        dx=10 #To avoid edges when computing the contrast

        if contrast=='full':
            x0=dx
            xf=-dx
            y0=x0
            yf=xf
        elif contrast=='corner':
            x0=dx
            xf=dx+400
            y0=x0
            yf=xf   
        cont0=np.round(100*np.std(im0[x0:xf,y0:yf,0])/np.mean(im0[x0:xf,y0:yf,0]),1)    
        cont1=np.round(100*np.std(im1[x0:xf,y0:yf,0])/np.mean(im1[x0:xf,y0:yf,0]),1)
        cont2=np.round(100*np.std(im2[x0:xf,y0:yf,0])/np.mean(im2[x0:xf,y0:yf,0]),1)
    #Refresh frames
    def animate(i):
        print('Rendering frame:',i)
        if axis==2:
            #Compute contrasts
            cont0=np.round(100*np.std(im0[x0:xf,y0:yf,i])/np.mean(im0[x0:xf,y0:yf,i]),1)
            cont1=np.round(100*np.std(im1[x0:xf,y0:yf,i])/np.mean(im1[x0:xf,y0:yf,i]),1)
            cont2=np.round(100*np.std(im2[x0:xf,y0:yf,i])/np.mean(im2[x0:xf,y0:yf,i]),1)
           
            #Plots
            axs[0].imshow(im0[:,:,i],cmap=cmap,vmin=min,vmax=max)
            axs[1].imshow(im1[:,:,i],cmap=cmap,vmin=min,vmax=max)
            axs[2].imshow(im2[:,:,i],cmap=cmap,vmin=min,vmax=max)
            tx0.set_text('%g'%cont0+r'$\,\%$')
            tx1.set_text('%g'%cont1+r'$\,\%$')
            tx2.set_text('%g'%cont2+r'$\,\%$')
            txframe.set_text('Frame #%g'%(i+1))
    
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


def remove_tick_labels(axs):
    """
    Function to remove tick labels from imshow plots
    """
    for ax in axs.flatten():
        ax.set_xticklabels([])
        ax.set_yticklabels([])

def contrast_along_series(ima,ind1,ind2,region_contrast='full'):
    """
    Computes the contrast along a series of images
        ima=3D image with dimensions (x,y,t)
        ind1, ind2: initial and final image indices
        contrast_region: 'full' for computing contrast over the whole image,
                    'corner' for computing contrast over a corner
    """
    contrast=np.zeros(ind2-ind1)
    i=-1
    dx=10 #To avoid edges when computing the contrast

    if region_contrast=='full':
        x0=dx
        xf=-dx
        y0=x0
        yf=xf
    elif region_contrast=='corner':
        x0=dx
        xf=dx+400
        y0=x0
        yf=xf   
    for ind in range(ind1,ind2):
        i+=1
        contrast[i]=100*np.std(ima[x0:xf,y0:yf,ind])/\
        np.mean(ima[x0:xf,y0:yf,ind])
    return contrast