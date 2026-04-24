"""Reusable plotting helpers for plume image grids."""

import matplotlib.pyplot as plt
import numpy as np
# import torch
from .utils import NormalizeData, labelfigs, layout_fig, number_to_letters

def create_axes_grid(n_plots, n_per_row, plot_height, n_rows=None, figsize='auto'):
    """
    Create a grid of axes.

    Args:
        n_plots: Number of plots.
        n_per_row: Number of plots per row.
        plot_height: Height of each plot.
        n_rows: Number of rows. If None, it is calculated from n_plots and n_per_row.
        
    Returns:
        axes: Axes object.
    """
    
    if figsize == 'auto':
        figsize = (16, plot_height*n_plots//n_per_row+1)
    elif isinstance(figsize, tuple):
        pass
    elif figsize != None:
        raise ValueError("figsize must be a tuple or 'auto'")
    
    fig, axes = plt.subplots(n_plots//n_per_row+1*int(n_plots%n_per_row>0), n_per_row, figsize=figsize)
    axes = trim_axes(axes, n_plots)
    return fig, axes


def trim_axes(axs, N):

    """
    Reduce *axs* to *N* Axes. All further Axes are removed from the figure.
    """
    # axs = axs.flatten()
    # for ax in axs[N:]:
    #     ax.remove()
    # return axs[:N]
    axes = np.asarray(axs).ravel()
    for i in range(N, len(axes)):
        axes[i].remove()
    return axes[:N]


def show_images(images, labels=None, img_per_row=8, img_height=1, label_size=12, title=None, show_colorbar=False, 
                clim='auto', cmap='viridis', scale_range=False, hist_bins=None, show_axis=False, axes=None, save_path=None):
    
    '''
    Plots multiple images in grid.
    
    images
    labels: labels for every images;
    img_per_row: number of images to show per row;
    img_height: height of image in axes;
    show_colorbar: show colorbar;
    clim: int or list of int, value of standard deviation of colorbar range;
    cmap: colormap;
    scale_range: scale image to a range, default is False, if True, scale to 0-1, if a tuple, scale to the range;
    hist_bins: number of bins for histogram;
    show_axis: show axis
    '''
    
    assert type(images) == list or type(images) == np.ndarray, "do not use torch.tensor for hist"
    if type(clim) == list:
        assert len(images) == len(clim), "length of clims is not matched with number of images"

    h = images[0].shape[1] // images[0].shape[0]*img_height + 1
    if not labels:
        labels = range(len(images))
        
    if isinstance(axes, type(None)):
        if hist_bins: # add a row for histogram
            fig, axes = create_axes_grid(len(images)*2, img_per_row, img_height*2, n_rows=None, figsize='auto')
        else:
            fig, axes = create_axes_grid(len(images), img_per_row, img_height, n_rows=None, figsize='auto')
        
    axes = np.asarray(axes).ravel()
    # if hist_bins:
    #     trim_axes(axes, len(images)*2)
    # else:
    #     trim_axes(axes, len(images))


    for i, img in enumerate(images):

        if hist_bins:
            # insert histogram in after the row
            index = i + (i//img_per_row)*img_per_row
#         if torch.is_tensor(x_tensor):
#             if img.requires_grad: img = img.detach()
#             img = img.numpy()
        else:
            index = i
            
        if isinstance(scale_range, bool): 
            if scale_range: img = NormalizeData(img)
                    
        # if len(images) <= img_per_row and not hist_bins:
        #     index = i%img_per_row
        # else:
        #     index = (i//img_per_row)*n, i%img_per_row
        # print(i, index)

        axes[index].set_title(labels[i], fontsize=label_size)
        im = axes[index].imshow(img, cmap=cmap)

        if clim != 'auto':
            m, s = np.mean(img), np.std(img) 
            if type(clim) == list:
                im.set_clim(m-clim[i]*s, m+clim[i]*s) 
            elif type(clim) == int:
                im.set_clim(m-clim*s, m+clim*s) 
            elif type(clim) == tuple:
                im.set_clim(*clim)

        if show_colorbar:
            fig.colorbar(im, ax=axes[index])
            
        if show_axis:
            axes[index].tick_params(axis="x",direction="in", top=True)
            axes[index].tick_params(axis="y",direction="in", right=True)
        else:
            axes[index].axis('off')

        if hist_bins:
            index_hist = index+img_per_row
            # index_hist = index*2+1
            h = axes[index_hist].hist(img.flatten(), bins=hist_bins)

        if title:
            fig.suptitle(title, fontsize=15)
            plt.tight_layout(pad=0.5)
        else:
            plt.tight_layout()
    # if save_path and isinstance(axes, type(None)): # this is not effective because axes are defined after the function is called
    #     plt.savefig(save_path+'.svg', dpi=300)
    #     plt.savefig(save_path+'.png', dpi=300)
    
    # print(axes)
    # if isinstance(axes, type(None)): # this is not effective because axes are defined after the function is called
    #     plt.show()
    
    
