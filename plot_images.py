import matplotlib.pyplot as plt


def plot_digits(height, width, data, target=None, image_shape=None, title=None):
    image_shape = image_shape or ([int(len(data[0])**.5),]*2 if len(data[0].shape)==1 else data[0].shape)
    fig, axs = plt.subplots(nrows=height, ncols=width, figsize=(width,height),
                            subplot_kw={'xticks':[], 'yticks':[]},
                            gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for (i,sp) in enumerate(axs.flat):
        sp.imshow(data[i].reshape(*image_shape), cmap='binary', interpolation='nearest', clim=(0,16))
        if target is None:
            sp.axis("off")
        else:
            sp.text(0.85, 0.05, str(y[i]), transform=sp.transAxes, color='navy')
    return(fig,axs)


#USE THIS ONE INSTEAD OF THE ABOVE FUNCTION
def plot_multiple_images(nrows, ncols, data, target=None, image_shape=None, figsize=None, title=None, **kwargs):
    image_shape = image_shape or ([int(len(data[0])**.5),]*2 if len(data[0].shape)==1 else data[0].shape)
    figsize = figsize or (ncols,nrows)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,
                            subplot_kw={'xticks':[], 'yticks':[]},
                            gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for (i,sp) in enumerate(axs.flat):
        sp.imshow(data[i].reshape(*image_shape), **kwargs)
        if target is None:
            sp.axis("off")
        else:
            sp.text(0.85, 0.05, str(y[i]), transform=sp.transAxes, color='navy')
    return(fig,axs)

##############################################################################

from sklearn.datasets import load_digits
d = load_digits()
X,y = d.data, d.target
pn = [x.reshape(8,8) for x in X]

#add noise to digits
import numpy as np
X = np.random.normal(loc=X, scale=3)

plot_digits(height=5, width=10, data=X)

#PCA
from sklearn.decomposition import PCA
pca = PCA(0.50).fit(X)
print ("number of components", pca.n_components_)

Xpca = pca.transform(X)
Xfiltered = pca.inverse_transform(Xpca)
plot_digits(height=5, width=10, data=Xfiltered, title="dfdf")
