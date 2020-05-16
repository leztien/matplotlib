import matplotlib.pyplot as plt


def plot_digits(height, width, data, target=None, image_shape=None):
    image_shape = image_shape or ([int(len(data[0])**.5),]*2 if len(data[0].shape)==1 else data[0].shape)
    fig, axs = plt.subplots(nrows=height, ncols=width, figsize=(width,height),
                            subplot_kw={'xticks':[], 'yticks':[]},
                            gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for (i,sp) in enumerate(axs.flat):
        sp.imshow(data[i].reshape(*image_shape), cmap='binary', interpolation='nearest', clim=(0,16))
        sp.text(0.85, 0.05, str(y[i]), transform=sp.transAxes, color='navy')
        if target is None: sp.axis("off")
    return(fig,axs)

##############################################################################

from sklearn.datasets import load_digits
d = load_digits()
X,y = d.data, d.target
pn = [x.reshape(8,8) for x in X]

plot_digits(height=5, width=10, data=X, target=y)
