import os
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import squareform
#from ripser import ripser
#from persim import plot_diagrams, PersImage
path = os.getcwd()


# persistence image
def persistence_image(dgm, resolution = [50,50], return_raw = False, normalization = True, bandwidth = 1., power = 1., dimensional = 0):
    PXs, PYs = dgm[:, 0:1], dgm[:, 1:2] #np.vstack([dgm[:, 0:1] for dgm in dgms]), np.vstack([dgm[:, 1:2] for dgm in dgms])
    if len(PXs) == 0 and len(PYs) == 0:
        return np.ones(resolution)*(10**-8)
    xm, xM, ym, yM = PXs.min(), PXs.max(), PYs.min(), PYs.max()
    x = np.linspace(xm, xM, resolution[0])
    y = np.linspace(ym, yM, resolution[1])
    X, Y = np.meshgrid(x, y)
    Zfinal = np.zeros(X.shape)
    X, Y = X[:, :, np.newaxis], Y[:, :, np.newaxis]

    # Compute persistence image
    P0, P1 = np.reshape(dgm[:, 0], [1, 1, -1]), np.reshape(dgm[:, 1], [1, 1, -1])
    weight = np.abs(P1 - P0)
    distpts = np.sqrt((X - P0) ** 2 + (Y - P1) ** 2)

    if return_raw:
        lw = [weight[0, 0, pt] for pt in range(weight.shape[2])]
        lsum = [distpts[:, :, pt] for pt in range(distpts.shape[2])]
    else:
        weight = weight ** power
        Zfinal = (np.multiply(weight, np.exp(-distpts ** 2 / bandwidth))).sum(axis=2)

    output = [lw, lsum] if return_raw else Zfinal

    if normalization:
        norm_output = (output - np.min(output))/(np.max(output) - np.min(output))
    else:
        norm_output = output

    return norm_output
