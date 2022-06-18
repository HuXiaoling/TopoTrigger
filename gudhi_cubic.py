import numpy as np
import gudhi as gd
from pylab import *
import time

t0=time.time();
im = imread('patch2_epoch10_lh.png')
im_vector = np.asarray(im).flatten()
im_cubic = gd.CubicalComplex(
    dimensions = [200 ,200],
    top_dimensional_cells = im_vector
)
Diag = im_cubic.persistence(homology_coeff_field=2, min_persistence=0)
pairs = im_cubic.cofaces_of_persistence_pairs()
print (pairs)

print('time %.3f'%(time.time()-t0));