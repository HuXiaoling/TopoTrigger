# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# from matplotlib.colors import LogNorm
# import matplotlib.pyplot as plt
#
# try:
#     import numpy as np
# except:
#     exit()
#
# from deap import benchmarks
#
# # NUMMAX = 5
# # A = 10 * np.random.rand(NUMMAX, 2)
# # C = np.random.rand(NUMMAX)
#
# A = [[0.5, 0.5], [0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]]
# C = [0.002, 0.005, 0.005, 0.005, 0.005]
#
#
# def shekel_arg0(sol):
#     return benchmarks.shekel(sol, A, C)[0]
#
#
# fig = plt.figure()
# # ax = Axes3D(fig, azim = -29, elev = 50)
# ax = Axes3D(fig)
# X = np.arange(0, 1, 0.01)
# Y = np.arange(0, 1, 0.01)
# X, Y = np.meshgrid(X, Y)
# Z = np.fromiter(map(shekel_arg0, zip(X.flat, Y.flat)), dtype=np.float, count=X.shape[0] * X.shape[1]).reshape(X.shape)
#
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, norm=LogNorm(), cmap=cm.jet, linewidth=0.2)
# plt.axis('off')
#
# # plt.xlabel("x")
# # plt.ylabel("y")
#
# plt.show()
# plt.savefig("test.png", bbox_inches='tight')

from skimage.color import rgb2gray
import skimage.io
import numpy
import torchvision.datasets.folder
import torchvision.transforms.functional as Ft
import torchvision.transforms as Ts
import imageio

img = skimage.io.imread('class_16_example_0_poison.png')
h, w, c = img.shape
dx = int((w - 224) / 2)
dy = int((w - 224) / 2)
img = img[dy:dy + 224, dx:dx + 224, :]
# perform tensor formatting and normalization explicitly
# convert to CHW dimension ordering
# img = numpy.transpose(img, (2, 0, 1))
# convert to NCHW dimension ordering
# normalize the image
img = img - numpy.min(img)
img = img / numpy.max(img)
skimage.io.imsave("test.png", img)
