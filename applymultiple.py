import time
start = time.clock()
import math
import skimage
import numpy as np
import os
import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings('ignore', category=FutureWarning)
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
#from keras import backend
from keras.layers import *
from keras.models import load_model
from skimage.measure import compare_psnr
# from unet2multiple import cross_entropy_balanced
import os
pngDir = './png/'
# model = load_model('check8.024（200）/fseg-'+'80.hdf5',
#                  #'impd.hdf5',
#                  custom_objects={
#                      'cross_entropy_balanced': cross_entropy_balanced
#                  }
#                  )
####


model = load_model(
                  'check256-480/fseg-' + '90.hdf5'
                  )
####trained model
def main():
  goValidTest()





def goValidTest():
  #############################################
  # seismPath = "./data81-696/test/seis/"
  # faultPath = "./data81-696/test/fault/"
  # filepath3 = "./data56-696/test/out/"
  ##########
  seismPath = "./data256-480/test/seis/"
  faultPath = "./data256-480/test/fault/"
  filepath3 = "./data256-480/test/output/"
  ################input/fault/output###################################
  #####
   n1, n2 = 480, 256
  ################input/fault/output size###################################
  # for i in range(209):
  dk = 3
  gx = np.fromfile(seismPath+str(dk)+'.dat',dtype=np.single)
  fx = np.fromfile(faultPath+str(dk)+'.dat',dtype=np.single)
  gx = np.reshape(gx,(n1,n2))
  fx = np.reshape(fx,(n1,n2))
  # gx = np.transpose(gx)
  # fx = np.transpose(fx)
  # plot2d4(gx, fx, gx, fx)
  #####Standard deviation normalization#######################
  # gm = np.mean(gx)
  # gs = np.std(gx)
  # gx = gx-gm
  # gx = gx/gs
  # ###########################
  # fm = np.mean(fx)
  # fs = np.std(fx)
  # fx = fx - fm
  # fx = fx / fs
  #######max-value normalization####################
  xs = np.max(abs(gx))
  gx = gx / xs
  Ys = np.max(abs(fx))
  fx = fx / Ys
  #####################################
  '''
  gmin = np.min(gx)
  gmax = np.max(gx)
  gx = gx-gmin
  gx = gx/(gmax-gmin)
  '''
  # gx = np.transpose(gx)
  # fx = np.transpose(fx)
  fp = model.predict(np.reshape(gx,(1,n1,n2,1)),verbose=1)
  #####Denormalization##############
  fp = fp*Ys
  gx = gx*Ys
  fx = fx*Ys
  #####normalization of predicted data##############
  # pm = np.mean(fp)
  # ps = np.std(fp)
  # fp = fp - pm
  # fp = fp / ps
  fp = fp[0, :, :, 0]
#####################################Timer statement##############################################################################
  # end = time.clock()
  # print('Runing time:%s Seconds' % (end - start))
########################################################################################################################
  fp1 = fp
  fp2 = gx - fx
  fp3 = gx - fp
  # gx = np.transpose(gx)
  # plot2d(gx, fx, fp2)
  # fp1 = np.transpose(fp1)
  ############################################################
  # fp1.tofile(filepath3 + str(dk)+'predict.dat')
  # fp2.tofile(filepath3 + str(dk) + 'yuancha.dat')
  # fp3.tofile(filepath3 + str(dk) + 'wangcha.dat')
  ############################################################
  # fp.tofile(filepath3 + '20-unet(2).dat')
  # fp2 = np.transpose(fp2)
  ######
  # fp2.tofile(filepath3 + str(dk) + 'predict_cha.dat')
  #############################
  fminus = fx - fp
  ###############################################################
  # fminus.tofile(filepath3 + str(dk) + 'yuanwangcha.dat')
  ###############################################################

  gx = np.reshape(gx, (n2, n1))
  fx = np.reshape(fx, (n2, n1))
  fp = np.reshape(fp, (n2, n1))
  fminus = np.reshape(fminus, (n2, n1))
  fp3 = np.reshape(fp3, (n2, n1))
  fp2 = np.reshape(fp2, (n2, n1))
  gx = np.transpose(gx)
  fx = np.transpose(fx)
  # fp = np.transpose(fp)
  fp1 = np.transpose(fp)
  fminus = np.transpose(fminus)
  fp3 = np.transpose(fp3)
  fp2 = np.transpose(fp2)
  # plot2d(gx, fx, fp)
  # plot2d4(gx, fp2, fp3,fminus)
  ###############Plot a graph############################
  plot2d4(gx, fx, fp1, fminus)
  ###########################################
  # plot2d4(gx, fx, fp, fp3)
  # plt.figure(2)
  # plot2d5(gx, gx - fx, gx - fp1, fminus)
  # fp = fp[0,:,:,:,0]
  # gx1 = gx[50,:,:]
  # fx1 = fx[50,:,:]
  # fp1 = fp[50,:,:]
  # gx2 = gx[:,29,:]
  # fx2 = fx[:,29,:]
  # fp2 = fp[:,29,:]
  # gx3 = gx[:,:,29]
  # fx3 = fx[:,:,29]
  # fp3 = fp[:,:,29]
  # plot2d(gx1,fx1,fp1,png='fp1')
  # plot2d(gx2,fx2,fp2,png='fp2')
  # plot2d(gx3,fx3,fp3,png='fp3')
  # fp = np.transpose(fp)
  fp.tofile(filepath3 + str(dk)+'.dat')


def plot2d(gx,fx,fp):
  vmin, vmax = np.percentile(gx, [2, 98])
  # plt.figure(figsize=(16, 4))
  plt.figure(figsize=(8, 4), dpi=100)
  plt.figure(1)
  plt.subplot(131)
  plt.imshow(gx, cmap='gray', vmin=vmin, vmax=vmax)
  # plt.yticks([0, 20, 40, 60, 80,100,120],[0, 0.08, 0.16, 0.24, 0.32,0.40,0.48])
  # new_ticks = np.linspace(-1, 2, 5)
  # plt.yticks(new_ticks)
  plt.xlabel('trace')
  plt.ylabel('t(s)')
  plt.subplot(132)
  plt.imshow(fx, cmap='gray', vmin=vmin, vmax=vmax)
  # plt.yticks([0, 20, 40, 60, 80, 100, 120], [0, 0.08, 0.16, 0.24, 0.32, 0.40, 0.48])
  plt.xlabel('trace')
  # plt.ylabel('ms')
  plt.subplot(133)
  plt.imshow(fp, cmap='gray', vmin=vmin, vmax=vmax)
  # plt.yticks([0, 20, 40, 60, 80, 100, 120], [0, 0.08, 0.16, 0.24, 0.32, 0.40, 0.48])
  plt.xlabel('trace')
  # plt.ylabel('ms')
  # plt.suptitle('Comparison of results')
  plt.show()
########################################################################################################################
# def plot2d2(mx):
#   vmin, vmax = np.percentile(mx, [2, 98])
#   # plt.figure(figsize=(16, 4))
#   plt.figure(figsize=(9, 4), dpi=100)
#   plt.figure(1)
#   # plt.subplot(131)
#   plt.imshow(mx, cmap='gray', vmin=vmin, vmax=vmax)
#   plt.yticks([0, 20, 40, 60, 80,100,120],[0, 0.08, 0.16, 0.24, 0.32,0.40,0.48])
#   plt.xlabel('trace')
#   plt.ylabel('t(s)')
#   plt.show()
# ########################################################################################################################


def plot2d1(fx):
  # vmin, vmax = np.percentile(gx, [2, 98])
  vmin, vmax = np.percentile(fx, [2, 98])
  # plt.figure(figsize=(16, 4))
  plt.figure(figsize=(8, 4), dpi=200)
  plt.imshow(fx, cmap='gray', vmin=vmin, vmax=vmax)
  plt.yticks([0, 20, 40, 60, 80,100,120],[0, 0.08, 0.16, 0.24, 0.32,0.40,0.48])
  #new_ticks = np.linspace(-1, 2, 5)
  # plt.yticks(new_ticks)
  plt.xlabel('trace')
  plt.ylabel('t(s)')
  plt.show()

def plot2d2(gx):
  # vmin, vmax = np.percentile(gx, [2, 98])
  vmin, vmax = np.percentile(gx, [2, 98])
  # plt.figure(figsize=(16, 4))
  plt.figure(figsize=(8, 4), dpi=200)
  plt.imshow(gx, cmap='gray', vmin=vmin, vmax=vmax)
  plt.yticks([0, 20, 40, 60, 80,100,120],[0, 0.08, 0.16, 0.24, 0.32,0.40,0.48])
  #new_ticks = np.linspace(-1, 2, 5)
  # plt.yticks(new_ticks)
  plt.xlabel('trace')
  plt.ylabel('t(s)')
  plt.show()


def plot2d3(fp):
  # vmin, vmax = np.percentile(gx, [2, 98])
  vmin, vmax = np.percentile(fp, [2, 98])
  # plt.figure(figsize=(16, 4))
  plt.figure(figsize=(8, 4), dpi=200)
  plt.imshow(fp, cmap='gray', vmin=vmin, vmax=vmax)
  plt.yticks([0, 20, 40, 60, 80,100,120],[0, 0.08, 0.16, 0.24, 0.32,0.40,0.48])
  #new_ticks = np.linspace(-1, 2, 5)
  # plt.yticks(new_ticks)
  plt.xlabel('trace')
  plt.ylabel('t(s)')
  plt.show()

###################################################################################
def plot2d4(gx,fx,fp,fminus):
  vmin, vmax = np.percentile(gx, [2, 98])
  # plt.figure(figsize=(16, 4))
  # plt.figure(figsize=(16, 5), dpi=100)
  plt.figure(figsize=(8, 6), dpi=100)
  plt.figure(1)
  plt.subplot(141)
  plt.imshow(gx, cmap='gray', vmin=vmin, vmax=vmax)
  # plt.yticks([0, 20, 40, 60, 80,100,120],[0, 0.08, 0.16, 0.24, 0.32,0.40,0.48])
  # new_ticks = np.linspace(-1, 2, 5)
  # plt.yticks(new_ticks)
  plt.xlabel('trace')
  plt.ylabel('t(s)')
  plt.subplot(142)
  plt.imshow(fx, cmap='gray', vmin=vmin, vmax=vmax)
  # plt.yticks([0, 20, 40, 60, 80, 100, 120], [0, 0.08, 0.16, 0.24, 0.32, 0.40, 0.48])
  plt.xlabel('trace')
  # plt.ylabel('ms')
  plt.subplot(143)
  plt.imshow(fp, cmap='gray', vmin=vmin, vmax=vmax)
  # plt.yticks([0, 20, 40, 60, 80, 100, 120], [0, 0.08, 0.16, 0.24, 0.32, 0.40, 0.48])
  plt.xlabel('trace')
  # plt.ylabel('ms')
  # plt.suptitle('Comparison of results')
  plt.subplot(144)
  plt.imshow(fminus, cmap='gray', vmin=vmin, vmax=vmax)
  # plt.yticks([0, 20, 40, 60, 80, 100, 120], [0, 0.08, 0.16, 0.24, 0.32, 0.40, 0.48])
  plt.xlabel('trace')
  plt.show()
########################################################################################################################
def plot2d5(m1,m2,m3,m4):
  vmin, vmax = np.percentile(m1, [2, 98])
  # plt.figure(figsize=(16, 4))
  plt.figure(figsize=(16, 5), dpi=100)
  plt.figure(1)
  plt.subplot(141)
  plt.imshow(m1, cmap='gray', vmin=vmin, vmax=vmax)
  plt.yticks([0, 20, 40, 60, 80,100,120],[0, 0.08, 0.16, 0.24, 0.32,0.40,0.48])
  # new_ticks = np.linspace(-1, 2, 5)
  # plt.yticks(new_ticks)
  plt.xlabel('trace')
  plt.ylabel('t(s)')
  plt.subplot(142)
  plt.imshow(m2, cmap='gray', vmin=vmin, vmax=vmax)
  plt.yticks([0, 20, 40, 60, 80, 100, 120], [0, 0.08, 0.16, 0.24, 0.32, 0.40, 0.48])
  plt.xlabel('trace')
  # plt.ylabel('ms')
  plt.subplot(143)
  plt.imshow(m3, cmap='gray', vmin=vmin, vmax=vmax)
  plt.yticks([0, 20, 40, 60, 80, 100, 120], [0, 0.08, 0.16, 0.24, 0.32, 0.40, 0.48])
  plt.xlabel('trace')
  # plt.ylabel('ms')
  # plt.suptitle('Comparison of results')
  plt.subplot(144)
  plt.imshow(m4, cmap='gray', vmin=vmin, vmax=vmax)
  plt.yticks([0, 20, 40, 60, 80, 100, 120], [0, 0.08, 0.16, 0.24, 0.32, 0.40, 0.48])
  plt.xlabel('trace')
  plt.show()
########################################################################################################################
# def plot2d1(gx,fp):
#   # vmin, vmax = np.percentile(gx, [2, 98])
#   vmin, vmax = np.percentile(gx, [2, 98])
#   # plt.figure(figsize=(16, 4))
#   plt.figure(figsize=(8, 4), dpi=200)
#   plt.figure(1)
#   plt.subplot(121)
#   plt.imshow(gx, cmap='gray', vmin=vmin, vmax=vmax)
#   # plt.yticks([0, 20, 40, 60, 80,100,120],[0, 0.08, 0.16, 0.24, 0.32,0.40,0.48])
#   # new_ticks = np.linspace(-1, 2, 5)
#   # plt.yticks(new_ticks)
#   plt.xlabel('trace')
#   plt.ylabel('t(s)')
#   plt.subplot(122)
#   plt.imshow(fp, cmap='gray', vmin=vmin, vmax=vmax)
#   # plt.yticks([0, 20, 40, 60, 80, 100, 120], [0, 0.08, 0.16, 0.24, 0.32, 0.40, 0.48])
#   plt.xlabel('trace')
#   # plt.ylabel('ms')
#   # plt.suptitle('Comparison of results')
#   plt.show()
########################################################################################################################

########################################################################################################################
# def plot2d(gx,fx,fp,at=1,png=None):
#   fig = plt.figure(figsize=(15,5))
  #####################################################################################################################
  # ax = fig.add_subplot(131)
  # ax.imshow(gx, vmin=-2, vmax=2, cmap=plt.cm.bone, interpolation='bicubic', aspect=at)
  # ax = fig.add_subplot(132)
  # ax.imshow(fx, vmin=0, vmax=1, cmap=plt.cm.bone, interpolation='bicubic', aspect=at)
  # ax = fig.add_subplot(133)
  # ax.imshow(fp, vmin=0, vmax=1.0, cmap=plt.cm.bone, interpolation='bicubic', aspect=at)
  ######################################################################################
  # ax = fig.add_subplot(131)
  # ax.imshow(gx, vmin=0, vmax=1, cmap=plt.cm.bone, interpolation='bicubic', aspect=at)
  # ax = fig.add_subplot(132)
  # ax.imshow(fx, vmin=0, vmax=1, cmap=plt.cm.bone, interpolation='bicubic', aspect=at)
  # ax = fig.add_subplot(133)
  # ax.imshow(fp,vmin=0,vmax=1,cmap=plt.cm.bone,interpolation='bicubic',aspect=at)
  ######################################################################################
  # plt = fig.add_subplot(131)
  # vmin, vmax = np.percentile(gx, [2, 98])
  # plt.imshow(gx, cmap='gray', vmin=vmin, vmax=vmax)
  # plt = fig.add_subplot(132)
  # vmin, vmax = np.percentile(fx, [2, 98])
  # plt.imshow(fx, cmap='gray', vmin=vmin, vmax=vmax)
  # # plt = fig.add_subplot(133)
  #######################################################################################
  # vmin, vmax = np.percentile(gx, [2, 98])
  # plt.imshow(gx, cmap='gray', vmin=vmin, vmax=vmax)
  # plt.subplot(132)
  # plt.imshow(fx, cmap='gray', vmin=vmin, vmax=vmax)
  # plt.subplot(133)
  # plt.imshow(fp, cmap='gray', vmin=vmin, vmax=vmax)
  # plt.show()
  #######################################################################################

  #####################################################################################

##########################################################################################################################
# def plot2d(gx, fx, fp, at=1, png=None):
#   fig = plt.figure(figsize=(15, 5))
#   ax = fig.add_subplot(131)
#   ax.imshow(gx, vmin=-2, vmax=2, cmap=plt.cm.bone, interpolation='bicubic', aspect=at)
#   ax = fig.add_subplot(132)
#   ax.imshow(fx, vmin=0, vmax=1, cmap=plt.cm.bone, interpolation='bicubic', aspect=at)
#   ax = fig.add_subplot(133)
#   ax.imshow(fp, vmin=0, vmax=1.0, cmap=plt.cm.bone, interpolation='bicubic', aspect=at)
  #######################################################################################
  # ax = fig.add_subplot(131)
  # ax.imshow(gx, cmap=plt.cm.bone, interpolation='bicubic', aspect=at)
  # ax = fig.add_subplot(132)
  # ax.imshow(fx, cmap=plt.cm.bone, interpolation='bicubic', aspect=at)
  # ax = fig.add_subplot(133)
  # ax.imshow(fp, cmap=plt.cm.bone, interpolation='bicubic', aspect=at)
  # if png:
  #   plt.savefig(pngDir+png+'.png')
  # #cbar = plt.colorbar()
  # #cbar.set_label('Fault probability')
  # plt.tight_layout()
  # plt.show()
########################################################################################################################
##################
# def plot2d(gx,fx,at=1,png=None):
#   fig = plt.figure(figsize=(15,5))
#   #fig = plt.figure()
#   ax = fig.add_subplot(131)
#   ax.imshow(gx,vmin=-2,vmax=2,cmap=plt.cm.bone,interpolation='bicubic',aspect=at)
#   ax = fig.add_subplot(132)
#   # ax.imshow(fx,vmin=0,vmax=1,cmap=plt.cm.bone,interpolation='bicubic',aspect=at)
#   ax.imshow(fx, vmin=-2, vmax=2, cmap=plt.cm.bone, interpolation='bicubic', aspect=at)
#   if png:
#     plt.savefig(pngDir+png+'.png')
#   #cbar = plt.colorbar()
#   #cbar.set_label('Fault probability')
#   plt.tight_layout()
#   plt.show()

if __name__ == '__main__':
    main()

end = time.clock()
print('Runing time:%s Seconds'%(end-start))


