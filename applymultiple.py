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
####当有自定义的损失函数时需要加上custom_objects
# model = load_model('check7.0398（200）/fseg-'+'200.hdf5')
model = load_model(
                   # 'check14.1/fseg'+'.hdf5'
                   # 'check15.2/fseg-'+'200.hdf5'
                   # 'check16.2/fseg-'+'200.hdf5'
                   # 'check17.1/fseg-'+'400.hdf5'
                   #  'check21.1/fseg-'+'200.hdf5'
                   #  'check20.3/fseg-' + '300.hdf5'
                  # 'check24.2/fseg-' + '300.hdf5'
                  #   'check25.2/fseg-' + '1000.hdf5'
                  # 'check1/fseg-' + '600.hdf5'
                  # 'check9.1/fseg-' + '300.hdf5'
                  # 'check22.1/fseg-' + '200.hdf5'
                  'check256-480/fseg-' + '105.hdf5'
                  # 'check56-696/fseg-' + '180.hdf5'
                  # 'check14.31/fseg-'+'200.hdf5'
                   # 'check5/fseg-' + '200.hdf5'
                   # 'check6/fseg-' + '40.hdf5'
#                  #'impd.hdf5',
#                  custom_objects=None
                  )
def main():
  # goTrainTest()
  goValidTest()
  # goF3Test()
  # goF2Test()
  # Test_batch()


def goTrainTest():
  #def __init__(self,data_IDs):
  # train_ID = range(100)
  # data_IDs = train_ID
  # self.data_IDs = data_IDs
  # data_IDs_temp = [self.data_IDs[k] for k in indexes]
  ####端到端###
  # seismPath = "./data7/train1/seis/"
  # faultPath = "./data7/train1/fault/"
  # filepath3 = "./data7/train1/prediction/"
  ############
  ####Radon变换##################################################################
  # seismPath = "./data9/train6/low/"
  # faultPath = "./data9/train6/high/"
  # filepath3 = "./data9/h_output6/train/"
  ######################################################################
  seismPath = "./data11/train1/seis/"
  faultPath = "./data11/train1/fault/"
  # seismPath = "./data7/train/seis/"
  # faultPath = "./data7/train/fault/"
  filepath3 = "./data11/fault_output1/train/"
  #######
  # filePath = "./testdata/test1/"
  # mx = np.fromfile(filePath +'0.dat', dtype=np.single)
  # mx = np.reshape(mx, (128, 128))
  # mx = np.transpose(mx)
  # plot2d2(mx)
  # seismPath = "./data5/train/fault/"
  # faultPath = "./data5/train/seis/"
  n1,n2=128,128
  # n1, n2 = 64, 64
  dk = 51
  gx = np.fromfile(seismPath+str(dk)+'.dat',dtype=np.single)
  fx = np.fromfile(faultPath+str(dk)+'.dat',dtype=np.single)
  gx = np.reshape(gx,(n1,n2))
  fx = np.reshape(fx,(n1,n2))
  ######################################################################################################################
  # gx1 = gx[32:-32, 32:-32]
  ########################################
  # gm = np.mean(gx)
  # gs = np.std(gx)
  # gx = gx-gm
  # gx = gx/gs
  # fm = np.mean(fx)
  # fs = np.std(fx)
  # fx = fx - fm
  # fx = fx / fs
  # gx.tofile(filepath3 + '10.dat')
  ##########################
  # xs = np.max(gx)
  # gx = gx / xs
  # gx.tofile(filepath3 + '9.dat')
  # xm = np.mean(gx)
  # gx = gx - xm
  ######################################################################################################################
  # fm = np.mean(fx)
  # fs = np.std(fx)
  # fx = fx - fm
  # fx = fx / fs
  ###################################
  # xm = np.mean(gx)
  # gx = gx - xm
  # Ym = np.mean(fx)
  # fx = fx - Ym
  # fx.tofile(filepath3 + '3.dat')
  ##################################
  # gmin = np.min(gx)
  # gmax = np.max(gx)
  # gx = gx - gmin
  # gx = gx / (gmax - gmin)
  ################################
  gm = np.mean(gx)
  gs = np.std(gx)
  gx = gx - gm
  gx = gx / gs
  ###########################
  fm = np.mean(fx)
  fs = np.std(fx)
  fx = fx - fm
  fx = fx / fs
  # ##############################
  # xs = np.max(abs(gx))
  # gx = gx / xs
  # Ys = np.max(abs(fx))
  # fx = fx / Ys
  ######################################################################################################################
  # gx1 = gx[64:, 64:]
  # gx1.tofile(filepath3 + '5.dat')
  # gx.tofile(filepath3 + '1.dat')
  # fx.tofile(filepath3 + '2.dat')
  gx = np.transpose(gx)
  fx = np.transpose(fx)
  # gx.tofile(filepath3 + '1.dat')
  # fx.tofile(filepath3 + '2.dat')
  fp = model.predict(np.reshape(gx,(1,n1,n2,1)),verbose=1)
  fp = fp[0, :, :, 0]
  fminus = fx-fp
  #############################################
  fp = np.transpose(fp)
  fp.tofile(filepath3 + str(dk)+'(10).dat')
  #############################################
  # plot2d(gx, fx, fp)
  plot2d4(gx, fx, fp, fminus)
  ######################################################################################################################
  # plot2d1(fx)
  # plot2d2(gx)
  # plot2d3(fp)
  ###########################图像数值显示修正代码###########################################################################
  # clip = 1e-0  # 显示范围，负值越大越明显
  # # clip = 10
  #
  # vmin, vmax = -clip, clip
  # # Figure
  # figsize = (15, 5)  # 设置图形的大小
  # fig, axs = plt.subplots(nrows=1, ncols=3, figsize=figsize, facecolor='w', edgecolor='k',
  #                         squeeze=False, sharex=True, dpi=100)
  # axs = axs.ravel()  # 将多维数组转换为一维数组
  # # fp = np.reshape(fp, (128, 128))
  # # fx = np.reshape(fx, (128, 128))
  # axs[0].imshow(gx, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
  # # axs[0].imshow(gx, cmap=plt.cm.seismic)
  # # axs[0].set_title('Clear')
  # # axs[0].grid(False)
  #
  # clip = 1e-0  # 显示范围，负值越大越明显
  # axs[1].imshow(fx, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
  # # noisy_psnr = psnr(data,data_test)
  # # noisy_psnr=round(noisy_psnr, 2)
  # # axs[1].set_title('Noisy, psnr='+ str(noisy_psnr))
  # # axs[1].grid(False)
  #
  # clip = 1e-0  # 显示范围，负值越大越明显
  # vmin, vmax = -clip, clip
  # axs[2].imshow(fp, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
  # # Denoised_psnr = psnr(data,data_cons)
  # # Denoised_psnr=round(Denoised_psnr, 2)
  # # axs[2].set_title('Denoised proposed, psnr='+ str(Denoised_psnr))
  # # axs[2].grid(False)
  # plt.show()

  ######################################################################################################################
  # pm = np.mean(fp)
  # ps = np.std(fp)
  # fp = fp - pm
  # fp = fp / ps
  # fp = np.transpose(fp)
  # fp.tofile(filepath3 + '3.dat')
  ##############################################
  # pm = np.max(fp)
  # fp = fp / pm
  # pm = np.mean(fp)
  # fp = fp-pm

  ######################################################################################################################
  #fp = np.transpose(fp)
  #fp = fp[0,:,:,0]
  # gx1 = gx[50,:,:]
  # fx1 = fx[50,:,:]
  # fp1 = fp[50,:,:]
  # plot2d(gx1,fx1,fp1,png='fp')
  # plot2d(gx, fx, fp)
  # plot2d(gx, fx)

def goValidTest():
  ####端到端#########################################
  # seismPath = "./data81-696/test/seis/"
  # faultPath = "./data81-696/test/fault/"
  # filepath3 = "./data56-696/test/out/"
  ##########
  seismPath = "./data256-480/test/seis/"
  faultPath = "./data256-480/test/fault/"
  filepath3 = "./data256-480/test/output/"
  ###################################################
  # seismPath = "./data20/train/seis/"
  # faultPath = "./data20/train/fault/"
  # filepath3 = "./data20/fault_output4/"
  ###########端实际数据SRME#############################
  # seismPath = "./data25/train2/seis/"
  # faultPath = "./data25/train2/fault/"
  # filepath3 = "./data25/fault_output4/"
  ##########端实际数据Radon#############################
  # seismPath = "./data22/train/seis/"
  # faultPath = "./data22/train/fault/"
  # filepath3 = "./data22/fault_output4/"
  ####################################################
  # seismPath = "./data16/train/seis/"
  # faultPath = "./data16/train/fault/"
  # filepath3 = "./data16/fault_output/"
  # seismPath = "./data7/test1/seis/"
  # faultPath = "./data7/test1/fault/"
  # filepath3 = "./data7/test1/prediction/"
  #####
  ####Radon变换####
  # seismPath = "./data17/train/low/"
  # faultPath = "./data17/train/high/"
  # filepath3 = "./data17/fault_output2/"
  # seismPath = "./data21/train/low/"
  # faultPath = "./data21/train/high/"
  # filepath3 = "./data21/fault_output3/"
  # seismPath = "./data15/train/low/"
  # faultPath = "./data15/train/high/"
  # filepath3 = "./data15/fault_output/"
  ################
  # seismPath = "./data9/test1/low/"
  # faultPath = "./data9/test1/high/"
  # filepath3 = "./data9/h_output1/test/"
  #####
  ####image###
  ##valid##
  # seismPath = "./data11/validation1/seis/"
  # faultPath = "./data11/validation1/fault/"
  # filepath3 = "./data11/fault_output1/valid/"
  ##test##
  # seismPath = "./data5/train/seis/"
  # faultPath = "./data5/train/fault/"
  # filepath3 = "./data5/fault_output/"
  #########
  # seismPath = "./data16/train/seis/"
  # faultPath = "./data16/train/fault/"
  # filepath3 = "./data16/fault_output/"
  #########
  # filepath3 = "./data4/test/output2/"
  ####################################
  # filepath3 = "./data9/h_output6/valid/"
  ####################################
  # n1, n2 = 496, 96
  # n1, n2 = 256, 64
  # n1, n2 = 1600, 32
  # n1, n2 = 1600, 64
  # n1, n2 = 1600, 128
  # n1, n2 = 1600, 32
  # n1,n2 = 1024,32
  #n1, n2 = 696, 80
  n1, n2 = 480, 256
  # n1, n2 = 240, 480
  # n1, n2 = 624,192
  # n1, n2 = 496, 96
  # n1, n2 = 1536, 64
  # n1, n2 = 800, 64
  # for i in range(209):
  dk = 3
  gx = np.fromfile(seismPath+str(dk)+'.dat',dtype=np.single)
  fx = np.fromfile(faultPath+str(dk)+'.dat',dtype=np.single)
  gx = np.reshape(gx,(n1,n2))
  fx = np.reshape(fx,(n1,n2))
  # gx = np.transpose(gx)
  # fx = np.transpose(fx)
  # plot2d4(gx, fx, gx, fx)
  #####标准差归一化#######################
  # gm = np.mean(gx)
  # gs = np.std(gx)
  # gx = gx-gm
  # gx = gx/gs
  # ###########################
  # fm = np.mean(fx)
  # fs = np.std(fx)
  # fx = fx - fm
  # fx = fx / fs
  #######除最大值归一化####################
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
  fp = fp*Ys
  gx = gx*Ys
  fx = fx*Ys
  #####预测数据归一化##############
  # pm = np.mean(fp)
  # ps = np.std(fp)
  # fp = fp - pm
  # fp = fp / ps
  fp = fp[0, :, :, 0]
#####################################计时器##############################################################################
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
  ###########################################
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

def goF3Test():
  seismPath = "./data2/train/seis/"
  faultPath = "./data2/train/fault/"
  filepath3 = "./data2/test/output/"
  # filepath1 = "./data7/test/prediction/"
  # faultPath = "./data7/test/fault/"
  n1, n2 = 1024,  32
  dk = 1
  # gx = np.fromfile(seismPath + 'nmo_add.dat', dtype=np.single)
  # fx = np.fromfile(faultPath + 'p1_add.dat', dtype=np.single)
  gx = np.fromfile(seismPath + str(dk) + '.dat', dtype=np.single)
  fx = np.fromfile(faultPath + str(dk) + '.dat', dtype=np.single)
  gx = np.reshape(gx, (n1, n2))
  fx = np.reshape(fx, (n1, n2))
  #####标准差归一化#######################
  gm = np.mean(gx)
  gs = np.std(gx)
  gx = gx - gm
  gx = gx / gs
  ###########################
  fm = np.mean(fx)
  fs = np.std(fx)
  fx = fx - fm
  fx = fx / fs
  #######除最大值归一化####################
  # xs = np.max(abs(gx))
  # gx = gx / xs
  # Ys = np.max(abs(fx))
  # fx = fx / Ys
  '''
  gmin = np.min(gx)
  gmax = np.max(gx)
  gx = gx-gmin
  gx = gx/(gmax-gmin)
  '''
  # gx = np.transpose(gx)
  # fx = np.transpose(fx)
  fp = model.predict(np.reshape(gx, (1, n1, n2, 1)), verbose=1)
  #####预测数据归一化##############
  # pm = np.mean(fp)
  # ps = np.std(fp)
  # fp = fp - pm
  # fp = fp / ps
  fp = fp[0, :, :, 0]
  #####################################计时器##############################################################################
  end = time.clock()
  print('Runing time:%s Seconds' % (end - start))
  ########################################################################################################################
  fp1 = fp
  fp2 = fx - fp
  # gx = np.transpose(gx)
  # plot2d(gx, fx, fp2)
  # fp1 = np.transpose(fp1)
  fp1.tofile(filepath3 + str(dk) + '.dat')
  # fp.tofile(filepath3 + '20-unet(2).dat')
  # fp2 = np.transpose(fp2)
  fp2.tofile(filepath3 + str(dk) + 'cha.dat')
  #######
  fminus = fx - fp
  gx = np.reshape(gx, (n2, n1))
  fx = np.reshape(fx, (n2, n1))
  fp = np.reshape(fp, (n2, n1))
  fminus = np.reshape(fminus, (n2, n1))
  gx = np.transpose(gx)
  fx = np.transpose(fx)
  fp = np.transpose(fp)
  fminus = np.transpose(fminus)
  # plot2d(gx, fx, fp)
  plot2d4(gx, fx, fp, fminus)

def goF2Test():
  # seismPath = "./data12/train/seis8/"
  # faultPath = "./data12/train/fault8/"
  # filepath3 = "./data12/test/output2/"
  # seismPath = "./data1/train/seis/"
  # faultPath = "./data1/train/fault/"
  # filepath3 = "./data1/fault_output/"
  ######################################
  seismPath = "./data9/train/seis/"
  faultPath = "./data9/train/fault/"
  filepath3 = "./data9/fault_output/"
  ######################################
  # seismPath = "./data9/test/seis/"
  # faultPath = "./data9/test/fault/"
  # filepath3 = "./data9/fault_output/"
  #####################################
  # filepath1 = "./data7/test/prediction/"
  # faultPath = "./data7/test/fault/"
  # n1, n2 = 496, 96
  # n1, n2 = 256, 128
  # n1, n2 = 768, 768
  # n1, n2 = 320, 16
  # n1, n2 = 400, 48
  n1, n2 = 128, 128
  dk = 44
  # gx = np.fromfile(seismPath + 'new_768.dat', dtype=np.single)
  # fx = np.fromfile(faultPath + 'ori_768.dat', dtype=np.single)
  gx = np.fromfile(seismPath + str(dk) + '.dat', dtype=np.single)
  fx = np.fromfile(faultPath + str(dk) + '.dat', dtype=np.single)
  gx = np.reshape(gx, (n1, n2))
  fx = np.reshape(fx, (n1, n2))
  #####标准差归一化#######################
  # gm = np.mean(gx)
  # gs = np.std(gx)
  # gx = gx - gm
  # gx = gx / gs
  # ###########################
  # fm = np.mean(fx)
  # fs = np.std(fx)
  # fx = fx - fm
  # fx = fx / fs
  #######除最大值归一化####################
  xs = np.max(abs(gx))
  gx = gx / xs
  Ys = np.max(abs(fx))
  fx = fx / Ys
  # '''
  # gmin = np.min(gx)
  # gmax = np.max(gx)
  # gx = gx-gmin
  # gx = gx/(gmax-gmin)
  # '''
  # gx = np.transpose(gx)
  # fx = np.transpose(fx)
  fp = model.predict(np.reshape(gx, (1, n1, n2, 1)), verbose=1)
  # fp = model.predict(gx, verbose=1)
  #####预测数据归一化##############
  # pm = np.mean(fp)
  # ps = np.std(fp)
  # fp = fp - pm
  # fp = fp / ps
  fp = fp[0, :, :, 0]
  #####################################计时器##############################################################################
  end = time.clock()
  print('Runing time:%s Seconds' % (end - start))
  ########################################################################################################################
  fp1 = fp
  # fp2 = fx - fp
  fp2 = gx - fx
  # gx = np.transpose(gx)
  # plot2d(gx, fx, fp2)
  # fp1 = np.transpose(fp1)
  # fp1.tofile(filepath3 + str(dk) + '.dat')
  # fp.tofile(filepath3 + '20-unet(2).dat')
  # fp2 = np.transpose(fp2)
  ############################################
  # fp2.tofile(filepath3 + str(dk) + 'cha.dat')
  #######
  fminus = fp - fx
  # fminus = gx - fp
  gx = np.reshape(gx, (n2, n1))
  fx = np.reshape(fx, (n2, n1))
  fp = np.reshape(fp, (n2, n1))
  fminus = np.reshape(fminus, (n2, n1))
  gx = np.transpose(gx)
  fx = np.transpose(fx)
  fp = np.transpose(fp)
  fminus = np.transpose(fminus)
  plot2d(gx, fx, fp)
  # plot2d4(gx, fx, fp, fminus)

def Test_batch():
  # def __init__(self,data_IDs):
  # train_ID = range(100)
  # data_IDs = train_ID
  # self.data_IDs = data_IDs
  # data_IDs_temp = [self.data_IDs[k] for k in indexes]
  ####端到端###
  # seismPath = "./data7/train1/seis/"
  # faultPath = "./data7/train1/fault/"
  # filepath3 = "./data7/train1/prediction/"
  ############
  ####Radon变换##################################################################
  # seismPath = "./data9/train6/low/"
  # faultPath = "./data9/train6/high/"
  # filepath3 = "./data9/h_output6/train/"
  ####成像域##################################################################
  # seismPath = "./data11/train1/seis/"
  # faultPath = "./data11/train1/fault/"
  #########################################
  # seismPath = "./data11/25_data_batch/"
  seismPath = "./data6/test/"
  # faultPath = "./data11/9_data_batch/"
  # seismPath = "./data7/train/seis/"
  # faultPath = "./data7/train/fault/"
  # filepath3 = "./data11/fault_output_batch/"
  filepath3 = "./data6/test/"
  ##############单独数据预测#########
  n1, n2 = 416, 256
  gx = np.fromfile(seismPath + 'imagv96_m_que30s_bu1.dat', dtype=np.single)
  # fx = np.fromfile(faultPath+str(i)+'.dat',dtype=np.single)
  gx = np.reshape(gx, (n1, n2))
  # gx = np.transpose(gx)
  plot2d1(gx)

  gm = np.mean(gx)
  gs = np.std(gx)
  gx = gx - gm
  gx = gx / gs
  ###########################
  # fm = np.mean(fx)
  # fs = np.std(fx)
  # fx = fx - fm
  # fx = fx / fs
  # gx = np.transpose(gx)
  fp = model.predict(np.reshape(gx, (1, n1, n2, 1)), verbose=1)
  # mx = np.reshape(gx,(1, n2, n1, 1))
  # # plot2d1(mx)
  # fp = model.predict(mx, verbose=1)
  fp = fp[0, :, :, 0]
  # fp = np.transpose(fp)
  fp.tofile(filepath3 + 'imagv96_m_que30s_bu_predict3.dat')
  ################################
  ################################批量数据预测#######
  # seismPath = "./data11/25_data_batch/"
  # faultPath = "./data11/9_data_batch/"
  # filepath3 = "./data11/fault_output_batch/"
  # n1,n2=128,128
  # # n1, n2 = 64, 64
  # # dk = 72
  # for i in range(144):
  #     gx = np.fromfile(seismPath+str(i)+'.dat',dtype=np.single)
  #     # fx = np.fromfile(faultPath+str(i)+'.dat',dtype=np.single)
  #     gx = np.reshape(gx,(n1,n2))
  #     # fx = np.reshape(fx,(n1,n2))
  #     ######################################################################################################################
  #     # gx1 = gx[32:-32, 32:-32]
  #     ########################################
  #     # gm = np.mean(gx)
  #     # gs = np.std(gx)
  #     # gx = gx-gm
  #     # gx = gx/gs
  #     # fm = np.mean(fx)
  #     # fs = np.std(fx)
  #     # fx = fx - fm
  #     # fx = fx / fs
  #     # gx.tofile(filepath3 + '10.dat')
  #     ##########################
  #     # xs = np.max(gx)
  #     # gx = gx / xs
  #     # gx.tofile(filepath3 + '9.dat')
  #     # xm = np.mean(gx)
  #     # gx = gx - xm
  #     ######################################################################################################################
  #     # fm = np.mean(fx)
  #     # fs = np.std(fx)
  #     # fx = fx - fm
  #     # fx = fx / fs
  #     ###################################
  #     # xm = np.mean(gx)
  #     # gx = gx - xm
  #     # Ym = np.mean(fx)
  #     # fx = fx - Ym
  #     # fx.tofile(filepath3 + '3.dat')
  #     ##################################
  #     # gmin = np.min(gx)
  #     # gmax = np.max(gx)
  #     # gx = gx - gmin
  #     # gx = gx / (gmax - gmin)
  #     ################################
  #     gm = np.mean(gx)
  #     gs = np.std(gx)
  #     gx = gx - gm
  #     gx = gx / gs
  #     ###########################
  #     # fm = np.mean(fx)
  #     # fs = np.std(fx)
  #     # fx = fx - fm
  #     # fx = fx / fs
  #     # ##############################
  #     # xs = np.max(abs(gx))
  #     # gx = gx / xs
  #     # Ys = np.max(abs(fx))
  #     # fx = fx / Ys
  #     ######################################################################################################################
  #     # fx.tofile(filepath3 + '2.dat')
  #     # gx = np.transpose(gx)
  #     # fx = np.transpose(fx)
  #     # gx.tofile(filepath3 + '1.dat')
  #     # fx.tofile(filepath3 + '2.dat')
  #     fp = model.predict(np.reshape(gx,(1,n1,n2,1)),verbose=1)
  #     fp = fp[0, :, :, 0]
  #     # fminus = fx-fp
  #     #############################################
  #     # fp = np.transpose(fp)
  #     fp.tofile(filepath3 + str(i)+'.dat')
########################################################################################################################
  # # seismPath = "./data6/validation/seis/"
  # seismPath = "./data3/prediction/f3d/"
  # #seismPath = "./data3/prediction/f3d/"
  # n3, n2 = 384, 512
  # # n3, n2 = 128, 128
  # gx = np.fromfile(seismPath + 'gxl.dat', dtype=np.single)
  # gx = np.reshape(gx, (n3, n2))
  # gm = np.mean(gx)
  # gs = np.std(gx)
  # gx = gx - gm
  # gx = gx / gs
  # '''
  # gmin = np.min(gx)
  # gmax = np.max(gx)
  # gx = gx-gmin
  # gx = gx/(gmax-gmin)
  # '''
  # gx = np.transpose(gx)
  # fp = model.predict(np.reshape(gx, (1, n2, n3, 1)), verbose=1)
  # fp = fp[0, :, :, 0]
  # # fp = fp[0,:,:,:,0]
  # # gx1 = gx[99,:,:]
  # # fp1 = fp[99,:,:]
  # # gx2 = gx[:,29,:]
  # # fp2 = fp[:,29,:]
  # # gx3 = gx[:,:,29]
  # # fp3 = fp[:,:,29]
  # # plot2d(gx1,fp1,fp1,at=1,png='f3d/fp1')
  # # plot2d(gx2,fp2,fp2,at=2,png='f3d/fp2')
  # # plot2d(gx3,fp3,fp3,at=2,png='f3d/fp3')
  # plot2d(gx, fp, fp)

  # # n1,n2=161,201
  # n2, n1 = 201, 161
  # gx = np.fromfile(seismPath+'0.dat',dtype=np.single)
  # gx = np.reshape(gx,(n1,n2))
  # gm = np.mean(gx)
  # gs = np.std(gx)
  # gx = gx-gm
  # gx = gx/gs
  # '''
  # gmin = np.min(gx)
  # gmax = np.max(gx)
  # gx = gx-gmin
  # gx = gx/(gmax-gmin)
  # '''
  # gx = np.transpose(gx)
  # fp = model.predict(np.reshape(gx,(1,n1,n2,1)),verbose=1)
  # pm = np.mean(fp)
  # ps = np.std(fp)
  # fp = fp - pm
  # fp = fp / ps
  # fp = fp[0, :, :, 0]
  # plot2d(gx, gx, fp)
  # # fp = fp[0,:,:,:,0]
  # # gx1 = gx[99,:,:]
  # # fp1 = fp[99,:,:]
  # # gx2 = gx[:,29,:]
  # # fp2 = fp[:,29,:]
  # # gx3 = gx[:,:,29]
  # # fp3 = fp[:,:,29]
  # # plot2d(gx1,fp1,fp1,at=1,png='f3d/fp1')
  # # plot2d(gx2,fp2,fp2,at=2,png='f3d/fp2')
  # # plot2d(gx3,fp3,fp3,at=2,png='f3d/fp3')

#########################画图地震数据图(源代码示例)###################################################################################

# data_cons= dncnn.predict(data_test)

# clip = 1e-0#显示范围，负值越大越明显
# vmin, vmax = -clip, clip
#         # Figure
# figsize=(10, 10)#设置图形的大小
# fig, axs = plt.subplots(nrows=1, ncols=3, figsize=figsize, facecolor='w', edgecolor='k',
#                                squeeze=False,sharex=True,dpi=100)
# axs = axs.ravel()#将多维数组转换为一维数组
# fp = np.reshape(fp, (128, 128))
# fx = np.reshape(fx, (128, 128))
# axs[0].imshow(gx, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
# # axs[0].set_title('Clear')
# # axs[0].grid(False)
#
# clip = 1e-0#显示范围，负值越大越明显
# axs[1].imshow(fx, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
# # noisy_psnr = psnr(data,data_test)
# # noisy_psnr=round(noisy_psnr, 2)
# # axs[1].set_title('Noisy, psnr='+ str(noisy_psnr))
# # axs[1].grid(False)
#
# clip = 1e-0#显示范围，负值越大越明显
# vmin, vmax = -clip, clip
# axs[2].imshow(fp, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
# Denoised_psnr = psnr(data,data_cons)
# Denoised_psnr=round(Denoised_psnr, 2)
# axs[2].set_title('Denoised proposed, psnr='+ str(Denoised_psnr))
# axs[2].grid(False)
########################################################################################################################
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


