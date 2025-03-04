import time
from numpy.random import seed
seed(12345)
from tensorflow import set_random_seed
set_random_seed(1234)
import os
#在python环境下对文件，文件夹执行操作的模块#
import random
#导入随机数模块儿#
import numpy as np
#导入基础数据处理库#
#from ...import *从模块儿中导入其所含的整个函数#
import skimage
import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, TensorBoard
#从回调函数中导入ModelCheckpoint模型保存，ReduceLROnPlateau自动降低学习率，
# LearningRateScheduler自适应调节学习率，Tensorboard可视化模板#
from keras import backend as keras
#后端模块导入#
from utilsmultiple import DataGenerator
from unet3multiple import *
# from cnnmultiple import *
# from unet2multiple import cross_entropy_balanced
start = time.clock()

def main():
  goTrain()

def goTrain():
  # input image dimensions
  params = {'batch_size':1,
          # 'dim': (1600, 64),
          #   'dim': (256, 64),
          #   'dim': (256, 128),
          #   'dim': (1600, 128),
           #'dim': (128, 128),
            #   'dim': (480, 256),
            'dim': (488, 248),
            #   'dim': (696, 56),
          #    'dim': (200, 256),
          #   'dim': (400, 48),
          # 'dim': (480, 240),
          # 'dim':(192,48),
          # 'dim': (624, 192),
          # 'dim': (496, 96),
          #   'dim': (1536, 64),
          #   'dim': (800, 64),
          # 'dim': (496, 96),
          'n_channels':1,
          'shuffle': True}
  # params = {'batch_size': 1,
  #           'dim': (64, 64),
  #           'n_channels': 1,
  #           'shuffle': True}
  ####端到端#####
  # seismPathT = "./data3/train/seis/"
  # faultPathT = "./data3/train/fault/"
  # seismPathV = "./data3/validation/seis/"
  # faultPathV = "./data3/validation/fault/"
  #############
  # seismPathT = "./data20/train/seis/"
  # faultPathT = "./data20/train/fault/"
  # seismPathV = "./data20/validation/seis/"
  # faultPathV = "./data20/validation/fault/"
  #############
  # seismPathT = "./data25/train2/seis/"
  # faultPathT = "./data25/train2/fault/"
  # seismPathV = "./data25/validation2/seis/"
  # faultPathV = "./data25/validation2/fault/"
  #############
  # seismPathT = "./data1/train/seis/"
  # faultPathT = "./data1/train/fault/"
  # seismPathV = "./data1/validation/seis/"
  # faultPathV = "./data1/validation/fault/"
  #############
  # seismPathT = "./data9/train/seis/"
  # faultPathT = "./data9/train/fault/"
  # seismPathV = "./data9/validation/seis/"
  # faultPathV = "./data9/validation/fault/"
  ################
  #  seismPathT = "./data256-480/train/seis/"
  #   faultPathT = "./data256-480/train/fault/"
  #   seismPathV = "./data256-480/validation/seis/"
  #  faultPathV = "./data256-480/validation/fault/"
    seismPathT = "./data248-488/train/seis20/"
    faultPathT = "./data248-488/train/fault20/"
    seismPathV = "./data248-488/validation/seis/"
    faultPathV = "./data248-488/validation/fault/"
  #seismPathT = "./data56-696/train/seis/"
  #faultPathT = "./data56-696/train/fault/"
  ##seismPathV = "./data56-696/validation/seis/"
  #faultPathV = "./data56-696/validation/fault/"
  ################
  #seismPathT = "./data81-101-696/train/seis/"
  #faultPathT = "./data81-101-696/train/fault/"
  #seismPathV = "./data81-101-696/validation/seis/"
  #faultPathV = "./data81-101-696/validation/fault/"
  ##############

  ###Radon变换#####
  # seismPathT = "./data9/train6/low/"
  # faultPathT = "./data9/train6/high/"
  #
  # seismPathV = "./data9/validation6/low/"
  # faultPathV = "./data9/validation6/high/"

  # seismPathT = "./data15/train/low/"
  # faultPathT = "./data15/train/high/"
  # seismPathV = "./data15/validation/low/"
  # faultPathV = "./data15/validation/high/"

  # seismPathT = "./data17/train/low/"
  # faultPathT = "./data17/train/high/"
  # seismPathV = "./data17/validation/low/"
  # faultPathV = "./data17/validation/high/"

  # seismPathT = "./data21/train/low/"
  # faultPathT = "./data21/train/high/"
  # seismPathV = "./data21/validation/low/"
  # faultPathV = "./data21/validation/high/"

  # seismPathV = "./data14/validation/low/"
  # faultPathV = "./data14/validation/high/"
  #################
  ###images transform#####
  # seismPathT = "./data6/train1/seis/"
  # faultPathT = "./data6/train1/fault/"
  #
  # seismPathV = "./data6/validation1/seis/"
  # faultPathV = "./data6/validation1/fault/"
  ################
  train_ID = range(20)
  valid_ID = range(10)
  # train_ID = range(198)
  #  valid_ID = range(43)
  train_generator = DataGenerator(dpath=seismPathT,fpath=faultPathT,
                                  data_IDs=train_ID,**params)
  valid_generator = DataGenerator(dpath=seismPathV,fpath=faultPathV,
                                  data_IDs=valid_ID,**params)
  model = unet4(input_size=(None, None, 1))
  #导入神经网络#
  # model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy',
  #               metrics=['accuracy'])
  model.compile(optimizer=Adam(lr=1e-3), loss='mean_squared_error')
  ######定义损失函数和优化器######
  # model.compile(optimizer=Adam(lr=1e-4), loss='categorical crossentropy',
  #               metrics=['accuracy'])
  # model.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_logarithmic_error',
  #               metrics=['accuracy'])
  #########################回归问题不用准确率来度量模型好坏###################################################################
  # model.compile(optimizer=Adam(lr=1e-4), loss='mean_absolute_error')
  #配置训练方法时，告知训练时用的优化器、损失函数和准确率评测标准#
  model.summary()
  # 打印神经网络结构，统计参数数目#

  # checkpoint
  # filepath="check7.0398（200）/fseg-{epoch:02d}.hdf5"

  # filepath = "check56-696/fseg-{epoch:02d}.hdf5"
  filepath = "check248-488/fseg-{epoch:02d}.hdf5"
  ###########
  checkpoint = ModelCheckpoint(filepath, monitor='val_loss',
        verbose=1, save_best_only=True, mode='auto',period=30)
  ############
  # checkpoint = ModelCheckpoint(filepath,verbose=1, save_best_only=False, mode='max')
  ############
  #monitor：需要监视的量，通常为：val_acc 或 val_loss 或 acc 或 loss
  #verbose：信息展示模式，0或1。为1表示输出epoch模型保存信息，默认为0表示不输出该信息；
  #save_best_only：当设置为True时，将只保存在验证集上性能最好的模型，当设置为false时，保存所有模型；
  #mode：‘auto’，‘min’，‘max’之一，在save_best_only=True时决定性能最佳模型的评判准则，例如，当监测值为val_acc时，模式应为max，当检测值为val_loss时，模式应为min。
  #period:每个检查点之间的间隔（训练轮数epoch）/可以通过搭配save_best_only设置该值控制模型保存的个数和质量以节省存储空间
  ##############################################################
  ## logging = TrainValTensorBoard()                          ##
  ##############################################################
  # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
  #                              patience=20, min_lr=1e-8)
  ##############################################################
  ##callbacks_list = [checkpoint, logging]                    ##
  ##############################################################
  callbacks_list = [checkpoint]
  print("data prepared, ready to train!")
  # filepath3 = "./data77/111/"
  # filepath4 = "./data7/train2/seis/"
  # train_generator.tofile(filepath3 + str(data_IDs_temp[0]) + '.dat')
  # Fit the model
  history=model.fit_generator(generator=train_generator,
    validation_data=valid_generator,epochs=200,callbacks=callbacks_list,verbose=1)
  ###训练模型的数据传入以及相关参数设定###
  model.save('check81-101-696/fseg.hdf5')#################################################################################
  showHistory(history)


  end = time.clock()
  print('Runing time:%s Minutes' %( (end - start)/60))

def showHistory(history):
  # list all data in history
  print(history.history.keys())
  # fig = plt.figure(figsize=(10,6))

  # summarize history for accuracy
  # plt.plot(history.history['acc'])
  # plt.plot(history.history['val_acc'])
  # plt.title('Model accuracy',fontsize=20)
  # plt.ylabel('Accuracy',fontsize=20)
  # plt.xlabel('Epoch',fontsize=20)
  # plt.legend(['train', 'test'], loc='center right',fontsize=20)
  # plt.tick_params(axis='both', which='major', labelsize=18)
  # plt.tick_params(axis='both', which='minor', labelsize=18)
  # plt.show()
  with open('training_loss.txt', 'w') as f:
    for loss_value in history.history['loss']:
      f.write(str(loss_value) + '\n')
  with open('val_loss.txt', 'w') as f:
    for loss_value in history.history['val_loss']:
      f.write(str(loss_value) + '\n')
  # summarize history for loss
  # fig = plt.figure(figsize=(10,6))
  plt.figure(figsize=(10, 6), dpi=80)
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model loss',fontsize=20)
  plt.ylabel('Loss',fontsize=20)
  plt.xlabel('Epoch',fontsize=20)
  plt.legend(['train', 'valid'], loc='center right',fontsize=20)
  plt.tick_params(axis='both', which='major', labelsize=18)
  plt.tick_params(axis='both', which='minor', labelsize=18)
  plt.show()


#############################################################
  # end = time.clock()
  # print('Runing time:%s Seconds' % (end - start))
#############################################################

########################################################################################################
# class TrainValTensorBoard(TensorBoard):
#     def __init__(self, log_dir='./logd', **kwargs):
#         # Make the original `TensorBoard` log to a subdirectory 'training'
#         training_log_dir = os.path.join(log_dir, 'training')
#         super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)
#         # Log the validation metrics to a separate subdirectory
#         self.val_log_dir = os.path.join(log_dir, 'validation')
#     def set_model(self, model):
#         # Setup writer for validation metrics
#         self.val_writer = tf.summary.FileWriter(self.val_log_dir)
#         super(TrainValTensorBoard, self).set_model(model)
#     def on_epoch_end(self, epoch, logs=None):
#         # Pop the validation logs and handle them separately with
#         # `self.val_writer`. Also rename the keys so that they can
#         # be plotted on the same figure with the training metrics
#         logs = logs or {}
#         val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
#         for name, value in val_logs.items():
#             summary = tf.Summary()
#             summary_value = summary.value.add()
#             summary_value.simple_value = value.item()
#             summary_value.tag = name
#             self.val_writer.add_summary(summary, epoch)
#         self.val_writer.flush()
#         # Pass the remaining logs to `TensorBoard.on_epoch_end`
#         logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
#         logs.update({'lr': keras.eval(self.model.optimizer.lr)})
#         super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)
#     def on_train_end(self, logs=None):
#         super(TrainValTensorBoard, self).on_train_end(logs)
#         self.val_writer.close()
######################################################################################################

if __name__ == '__main__':
    main()

