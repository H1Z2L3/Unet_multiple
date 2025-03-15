import time
from numpy.random import seed
seed(12345)
from tensorflow import set_random_seed
set_random_seed(1234)
import os
import random
import numpy as np
#from ...import 
import skimage
import matplotlib.pyplot as plt
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, TensorBoard
from keras import backend as keras
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
            #   'dim': (480, 256),
            'dim': (488, 248),
          'n_channels':1,
          'shuffle': True}
  ####data size#####
  ################
  #  seismPathT = "./data256-480/train/seis/"
  #   faultPathT = "./data256-480/train/fault/"
  #   seismPathV = "./data256-480/validation/seis/"
  #  faultPathV = "./data256-480/validation/fault/"
    seismPathT = "./data248-488/train/seis20/"
    faultPathT = "./data248-488/train/fault20/"
    seismPathV = "./data248-488/validation/seis/"
    faultPathV = "./data248-488/validation/fault/"
  #######train/validation--input/fault#########

  ########number of train/validation data########
  train_ID = range(20)
  valid_ID = range(10)
 
  train_generator = DataGenerator(dpath=seismPathT,fpath=faultPathT,
                                  data_IDs=train_ID,**params)
  valid_generator = DataGenerator(dpath=seismPathV,fpath=faultPathV,
                                  data_IDs=valid_ID,**params)
  model = unet4(input_size=(None, None, 1))
  #Import a neural network#
  # model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy',
  #               metrics=['accuracy'])
  model.compile(optimizer=Adam(lr=1e-3), loss='mean_squared_error')
  ######Define the loss function and optimizer######
  # model.compile(optimizer=Adam(lr=1e-4), loss='categorical crossentropy',
  #               metrics=['accuracy'])
  # model.compile(optimizer=Adam(lr=1e-4), loss='mean_squared_logarithmic_error',
  #               metrics=['accuracy'])
  #########################回归问题不用准确率来度量模型好坏###################################################################
  # model.compile(optimizer=Adam(lr=1e-4), loss='mean_absolute_error')
  #Specify the optimizer, loss function, and accuracy metric when configuring the training method.#
  model.summary()
 

  # checkpoint
  # filepath="check7.0398（200）/fseg-{epoch:02d}.hdf5"

  # filepath = "check56-696/fseg-{epoch:02d}.hdf5"
  filepath = "check248-488/fseg-{epoch:02d}.hdf5"
  ###########
  checkpoint = ModelCheckpoint(filepath, monitor='val_loss',
        verbose=1, save_best_only=True, mode='auto',period=30)
  ############
  # checkpoint = ModelCheckpoint(filepath,verbose=1, save_best_only=False, mode='max')
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
  model.save('check248-488/fseg.hdf5')#################################################################################
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

