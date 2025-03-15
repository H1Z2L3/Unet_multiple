import numpy as np
import keras
import random
from keras.utils import to_categorical
# from skimage import util

class DataGenerator(keras.utils.Sequence):
  'Generates data for keras'
  def __init__(self,dpath,fpath,data_IDs, batch_size=1, dim=(128,128),
             n_channels=1, shuffle=True):
  # def __init__(self, dpath, fpath, data_IDs, batch_size=1, dim=(64, 64),
  #                n_channels=1, shuffle=True):
  #shuffle--Whether to shuffle the data
    'Initialization'
    self.dim   = dim
    self.dpath = dpath
    self.fpath = fpath
    self.batch_size = batch_size
    self.data_IDs   = data_IDs
    self.n_channels = n_channels
    self.shuffle    = shuffle
    self.on_epoch_end()

  def __len__(self):
    'Denotes the number of batches per epoch'
    return int(np.floor(len(self.data_IDs)/self.batch_size))

  def __getitem__(self, index):
    'Generates one batch of data'
    # Generate indexes of the batch
    bsize = self.batch_size
    indexes = self.indexes[index*bsize:(index+1)*bsize]

    # Find list of IDs
    data_IDs_temp = [self.data_IDs[k] for k in indexes]

    # Generate data
    X, Y = self.__data_generation(data_IDs_temp)

    return X, Y

  def gaussian_noise(self, x):  # Gaussian noise

    mu = 0.1
    sigma = 0.05
    noise = np.random.normal(mu,sigma,x.shape).astype(dtype=np.float32)
    x = x + noise
    # x = cv2.GaussianBlur(x, (9, 9), 5)

    return x

  def on_epoch_end(self):
    'Updates indexes after each epoch'
    self.indexes = np.arange(len(self.data_IDs))
    if self.shuffle == True:
      np.random.shuffle(self.indexes)

  def __data_generation(self, data_IDs_temp):
    'Generates data containing batch_size samples'
    # Initialization
    # filepath3 = "./data7/train2/fault/"
    # filepath4 = "./data7/train/fault1/"

    gx  = np.fromfile(self.dpath+str(data_IDs_temp[0])+'.dat',dtype=np.single)
    fx  = np.fromfile(self.fpath+str(data_IDs_temp[0])+'.dat',dtype=np.single)
    gx = np.reshape(gx,self.dim)
    fx = np.reshape(fx,self.dim)
    ################################
    def gaussian_noise1(x): 

      mu = 0
      sigma = 0.00002
      noise = np.random.normal(mu, sigma, x.shape).astype(dtype=np.float32)
      x = x + noise
      # x = cv2.GaussianBlur(x, (9, 9), 5)

      return x

    def gaussian_noise2(x): 

      mu = 0
      sigma = 0.00001
      noise = np.random.normal(mu, sigma, x.shape).astype(dtype=np.float32)
      x = x + noise
      # x = cv2.GaussianBlur(x, (9, 9), 5)

      return x

    def gaussian_noise3(x):  
      mu = 0
      sigma = 0.00005
      noise = np.random.normal(mu, sigma, x.shape).astype(dtype=np.float32)
      x = x + noise
      # x = cv2.GaussianBlur(x, (9, 9), 5)

      return x

    def gaussian_noise(x,mu,sigma):  
      noise = np.random.normal(mu, sigma, x.shape).astype(dtype=np.float32)
      x = x + noise
      # x = cv2.GaussianBlur(x, (9, 9), 5)

      return x

    # fx1 = -fx
    # gx1 = gx[64:, 64:]
    # fx1 = fx[64:, 64:]
    # gx1.tofile(filepath3 + str(data_IDs_temp[0]) + '.dat')
    # fx1.tofile(filepath4 + str(data_IDs_temp[0]) + '.dat')
    # gmin = np.min(gx)
    # gmax = np.max(gx)
    # gx = gx-gmin
    # gx = gx/(gmax-gmin)
    # # gx = gx*255
    # fmin = np.min(fx)
    # fmax = np.max(fx)
    # fx = fx - fmin
    # fx = fx / (fmax - fmin)
    # fx = fx*255

    #gx = gx*255
    ###########################################################################
    ###############################
    # xm = np.mean(gx)
    # xs = np.std(gx)
    # gx = gx-xm
    # gx = gx/xs
    # ###################
    # Ym = np.mean(fx)
    # Ys = np.std(fx)
    # fx = fx - Ym
    # fx = fx / Ys
    ########################################
    # gx.tofile(filepath3 + str(data_IDs_temp[0]) + '.dat')
    # fx.tofile(filepath4 + str(data_IDs_temp[0]) + '.dat')
    # ##########简单缩放####################
    xs = np.max(abs(gx))
    # xs = np.max(gx)
    # xs1 = np.min(abs(gx))
    # xs = xs - xs1
    gx = gx / xs
    Ys = np.max(abs(fx))
    # Ys = np.max(fx)
    # Ys1 = np.min(abs(Ys))
    # Ys = Ys - Ys1
    fx = fx / Ys
    ######################################
    # xm = np.mean(gx)
    # xs = np.std(gx)
    # gx = gx - xm
    # gx = gx / xs
    # ####
    # xm1 = np.mean(fx)
    # xs1 = np.std(fx)
    # fx = fx - xm1
    # fx = fx / xs1
    #####################################
    # #########################
    # xm = np.mean(gx)
    # gx = gx - xm
    # Ym = np.mean(fx)
    # fx = fx - Ym
    #####################################
    #################################Data Augmentation###########################################
    # gx = np.transpose(gx)
    # fx = np.transpose(fx)
    # fx.tofile(filepath4 + str(data_IDs_temp[0]) + '.dat')
    #in seismic processing, the dimensions of a seismic array is often arranged as
    #a[n3][n2][n1] where n1 represnts the vertical dimenstion. This is why we need
    #to transpose the array here in python
    # Generate data
    X = np.zeros((10, *self.dim, self.n_channels),dtype=np.single)
    Y = np.zeros((10, *self.dim, self.n_channels),dtype=np.single)
    X[0,] = np.reshape(gx, (*self.dim,self.n_channels))
    Y[0,] = np.reshape(fx, (*self.dim,self.n_channels))
    X[1,] = np.reshape(np.flipud(gx), (*self.dim,self.n_channels))
    Y[1,] = np.reshape(np.flipud(fx), (*self.dim,self.n_channels))
    # X[2,] = np.reshape(np.rot90(gx), (*self.dim, self.n_channels))
    # Y[2,] = np.reshape(np.rot90(fx), (*self.dim, self.n_channels))
    X[2,] = np.reshape(np.rot90(gx, 1), (*self.dim, self.n_channels))
    Y[2,] = np.reshape(np.rot90(fx, 1), (*self.dim, self.n_channels))
    X[3,] = np.reshape(np.rot90(gx, 2), (*self.dim, self.n_channels))
    Y[3,] = np.reshape(np.rot90(fx, 2), (*self.dim, self.n_channels))
    X[4,] = np.reshape(np.rot90(gx, 3), (*self.dim, self.n_channels))
    Y[4,] = np.reshape(np.rot90(fx, 3), (*self.dim, self.n_channels))
    X[5,] = np.reshape(np.fliplr(gx), (*self.dim, self.n_channels))
    Y[5,] = np.reshape(np.fliplr(fx), (*self.dim, self.n_channels))
    # X[5,] = np.reshape(np.random.normal(gx, 2), (*self.dim, self.n_channels))
    # Y[5,] = np.reshape(fx, (*self.dim, self.n_channels))
    X[6,] = np.reshape(gaussian_noise1(gx), (*self.dim, self.n_channels))
    Y[6,] = np.reshape(fx, (*self.dim, self.n_channels))
    X[7,] = np.reshape(gaussian_noise2(gx), (*self.dim, self.n_channels))
    Y[7,] = np.reshape(fx, (*self.dim, self.n_channels))
    X[8,] = np.reshape(gaussian_noise3(gx), (*self.dim, self.n_channels))
    Y[8,] = np.reshape(fx, (*self.dim, self.n_channels))
    X[9,] = np.reshape(gaussian_noise(gx,0.01,0.1), (*self.dim, self.n_channels))
    Y[9,] = np.reshape(fx, (*self.dim, self.n_channels))


    '''
    for i in range(4):
      X[i,] = np.reshape(np.rot90(gx,i,(2,1)), (*self.dim,self.n_channels))
      Y[i,] = np.reshape(np.rot90(fx,i,(2,1)), (*self.dim,self.n_channels))  
    '''
    return X,Y
