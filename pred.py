import numpy as np
from keras import optimizers
import pickle
from sklearn.model_selection import train_test_split
from keras.models import Sequential ,Model
from keras.layers import Dense, LSTM, Dropout,Input
import os
from keras.models import load_model
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras import backend as K
from keras.layers.core import Dense, RepeatVector
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from layers import AttentionLayer
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
def Normalize(list):
    list = np.array(list)
    low, high = np.percentile(list, [0, 100])
    delta = high - low
    if delta != 0:
        for i in range(0, len(list)):
            list[i] = (list[i]-low)/delta
    return  list,low,high
def loss(y_true, y_pred):
    return K.mean(K.square(y_pred-y_true),axis=-1)*1000

def acc_score(y_ture,y_pred):
    R2=1-tf.reduce_sum(tf.pow(y_ture-y_pred,2))/tf.reduce_sum(tf.pow(y_ture-tf.reduce_mean(y_pred),2))
    acc=tf.reduce_mean(tf.cast(R2,tf.float32))
    return acc
def FNoramlize(list,low,high):
    delta = high - low
    if delta != 0:
        for i in range(0, len(list)):
            list[i] = list[i]*delta + low
def load_model(model_path):
    model_path = os.path.join(os.getcwd(), model_path)
    model = load_model(model_path,
                       custom_objects={'AttentionLayer': AttentionLayer, "loss": loss, "acc_score": acc_score})
    return model
def read_pickle(data_path):
    pkl_path = os.path.join(os.getcwd(), data_path)
    pkl_file = open(pkl_path, 'rb')
    x = pickle.load(pkl_file)
    x, x_train_low, x_train_high = Normalize(x)
    return  x, x_train_low, x_train_high
os.environ["CUDA_VISIBLE_DEVICES"] = '1' #use GPU with ID=1
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True #allocate dynamically

model = load_model(args.model_path)

S1_data = read_pickle(args.S1_pre_path)
#S2_data = read_pickle(args.S2_pre_path)

vv=S1_data[:,:,0]
vh=S1_data[:,:,1]

vv=vv.reshape(vv.shape[0],1,vv.shape[1],1)
vh=vh.reshape(vh.shape[0],1,vh.shape[1],1)
pred=model.predict([vv,vh])

y_pickle=open(args.pred_path)
pickle.dump(pred,y_pickle)
y_pickle.close()
