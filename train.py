
import numpy as np
from keras import optimizers
import pickle
from sklearn.model_selection import train_test_split
from keras.models import Sequential ,Model
from keras.layers import Dense, LSTM, Dropout,Input
import os
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
from configs import args


def Normalize(list):
    list = np.array(list)
    low, high = np.percentile(list, [0, 100])
    delta = high - low
    if delta != 0:
        for i in range(0, len(list)):
            list[i] = (list[i]-low)/delta
    return  list,low,high

def FNoramlize(list,low,high):
    delta = high - low
    if delta != 0:
        for i in range(0, len(list)):
            list[i] = list[i]*delta + low
    return list
def image_to_list(image):  #image.shape = band,height,width,
    row = image.shape[1]
    col = image.shape[2]
    band = image.shape[0]
    image_pixel_all = []
    for i in range(row):
        for j in range(col):
            image_pixel= image[:,i,j]
            image_pixel_all.append(image_pixel)
    image_list = np.stack(image_pixel_all, axis=0)
    return image_list # n(row*col)*seq_length
class GRID:
    def write_img(self, filename, proj, geotrans, data):
    # 判断栅格数据的数据类型
        if('int8' in data.dtype.name):
            datatype = gdal.GDT_Byte
        elif 'int16' in data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        if len(data.shape) == 3:
            bands, lines, samples = data.shape
        else:
            bands, (lines, samples) = 1, data.shape

        driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
        dataset = driver.Create(filename,samples, lines, bands, datatype)
        dataset.SetGeoTransform(geotrans)  # 写入仿射变换参数
        dataset.SetProjection(proj)  # 写入投影

        if bands == 1:
            dataset.GetRasterBand(1).WriteArray(data)  # 写入数组数据
        else:
             for i in range(bands):
                dataset.GetRasterBand(i + 1).WriteArray(data[i])

        del dataset

def loss(y_true, y_pred):

    return K.mean(K.square(y_true-y_pred),axis=-1)*1000

def top_down_acc(y_true, y_pred):
    SSR=K.mean(K.square(y_pred-K.mean(y_true)),axis=-1)
    SST=K.mean(K.square(y_true-K.mean(y_true)),axis=-1)
    return SSR/SST

def acc_score(y_ture,y_pred):
    R2=1-tf.reduce_sum(tf.pow(y_ture-y_pred,2))/tf.reduce_sum(tf.pow(y_ture-tf.reduce_mean(y_pred),2))
    acc=tf.reduce_mean(tf.cast(R2,tf.float32))
    return acc

def read_pickle(path):
    train_x = pickle.load(open(path, 'rb'))
    train_x,x_train_low,x_train_high=Normalize(train_x)
    return train_x

def read_S1_pikle (path,S1_length,S1_feature):
    data = pickle.load(open(path, 'rb'))
    data_Nor, data_low, data_high = Normalize(data)
    X =data_Nor.reshape(data_Nor.shape[0], S1_feature,S1_length )
    X = np.array(X).transpose((0, 2, 1))
    return X, data_low, data_high

os.environ["CUDA_VISIBLE_DEVICES"] = '2' #use GPU with ID=1
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True #allocate dynamically


#s1_train = read_pickle (args.S1_path)
#s2_train = read_pickle (args.S2_path)

#read pickle
s1_train,S1_tr_low, S1_tr_high = read_S1_pikle(args.S1_tr_path,args.lenght,args.feature)
s2_train = pickle.load(open(args.S2_tr_path, 'rb'))
S1_te,S1_te_low, S1_te_high = read_S1_pikle(args.S1_te_path,args.lenght,args.feature)
S2_te = pickle.load(open(args.S2_te_path, 'rb'))

#s1_train = s1_train[:,:,np.newaxis].reshape( s1_train.shape[0],args.s1_band, args.s1_lenght).transpose((0,2, 1))

X_vv=s1_train[:,:,0]
X_vh=s1_train[:,:,1]
X_vv=X_vv.reshape(X_vv.shape[0],1,X_vv.shape[1],1)
X_vh=X_vh.reshape(X_vh.shape[0],1,X_vh.shape[1],1)

output_dim=1
input_dim=1
output_length=s2_train.shape[1]
input_length=s1_train.shape[1]
#1d-cnn
visible1=Input(shape=(1, input_length,input_dim))
cnn1=(TimeDistributed(Conv1D(filters=31, kernel_size=1, activation='relu')))(visible1)
cnn1=(TimeDistributed(Conv1D(filters=62, kernel_size=1, activation='relu')))(cnn1)
cnn1=(TimeDistributed(Flatten()))(cnn1)

visible2=Input(shape=( 1,input_length,input_dim))
cnn2=(TimeDistributed(Conv1D(filters=31, kernel_size=1, activation='relu')))(visible2)
cnn2=(TimeDistributed(Conv1D(filters=62, kernel_size=1, activation='relu')))(cnn2)
cnn2=(TimeDistributed(Flatten()))(cnn2)

merge=concatenate([cnn1,cnn2],axis=2)

lstm=(LSTM(args.hidden_dim,activation='tanh'))(merge)
encoder=(RepeatVector(output_length))(lstm)
#decoder2

decoder1=(LSTM(args.hidden_dim,activation='tanh',return_sequences=True))(encoder)
decoder2=(AttentionLayer(name='attention'))(decoder1)

output=(Dense(output_length,activation='linear'))(decoder2)
model = Model(inputs=[visible1, visible2], outputs=output)
model.compile(optimizer="rmsprop",loss=loss,metrics=[acc_score])
model.summary()
model.fit([X_vv,X_vh],s2_train,epochs=args.epoch,batch_size=args.batch_size,validation_split=args.val)
#model = Sequential()
#model.add(LSTM(hidden_dim,input_shape=(input_length,input_dim), return_sequences=False))
#model.add(Dense(hidden_dim, activation="tanh"))
#model.add(RepeatVector(input_length))
#model.add(LSTM(hidden_dim,input_shape=(input_length,input_dim), return_sequences=False))
#model.add(RepeatVector(output_length))
#model.add(LSTM(hidden_dim, return_sequences=True))
#model.add(TimeDistributed(Dense(output_dim=output_dim)))
#model.summary()
#model.compile(loss=loss, optimizer="rmsprop",metrics=[acc_score])
#model.fit(X_tr,Y_tr,
         # batch_size=1000,
         # epochs=20,
         # validation_split=0.2)

model.save(args.model_name)

