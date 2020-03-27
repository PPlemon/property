import keras
from keras.models import Sequential
from sklearn.decomposition import PCA
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import h5py
import numpy as np
import os
import pickle
import pandas
import zlib
import struct
import random

latent_rep_size=292
batch_size =128
epochs =1000
max_length=120

# 获取ae预处理数据
def load_dataset(filename, split = True):
    print(filename)
    print(os.path.isfile(filename))
    h5f = h5py.File(filename, 'r')
    if split:
        data_train = h5f['data_train'][:]
        property_train = h5f['property_train'][:]
    else:
        data_train = None
    data_test = h5f['data_test'][:]
    property_test = h5f['property_test'][:]
    charset = h5f['charset'][:]
    h5f.close()
    if split:
        return (data_train, data_test, property_train, property_test, charset)
    else:
        return (data_test, property_test, charset)

# 获取ae编码后数据
def load_property(filename, split = True):
    h5f = h5py.File(filename, 'r')
    data = h5f['latent_vectors'][:]
    property = h5f['property'][:]
    data_train, data_test, property_train, property_test = train_test_split(data, property, test_size=0.20)
    # if split:
    #     data_train = data[train_idx]
    #     property_train = property[train_idx]
    # else:
    #     data_train = None
    # data_test = data[test_idx]
    # property_test = property[test_idx]
    charset = h5f['charset'][:]
    h5f.close()
    if split:
        return (data_train, data_test, property_train, property_test, charset)
    else:
        return (data_test, property_test, charset)

#zlib压缩
def zl(smile):
    num = []
    smiles = smile.encode()
    compressed = zlib.compress(smiles).ljust(72)
    # while compressed:
    #     s = struct.unpack('B', compressed[-1:])[0]/250
    #     compressed = compressed[:-1]
    #     num.append(s)
    for c in compressed:
        c = round(c / 250, 1)
        if c > 1:
            c = round(random.random(), 1)
        num.append(c)
    return num

#不压缩
def zs(smile):
    num = []
    smiles = smile.encode().ljust(100)
    # compressed = zlib.compress(smiles).ljust(72)
    # while smiles:
    #     s = struct.unpack('B', smiles[-1:])[0]/100
    #     smiles = smiles[:-1]
    #     num.append(s)
    for c in smiles:
        c = round(c/100, 1)
        if c > 1:
            c = round(random.random(), 1)
        num.append(c)
    return num

# 获取smiles和属性值
data = pandas.read_hdf('data/smiles-end.h5', 'table')
smiles = data['structure'][:]
logp = data['logp'][:]
# 划分数据编号
train_idx, test_idx = map(np.array, train_test_split(smiles.index, test_size = 0.20))
train = []
test = []
property_train = logp[train_idx]
property_test = logp[test_idx]

# 获取压缩或不压缩数据
for smile in smiles[train_idx]:
    train.append(zl(smile))
for smile in smiles[test_idx]:
    test.append(zl(smile))
data_train = np.array(train)
data_test = np.array(test)

# 获取ae预处理数据
# data_train, data_test, property_train, property_test, charset = load_dataset('data/processed-big1.h5')
# 获取ae编码后数据
# data_train, data_test, property_train, property_test, charset = load_property('data/encoded.h5')

# 主成分分析降维
# pca = PCA(n_components = 80)
# data_train = pca.fit_transform(data_train)
# data_test = pca.fit_transform(data_test)


# 定义输入维度
# input_shape = (max_length, len(charset))
# input_shape = (latent_rep_size,)
input_shape = (72,)

# 回调函数
checkpointer = ModelCheckpoint(filepath = 'model-zl.h5',
                                   verbose = 1,
                                   save_best_only = True)

reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                                  factor = 0.2,
                                  patience = 3,
                                  min_lr = 0.0001)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=2)

# 模型
model = Sequential()
# model.add(Flatten(name='flatten_1'))
model.add(Dense(120, activation='linear', input_shape=input_shape))
# model.add(Dense(60, activation='linear'))
model.add(Dense(60, activation='linear'))
model.add(Dense(30, activation='linear'))
model.add(Dense(1, activation='linear'))


# build模型
# model.build((None, max_length, charset_length))
model.build((None, 72))
# model.build((None, latent_rep_size))

model.summary()

# 编译模型
model.compile(loss='mae', optimizer='Adam', metrics=['accuracy'])

# 训练模型
history = model.fit(data_train, property_train,

                            batch_size=batch_size,

                            epochs=epochs,

                            verbose=1,

                            callbacks = [checkpointer, reduce_lr, early_stopping],

                            validation_data=(data_test, property_test))

# 验证模型
score = model.evaluate(data_test, property_test, verbose=0)
print('Test loss', score[0])
print('Test accuracy', score[1])

# 预测
Y_predict = model.predict(data_test)
print(property_test)
print('property', Y_predict)

# plt.plot(test_y.'b-')
#
# plt.plot(Y_predict,'r--')
