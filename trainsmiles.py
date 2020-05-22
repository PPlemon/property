from __future__ import print_function

import argparse
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

NUM_EPOCHS = 1
BATCH_SIZE = 300
LATENT_DIM = 60
RANDOM_SEED = 1337
epochs = 1000
# def get_arguments():
#     parser = argparse.ArgumentParser(description='Molecular autoencoder network')
#     parser.add_argument('data', type=str, help='The HDF5 file containing preprocessed data.')
#     parser.add_argument('model', type=str,
#                         help='Where to save the trained model. If this file exists, it will be opened and resumed.')
#     parser.add_argument('--epochs', type=int, metavar='N', default=NUM_EPOCHS,
#                         help='Number of epochs to run during training.')
#     parser.add_argument('--latent_dim', type=int, metavar='N', default=LATENT_DIM,
#                         help='Dimensionality of the latent representation.')
#     parser.add_argument('--batch_size', type=int, metavar='N', default=BATCH_SIZE,
#                         help='Number of samples to process per minibatch during training.')
#     parser.add_argument('--random_seed', type=int, metavar='N', default=RANDOM_SEED,
#                         help='Seed to use to start randomizer for shuffling.')
#     return parser.parse_args()

def main():
    # args = get_arguments()
    np.random.seed(1337)

    from molecules.aemodel import MoleculeVAE
    from molecules.utils import basevector
    from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

    data = pandas.read_hdf('smiles-big.h5', 'table')
    keys = data['structure'].map(len) < 61
    data = data[keys]
    smiles = data['structure'][:]
    # smiles = data['smiles'][:]
    # logp = data['logp'][:]
    # qed = data['qed'][:]
    # sas = data['sas'][:]
    train = []
    test = []
    # 划分数据编号
    train_idx, test_idx = map(np.array, train_test_split(smiles.index, test_size=0.20))
    for smile in smiles[train_idx]:
        train.append(basevector(smile))
    for smile in smiles[test_idx]:
        test.append(basevector(smile))
    data_train = np.array(train)
    data_test = np.array(test)
    print(data_train.shape)
    print(data_test.shape)
    model = MoleculeVAE()#实例化一个模型
    if os.path.isfile('data/ae_model0'):
        model.load('smiles_model', latent_rep_size = LATENT_DIM)#模型存在加载模型
    else:
        model.create(latent_rep_size = LATENT_DIM)#不存在则新建一个模型

    checkpointer = ModelCheckpoint(filepath = 'smiles_model', #每个epoch后保存模型到指定文件
                                   verbose = 1,
                                   save_best_only = True)

    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',#当评价指标不在提升时，减少学习率
                                  factor = 0.2,
                                  patience = 3,
                                  min_lr = 0.0001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=2)
    print(data_train[0])
    history = model.autoencoder.fit(
        data_train,#输入数据
        data_train,#标签
        shuffle = True,#随机打乱样本
        epochs = epochs,
        batch_size = BATCH_SIZE,#整数，指定进行梯度下降时每个batch包含的样本数。训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步。
        callbacks = [checkpointer, reduce_lr, early_stopping],#回调函数
        validation_data = (data_test, data_test)#验证集
    )
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = history.epoch
    # epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

if __name__ == '__main__':
    main()
