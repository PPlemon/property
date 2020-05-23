import os
import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn.model_selection import train_test_split
from molecules.vaemodel import VAE
from molecules.util import basevector
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

batch_size = 300
latent_dim = 60
epochs = 1000
def main():
    data = pandas.read_hdf('data/smiles-big.h5', 'table')
    keys = data['structure'].map(len) < 61
    data = data[keys]
    smiles = data['structure'][:]
    # smiles = data['smiles'][:]
    # logp = data['logp'][:]
    # qed = data['qed'][:]
    # sas = data['sas'][:]
    train = []
    test = []
    train_idx, test_idx = map(np.array, train_test_split(smiles.index, test_size=0.20))
    for smile in smiles[train_idx]:
        train.append(basevector(smile))
    for smile in smiles[test_idx]:
        test.append(basevector(smile))
    data_train = np.array(train)
    data_test = np.array(test)
    print(data_train.shape)
    print(data_test.shape)
    model = VAE()
    if os.path.isfile('data/ae_model0'):
        model.load('smiles_model', latent_size=latent_dim)
    else:
        model.create(latent_size=latent_dim)
    check_pointer = ModelCheckpoint(filepath='smiles_model', verbose=1, save_best_only=True)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=2)
    print(data_train[0])
    history = model.auto_encoder.fit(
        data_train,
        data_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[check_pointer, reduce_lr, early_stopping],
        validation_data=(data_test, data_test)
    )
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = history.epoch

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
