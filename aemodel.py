import copy
from keras import backend as K
from keras import objectives
from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras.layers.core import Dense, Activation, Flatten, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import GRU
from keras.layers.convolutional import Convolution1D

class MoleculeVAE():
    autoencoder = None
    def create(self,
               max_length=80,
               latent_rep_size=60,
               weights_file=None):
        x = Input(shape=(max_length,))
        z = self._buildEncoder(x, latent_rep_size, max_length)
        self.encoder = Model(x, z)

        encoded_input = Input(shape=(latent_rep_size,))
        self.decoder = Model(
            encoded_input,
            self._buildDecoder(
                encoded_input,
                latent_rep_size,
                max_length,
            )
        )
        x1 = Input(shape=(max_length,))
        z1 = self._buildEncoder(x1, latent_rep_size, max_length)
        self.autoencoder = Model(
            x1,
            self._buildDecoder(
                z1,
                latent_rep_size,
                max_length,
            )
        )

        if weights_file:
            self.autoencoder.load_weights(weights_file)
            self.encoder.load_weights(weights_file, by_name = True)
            self.decoder.load_weights(weights_file, by_name = True)

        self.autoencoder.compile(optimizer = 'Adam',
                                 loss = 'mae',
                                 metrics = ['accuracy'])

    def _buildEncoder(self, x, latent_rep_size, max_length,):
        h = Dense(max_length, activation='relu', name='dense_0')(x)
        h = Dense(512, activation = 'relu', name='dense_1')(h)
        h = Dense(256, activation='relu', name='dense_2')(h)
        h = Dense(128, activation='relu', name='dense_3')(h)
        return Dense(latent_rep_size, activation='relu', name='dense_9')(h)

    def _buildDecoder(self, z, latent_rep_size, max_length):
        h = Dense(latent_rep_size, name='latent_input', activation='relu')(z)
        h = Dense(128, activation='relu', name='dense_4')(h)
        h = Dense(256, activation='relu', name='dense_5')(h)
        h = Dense(512, activation='relu', name='dense_6')(h)
        h = Dense(128, activation='relu', name='dense_7')(h)
        return Dense(max_length, activation='relu', name='dense_8')(h)

    def save(self, filename):
        self.autoencoder.save_weights(filename)
    
    def load(self, weights_file, latent_rep_size = 60):
        self.create(weights_file = weights_file, latent_rep_size = latent_rep_size)
