import copy
from keras import backend as K
from keras import objectives
from keras.models import Model
from keras.layers.core import Dense, Activation, Flatten, RepeatVector
from keras.layers import Input, Lambda, Dense, Activation, Flatten, Convolution1D, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import GRU
from keras.layers.convolutional import Convolution1D

class MoleculeVAE():
    autoencoder = None
    def create(self,
               max_length=160,
               latent_rep_size=80,
               weights_file=None):
        epsilon_std = 0.01
        def sampling(args):
            z_mean_, z_log_var_ = args
            batch_size = K.shape(z_mean_)[0]                   #返回张量形状
            epsilon = K.random_normal(shape=(batch_size, latent_rep_size), mean=0., stddev = epsilon_std)
            return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

        x = Input(shape=(max_length,))
        z_mean, z_log_var = self._buildEncoder(x, latent_rep_size, max_length)
        z = Lambda(sampling, output_shape=(latent_rep_size,), name='lambda')([z_mean, z_log_var])

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
        z_mean, z_log_var = self._buildEncoder(x1, latent_rep_size, max_length)
        z1 = Lambda(sampling, output_shape=(latent_rep_size,), name='lambda')([z_mean, z_log_var])
        self.autoencoder = Model(
            x1,
            self._buildDecoder(
                z1,
                latent_rep_size,
                max_length,
            )
        )

        def vae_loss(x, x_decoded_mean):
            xent_loss = max_length * objectives.binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            #K.mean求均值。K.square求平方
            return xent_loss + kl_loss


        if weights_file:
            self.autoencoder.load_weights(weights_file)
            self.encoder.load_weights(weights_file, by_name = True)
            self.decoder.load_weights(weights_file, by_name = True)

        self.autoencoder.compile(optimizer = 'Adam',
                                 loss = vae_loss,
                                 metrics = ['accuracy'])

    def _buildEncoder(self, x, latent_rep_size, max_length):
        h = Dense(max_length, activation='relu', name='dense_0')(x)
        h = Dense(512, activation='relu', name='dense_1')(h)
        h = Dense(256, activation='relu', name='dense_2')(h)
        h = Dense(128, activation='relu', name='dense_3')(h)
        z_mean = Dense(latent_rep_size, name='z_mean', activation='relu')(h)
        z_log_var = Dense(latent_rep_size, name='z_log_var', activation='relu')(h)
        return (z_mean, z_log_var)

    def _buildDecoder(self, z, latent_rep_size, max_length):
        h = Dense(latent_rep_size, name='latent_input', activation='relu')(z)
        h = Dense(128, activation='relu', name='dense_4')(h)
        h = Dense(256, activation='relu', name='dense_5')(h)
        h = Dense(512, activation='relu', name='dense_6')(h)
        h = Dense(128, activation='relu', name='dense_7')(h)
        return Dense(max_length, activation='relu', name='dense_8')(h)

    def save(self, filename):
        self.autoencoder.save_weights(filename)
    
    def load(self, weights_file, latent_rep_size = 80):
        self.create(weights_file = weights_file, latent_rep_size = latent_rep_size)
