from keras import backend as K
from keras import objectives
from keras.models import Model
from keras.layers import Input, Lambda, Dense

class VAE():
    auto_encoder = None
    latent = 60
    length = 80

    def create(self, length=length, latent_size=latent, weights_file=None):

        epsilon_std = 0.01

        def sampling(args):
            z_mean_, z_log_var_ = args
            batch_size = K.shape(z_mean_)[0]
            epsilon = K.random_normal(shape=(batch_size, latent_size), mean=0., stddev = epsilon_std)
            return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

        x = Input(shape=(length,))
        z_mean, z_log_var = self.build_Encoder(x, latent_size, length)
        z = Lambda(sampling, output_shape=(latent_size,), name='lambda')([z_mean, z_log_var])

        self.encoder = Model(x, z)

        encoded_input = Input(shape=(latent_size,))

        self.decoder = Model(
            encoded_input,
            self.build_Decoder(
                encoded_input,
                latent_size,
                length,
            )
        )
        x1 = Input(shape=(length,))
        z_mean, z_log_var = self.build_Encoder(x1, latent_size, length)
        z1 = Lambda(sampling, output_shape=(latent_size,), name='lambda')([z_mean, z_log_var])
        self.auto_encoder = Model(
            x1,
            self.build_Decoder(
                z1,
                latent_size,
                length,
            )
        )

        def vae_loss(x, x_decoded):
            xent_loss = length * objectives.binary_crossentropy(x, x_decoded)
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return xent_loss + kl_loss

        if weights_file:
            self.auto_encoder.load_weights(weights_file)
            self.encoder.load_weights(weights_file, by_name=True)
            self.decoder.load_weights(weights_file, by_name=True)

        self.auto_encoder.compile(optimizer='Adam', loss=vae_loss, metrics=['accuracy'])

    def build_Encoder(self, x, latent_size, length):
        h = Dense(length, activation='relu', name='dense_0')(x)
        h = Dense(512, activation='relu', name='dense_1')(h)
        h = Dense(256, activation='relu', name='dense_2')(h)
        h = Dense(128, activation='relu', name='dense_3')(h)
        z_mean = Dense(latent_size, name='z_mean', activation='relu')(h)
        z_log_var = Dense(latent_size, name='z_log_var', activation='relu')(h)
        return (z_mean, z_log_var)

    def build_Decoder(self, z, latent_size, length):
        h = Dense(latent_size, name='latent_input', activation='relu')(z)
        h = Dense(128, activation='relu', name='dense_4')(h)
        h = Dense(256, activation='relu', name='dense_5')(h)
        h = Dense(512, activation='relu', name='dense_6')(h)
        h = Dense(128, activation='relu', name='dense_7')(h)
        return Dense(length, activation='relu', name='dense_8')(h)

    def save(self, filename):
        self.auto_encoder.save_weights(filename)
    
    def load(self, weights_file, latent_size=latent):
        self.create(weights_file=weights_file, latent_size=latent_size)
