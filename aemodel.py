from keras.models import Model
from keras.layers import Input, Dense

class VAE():
    auto_encoder = None

    def create(self, max_length=80, latent_size=60, weights_file=None):

        x = Input(shape=(max_length,))
        z = self.build_Encoder(x, latent_size, max_length)
        self.encoder = Model(x, z)

        encoded_input = Input(shape=(latent_size,))
        self.decoder = Model(
            encoded_input,
            self.build_Decoder(
                encoded_input,
                latent_size,
                max_length,
            )
        )
        x1 = Input(shape=(max_length,))
        z1 = self.build_Encoder(x1, latent_size, max_length)
        self.auto_encoder = Model(
            x1,
            self.build_Decoder(
                z1,
                latent_size,
                max_length,
            )
        )

        if weights_file:
            self.auto_encoder.load_weights(weights_file)
            self.encoder.load_weights(weights_file, by_name = True)
            self.decoder.load_weights(weights_file, by_name = True)

        self.auto_encoder.compile(optimizer='Adam', loss='mae', metrics=['accuracy'])

    def build_Encoder(self, x, latent_size, length,):
        h = Dense(length, activation='relu', name='dense_0')(x)
        h = Dense(512, activation = 'relu', name='dense_1')(h)
        h = Dense(256, activation='relu', name='dense_2')(h)
        h = Dense(128, activation='relu', name='dense_3')(h)
        return Dense(latent_size, activation='relu', name='dense_9')(h)

    def build_Decoder(self, z, latent_size, length):
        h = Dense(latent_size, name='latent_input', activation='relu')(z)
        h = Dense(128, activation='relu', name='dense_4')(h)
        h = Dense(256, activation='relu', name='dense_5')(h)
        h = Dense(512, activation='relu', name='dense_6')(h)
        h = Dense(128, activation='relu', name='dense_7')(h)
        return Dense(length, activation='relu', name='dense_8')(h)

    def save(self, filename):
        self.auto_encoder.save_weights(filename)
    
    def load(self, weights_file, latent_size = 60):
        self.create(weights_file = weights_file, latent_size = latent_size)
