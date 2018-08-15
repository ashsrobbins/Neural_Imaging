from keras.layers import Input, Dense
from keras.models import Model

from keras.datasets import mnist

import numpy as np
import matplotlib.pyplot as plt

encoding_dim = 500
filename = 'Cluster_Data/clust1.npz'


npz = np.load(filename)

imported_data = npz[npz.files[0]]

x_dict = imported_data[()]

data_len = len(x_dict[1])
data_shape = (data_len,)



#Input Placeholder 
input_freqs = Input(shape=data_shape)

#Encoded representation of input
encoded = Dense(encoding_dim, activation='relu')(input_freqs)

decoded = Dense(data_len, activation='sigmoid')(encoded)

autoencoder = Model(input_freqs, decoded)

#Separate encoder model, mapping to encoded rep
encoder = Model(input_freqs, encoded)

#Decoder model
encoded_input = Input(shape=(encoding_dim,))

#Creates decoder layer from last layer of auto model
decoder_layer = autoencoder.layers[-1]

decoder = Model(encoded_input,decoder_layer(encoded_input))


autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


# (x_train, _), (x_test, _) = mnist.load_data()

x_train = np.array(list(x_dict.values()))
#Shuffle for fun
np.random.shuffle(x_train)

amax = np.amax(x_train)
x_train = x_train.real/amax


#Recreate training data
x_train_rec = np.zeros(x_train.shape)

for i in range(len(x_train)):
  x_train_rec[i] = np.fft.ifft(x_train[i])

#Split into train and test  
x_test = x_train[-10:]
x_train = x_train[:-10]

x_test_rec = x_train_rec[-10:]
x_train_rec = x_train_rec[:-10]



print('Shape of data', x_train.shape)



# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
# print x_train.shape
# print x_test.shape




autoencoder.fit(x_train, x_train,
                epochs=1000,
                batch_size=512,
                shuffle=True,
                validation_data=(x_test, x_test))
                
                
                

encoded_freqs = encoder.predict(x_test)
decoded_freqs = decoder.predict(encoded_freqs)
                
                
import matplotlib.pyplot as plt

n = 5  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.plot(range(3000),x_test_rec[i])
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    dec_rec = np.fft.ifft(decoded_freqs[i])
    dec_rec[0] = dec_rec[1]
    plt.plot(range(3000),dec_rec)
    
plt.show()
