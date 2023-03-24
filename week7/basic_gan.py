#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numpy.random import rand, randn
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

import os
import warnings
warnings.filterwarnings('ignore')

outputname = 'outputProgress.pdf'
from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages(outputname)


# Generate the actual data that we are targeting. So for example, here say
# f(x) = x^3  is what we want our generator to produce.
def generate_real_samples(n=100):
    #generate inputs in -0.5 to 0.5
    X1 = rand(n) - 0.5
    #generate output as X^3
    X2 = X1 * X1 * X1
    #stack arrays
    X1 = X1.reshape(n,1)
    X2 = X2.reshape(n,1)
    X = np.hstack((X1,X2))
    # we also generate a label for these points as 1
    y = np.ones((n,1))
    return X, y

# ---------------------------------------------------------
# Now we define the GAN : a generator, a discriminator and put them together
#Define a discriminator model
def define_discriminator(n_inputs=2):
    model = Sequential()
    model.add(Dense(16, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
    model.add(Dense(8, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Define a generator
# No compile here because we dont fit generator directly.
def  define_generator(latent_dim, n_outputs=2):
    model = Sequential()
    model.add(Dense(16, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
    model.add(Dense(8, activation='relu', kernel_initializer='he_uniform'))    
    model.add(Dense(n_outputs, activation='linear'))
    return model


# Define the GAN
def define_gan(generator, discriminator):
    # Disc. cannot be trained here.
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model
# ---------------------------------------------------------


# Generate points in latent space as input for generator
def generate_latent_points(latent_dim, n):
    #generate points in the latent space
    x_input = randn(latent_dim * n)
    #reshape into batch of inputs for network
    x_input = x_input.reshape(n, latent_dim)
    return x_input


# Generate examples using generator, label them as fake
def generate_fake_samples(generator, latent_dim, n):
    x_input = generate_latent_points(latent_dim, n)
    X = generator.predict(x_input)
    # we also generate a label for these points as 0
    y = np.zeros((n,1))
    return X, y


# In this function we check how our generator is doing
def summarize_performance(epoch, generator, discriminator, latent_dim, n=100):
    x_real, y_real = generate_real_samples(n)
    _, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)
    x_fake, y_fake = generate_fake_samples(generator, latent_dim, n)
    _, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
    print(f'At {epoch}, RealAcc={acc_real}, FakeAcc={acc_fake}')
    #Make scatter plot of real and fake points
    plt.scatter(x_real[:,0],x_real[:,1], color='red')
    plt.scatter(x_fake[:,0],x_fake[:,1], color='blue')
    plt.title(f'NEpochs = {epoch+1}')
    plt.xlim(-0.65,0.65)
    plt.ylim(-0.15,0.15)
    plt.savefig(pp,format='pdf')
    plt.close()



def train(gen_model, disc_model, gan_model, latent_dim, n_epochs=1000, n_batch=128, n_eval=2000):
    half_batch = int(n_batch / 2)

    for i in range(n_epochs):
        #Generate real points
        x_real, y_real = generate_real_samples(half_batch)
        #Generate points using the gen network
        x_fake, y_fake = generate_fake_samples(gen_model, latent_dim, half_batch)
        # Train the disc
        disc_model.train_on_batch(x_real, y_real)
        disc_model.train_on_batch(x_fake, y_fake)

        #Prepare points for the generator
        x_gan = generate_latent_points(latent_dim, n_batch)
        y_gan = np.ones((n_batch,1))
        #update the gan via the disc error
        gan_model.train_on_batch(x_gan, y_gan)

        if (i+1) % 500 ==0:
            print(f"Done {i} epochs...")
        
        #evaluate it every n_eval epochs
        if i == 100:
            summarize_performance(i, gen_model, disc_model, latent_dim)
        if i == 500:
            summarize_performance(i, gen_model, disc_model, latent_dim)
        if (i+1) % n_eval ==0:
            summarize_performance(i, gen_model, disc_model, latent_dim)


# ============================  ************ ========================================
# Main code starts here

# First we generate the points to visualize what our target it.            
data_example, _ = generate_real_samples(500)
plt.scatter(data_example[:,0], data_example[:,1], color='red')
plt.title('Original function')
plt.xlim(-0.65,0.65)
plt.ylim(-0.15,0.15)
plt.savefig(pp, format='pdf')
plt.close()

# Size of latent space
latent_dim = 5
nepochs = 10000
batchsize = 256


# Define our neural networks
mydisc = define_discriminator()
mygen  = define_generator(latent_dim)
mygan  = define_gan(mygen, mydisc)


# Use the generator before training to see how output looks
data_gen_pre, _ = generate_fake_samples(mygen, latent_dim, 500)
plt.scatter(data_example[:,0], data_example[:,1], color='red')
plt.scatter(data_gen_pre[:,0], data_gen_pre[:,1], color='blue')
plt.title('At start of training')
plt.xlim(-0.65,0.65)
plt.ylim(-0.15,0.15)
plt.savefig(pp, format='pdf')
plt.close()


# Train the generator
train(mygen, mydisc, mygan, latent_dim, nepochs, batchsize, 2000)

# Now use the generator again (after training) to see how output looks
data_gen, _ = generate_fake_samples(mygen, latent_dim, 500)
plt.scatter(data_example[:,0], data_example[:,1], color='red')
plt.scatter(data_gen[:,0], data_gen[:,1], color='blue')
plt.title('At end of training')
plt.xlim(-0.65,0.65)
plt.ylim(-0.15,0.15)
plt.savefig(pp, format='pdf')
plt.close()


pp.close()

mydisc.save('mydiscmod.h5')
mygen.save('mygenmod.h5')
mygan.save('myganmod.h5')

print('All done.')

