# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 09:01:52 2021

@author: CHRISTIAN
"""

import os
if True:
    fdir = 'D:/CreateurENtreprise/Florentin_thanks-boss/blob'
    os.chdir(fdir)
    print(os.getcwd())


import glob
import numpy as np
import pandas as pd
import cv2

import numpy as np        



'''
les imports de librairie
backend  : boite à outils sur les matrices et les couches keras
Layer: la classe des couche keras
Sequential pour la class du model à créer
Dense une couche de neurones classique, complètement connectée avec la couche précédente et la couche suivante
'''

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Activation, Dropout
from sklearn.model_selection import train_test_split

class MonComposant(Layer):
    '''ma class layer personalisée '''
    def __init__(self, output_dim, **kwargs):
        ''' initialisation, avec les dimensions d'entrée et sortie '''
        self.output_dim = output_dim
        super(MonComposant, self).__init__(**kwargs)
        
    def build(self, input_shape):
        ''' construction/définition des poids du layer '''        
        self.kernel = self.add_weight(name='kernel', shape=(input_shape[1], self.output_dim), initializer='normal', trainable=True)
        super(MonComposant, self).build(input_shape)
        
    def call(self, input_data):
        ''' traitement proprement dit du layer '''
        ''' renvoie le produit des matrices input_data et  kernel
            A(m,n) x B(n,p) = C(m,p) 
        '''
        
        return K.dot(input_data, self.kernel)
    
    def compute_output_shape(self, input_shape):
        ''' calcul du format de sortie de la couche '''          
        return (input_shape[0], self.output_dim)
    

'''
exemple 
https://machinelearningmastery.com/5-step-life-cycle-neural-network-models-keras/
''''
def creation_model():
    ''' création du model '''
    model = Sequential()
    
    model.add(Input(shape=(8,)))
    model.add(MonComposant(64, input_shape=(8,), name='mylayer')) # 'mylayer' sera le nom de notre couche cachée
    model.add(Activation('relu'))    
    model.add(Dropout(0.15))
    model.add(MonComposant(1, input_shape=(64,)))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    tf.keras.utils.plot_model(model, to_file='MonComposant.png', show_shapes=True)
    return model

    
def test_MonComposant(model):
    ''' entrainement et du model '''
    
    # dataset https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv
    dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
    X = dataset[:,0:8]
    Y = dataset[:,8]
    print(X.shape, Y.shape)
    # input_dim = X.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    
    model.summary()

    print()
    print('Analyse de la couche cachée "mylayer"')
    mylayer = model.get_layer('mylayer')
    print('-\tmylayer.kernel.shape:  ', mylayer.kernel.shape)
    print('-\tcompute_output_shape():', mylayer.compute_output_shape((8,)))
    print()
            
    # fit the network
    print('- Entrainement du model')
    history = model.fit(X_train, y_train, epochs=200, batch_size=10, verbose=0)
    # evaluate the network
    loss, accuracy = model.evaluate(X_train, y_train)
    print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
    probabilities = model.predict(X_test)    
    predictions = [float(np.round(p)) for p in probabilities]
    accuracy = np.mean(predictions == y_test)
    print("Prediction Accuracy: %.2f%%" % (accuracy*100))
    
    
    print('-\tpremier element du kernel:')
    print('-\t\t avant fit')
    print(np.matrix(kernel_avant_fit[0][0]))

    print('-\t\t après fit=', np.matrix(mylayer.kernel[0]))
           
    # make predictions
            

def test_exemple():
    # prepare sequence
    X = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # create model
    model = Sequential()
    model.add(Dense(2, input_dim=1))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'mape'])
    # train model
    history = model.fit(X, X, epochs=500, batch_size=len(X), verbose=2)
    pred = model.predict(X)
    
    model.summary()
    # to_file File name of the plot image.
