# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 17:52:34 2017

@author: mael
"""


################################################################## IMPORT


#import os    
#os.environ['THEANO_FLAGS'] = "device=cuda,floatX=float32" 
#import theano

import numpy as np
np.random.seed(123) # for reproducibility
from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten # core layers
from keras.layers import convolutional, pooling # CNN layers
from keras.utils import np_utils

from keras.datasets import mnist

################################################################## UTILITY

def setInDic(dic, path, value):
    """Path format: ["key1", 1, "key2"]. '1' means 2nd element of list."""
    if path == []:
        return
    if len(path) == 1:
        dic[path[0]] = value
    else:
        setInDic(dic[path[0]], path[1:], value)
    

##################################################################


"""
Config dictionnary:
------------------------------------------------------------
    
{
    'sequential' : // Sequential config
    'compilation' : {
        'loss' : // loss method,
        'optimizer' : // wich optimizer,
        'metrics' : // metrics to use
    },
    'training' : {
        'epochs' : // nb of epochs,
        'batch_size' : // batch size,
        'nb_samples' : // number of samples
    }
}
    
------------------------------------------------------------
"""

class TestMaker:
    """Build a Sequential from the 'config.sequential', trains it on the 'training' data, 
    and evaluates it on the 'evaluating' data."""
    def __init__(self, config, training, evaluating):
        self.config = config
        self.model = Sequential.from_config(self.config['sequential'])
        self.model.compile(loss      = self.config['compilation']['loss'],
                           optimizer = self.config['compilation']['optimizer'],
                           metrics   = self.config['compilation']['metrics'])
        self.training = training
        self.evaluating = evaluating
        self.config = config
        
    def run(self):
        """Run the test, and returns info"""
        nb = self.config['training']['nb_samples']
        trainingHistory = self.model.fit(self.training['x'][0:nb], self.training['y'][0:nb],
                                    batch_size = self.config['training']['batch_size'], 
                                    epochs     = self.config['training']['epochs'], 
                                    verbose    = 1)
        evaluatingScore = self.model.evaluate(self.evaluating['x'], self.evaluating['y'],
                                              verbose = 1)
        return {'trainingHistory':trainingHistory,
                'evaluatingScore':evaluatingScore}
  
class VaryingDicField:
    """Store a path in a recursive dictionnary, and a range of value to take"""
    def __init__(self, path, values):
        self.path = path
        self.values = values
        
    def __str__(self):
        return ".".join([str(i) for i in self.path]) + ' = ' + str(self.values)
    
    def _repr__(self):
        return "VaryingDicField(" + ".".join([str(i) for i in self.path]) + ', ' + str(self.values) + ")"
    
    def __eq__(self, other):
        return self.path == other.path and self.values == other.values
      
class TestsMaker:
    """Runs multiple tests, by varying the given parameters"""
    def __init__(self, config, varying, training, evaluating):
        self.config = config
        self.varying = varying
        self.training = training
        self.evaluating = evaluating
        self.results = []
        
    def recVaryingTesting(self, varying, configuration):
        if varying == []:
            test = TestMaker(self.config, self.training, self.evaluating)
            self.results.append((configuration, test.run()))
            return
        for value in varying[0].values:
            setInDic(self.config, varying[0].path, value)
            self.recVaryingTesting(varying[1:],
                                   configuration + [VaryingDicField(varying[0].path, value)])
            
    def run(self):
        self.recVaryingTesting(self.varying, [])
        return self.results
    
    def getVaryingByPath(self, path):
        for v in self.varying:
            if v.path == path:
                return v
    
    def plotResult(self, result):
        plt.figure()
        for metric in result[1]["trainingHistory"].history:
            Y = result[1]["trainingHistory"].history[metric]
            plt.plot(Y, 'x-', label = metric)
        plt.xlabel("Iterations")
        configToShow =  " ".join([str(param) for param in result[0]])
        configToShow += '\nEVALUATION SCORE:'
        configToShow += ' Loss: ' + str(result[1]['evaluatingScore'][0])
        for i in range(len(self.config["compilation"]["metrics"])):
            configToShow += ' ' + self.config["compilation"]["metrics"][i] + ': ' + str(result[1]['evaluatingScore'][i+1])
        plt.title(configToShow, wrap=True)
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def plotMinMidMaxResults(self):
        """Plots all combination with minimum, middle, and maximum values of
        varying parameters."""
        # testing every results
        for result in self.results:
            # checking every paramater value
            keep = True
            for param in result[0]:
                values = self.getVaryingByPath(param.path).values
                min_mid_max = [values[0], values[len(values) // 2], values[-1]]
                # if value doesn't correspond, continue to next result
                if param.values not in min_mid_max:
                    keep = False
                    break
            if keep:
                self.plotResult(result)
                    

    
    def plot2DEvaluationMetrics(self, x_param, configuration):
        """Plots the metrics relatively with x_param.
        Other parameter values can be specified in configuration,
        if there are not, the default values is the middle values in the varying values lists."""
        # complete configuration with middle values
#        print("configuration:")
#        print(configuration)
        for parameter in self.varying:
            # check if it is x_param
#            print('bouh1')
#            print(parameter.path)
#            print(x_param)
#            print('bouh2')
            if parameter.path == x_param:
#                print("equals")
                continue
            # check if parameter is already on configuration
            # TODO Speed up this terrible loop
            inConfiguration = False
            for p in configuration:
                if p.path == parameter.path:
                    inConfiguration = True
                    break
            if not inConfiguration:
                configuration.append(VaryingDicField(parameter.path,
                                                     parameter.values[len(parameter.values) // 2]))
#        print("configuration:")
#        print(configuration)
        # grab results coresponding to the configuration
        goodResults = []
        for result in self.results:
            configResult = result[0]
            # result is matching criterias until we found a non matching parameter
            match = True
            for parameter in configuration:
                if parameter not in configResult:
#                    print(parameter)
#                    print(configResult)
                    match = False
                    break
#            print("match: ", match)
            # if result have to be plotted
            if match:
                # get value of x_param in result
                for param in result[0]:
                    if x_param == param.path:
                        goodResults.append((param.values, result[1]['evaluatingScore']))
                        break
#        print(goodResults)
        # finally we can plot the data
        plt.figure()
        X = [d[0] for d in goodResults]
        Y = [d[1][0] for d in goodResults]
        plt.plot(X, Y, "x-", label="Loss")
        for i in range(len(self.config['compilation']['metrics'])):
            Y = [d[1][i+1] for d in goodResults]
            plt.plot(X, Y, "x-", label=self.config['compilation']['metrics'][i])
        plt.xlabel(".".join(str(i) for i in x_param))
        configToShow =  "\n".join([str(param) for param in configuration])
        plt.title(configToShow)
        plt.legend()
        plt.grid(True)
        plt.show()
                    
            
        


################################################################## PARAMETERS

NB_SAMPLE = 500 # multiple of 32
NB_EPOCH = 10
BATCH_SIZE = 32
NB_CONV_1 = 32
KERNEL_SIZE_CONV_1 = (3,3)
NB_CONV_2 = 32
KERNEL_SIZE_CONV_2 = (3,3)
POOL_SIZE = (2, 2)
DROPOUT_1 = 0.25
DROPOUT_2 = 0.5
DENSE_SIZE_1 = 128
DENSE_SIZE_2 = 10

################################################################## PREPARE DATA

# Load pre-shuffled MNIST data into train and tests sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train
y_train = y_train
x_test = x_test[0:NB_SAMPLE]
y_test = y_test[0:NB_SAMPLE]

#for i in range(5):
#    plt.imshow(X_train[i])

X_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
X_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Convert 1-dimensional class arrays to 10-dimensional class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

################################################################## BUILD NETWORK


# 7. Define model architecture
model = Sequential()
 
model.add(convolutional.Conv2D(NB_CONV_1, KERNEL_SIZE_CONV_1, activation='relu', input_shape=(28,28,1)))
model.add(convolutional.Conv2D(NB_CONV_2, KERNEL_SIZE_CONV_2, activation='relu'))
model.add(pooling.MaxPooling2D(pool_size=POOL_SIZE))
model.add(Dropout(DROPOUT_1))
 
model.add(Flatten())
model.add(Dense(DENSE_SIZE_1, activation='relu'))
model.add(Dropout(DROPOUT_2))
model.add(Dense(DENSE_SIZE_2, activation='softmax'))

config = {
            'sequential':model.get_config(),
            'compilation': {
                'loss' : 'categorical_crossentropy',
                'optimizer' : 'adam',
                'metrics' : ['accuracy']
            },
            'training': {
                'epochs' : 10,
                'batch_size' : 32,
                'nb_samples' : 500
            }
         }
           
tests = TestsMaker(config, [VaryingDicField(['training', 'nb_samples'], [500,5000]),
                            VaryingDicField(['sequential', 0, 'kernel_size'], [(2,2),(5,5),(10,10)]),
                            VaryingDicField(['sequential', 0, 'filters'], [5,15,32])],
                   {'x':X_train, 'y':Y_train},
                   {'x':X_test, 'y':Y_test})

################################################################## TRAINING & EVUALATION

tests.run()

#tests.plot2DEvaluationMetrics(['training', 'nb_samples'], [])
#tests.plot2DEvaluationMetrics(['sequential', 0, 'filters'], [])
#tests.plotResult(tests.results[0])
tests.plot2DEvaluationMetrics(['sequential', 0, 'kernel_size'], [])
tests.plotMinMidMaxResults()

#testResult = test.run()
#history = testResult['trainingHistory']

# Plotting
#epochs = np.linspace(1,NB_EPOCH,NB_EPOCH)
#plt.plot(epochs, history.history["acc"], "g") 
#plt.plot(epochs, history.history["loss"], "r", label="Loss")
#plt.xlabel("epoch")
#
#print("Loss: ", testResult['evaluatingScore'])