import sentencepiece as spm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, multilabel_confusion_matrix, mean_squared_error, r2_score
import tensorflow as tf
import pandas as pd
from glob import glob

import wandb
from wandb.keras import WandbCallback
#from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
#from tensorflow.keras.utils import np_utils



#from tensorflow.keras.optimizers import SGD
#from tensorflow.keras.optimizers import RMSprop, SGD, Adam, Nadam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, Callback, EarlyStopping

wandb.login()
sweep_config = {
    'method': 'bayes', #grid, random
    'metric': {
      'name': 'val_accuracy',
      'goal': 'maximize'   
    },
    'parameters': {
        
        
        'batch_size': {
            'values': [128, 256, 512]
        },
        
        "dropout1" :{
          'distribution': 'uniform',
          "min": 0.001,
          "max": 0.7
        },
        "importance" :{
          'distribution': 'uniform',
          "min": 1.0,
          "max": 3.0
        },
        
        'dims':{
            'values': [1,2,4,6,8,10,12,14,16]
        },
        'lay_2':{
            'values': [4,8,16,32,64]
        },
        'lay_3':{
            'values': [8,16,32,64,128,256]
        },
        'activation1': {
            'values': ['relu', 'elu', 'selu','gelu']
        },
        'optimiz':{
            'values': ['adam','RMSprop','nadam','adamax']
        }
        
    }
}

if __name__ == "__main__":
    
    for c in [2500,5000,10000,16000,32000,64000]:
        sweep_id = wandb.sweep(sweep_config, entity="trekvar", project="CL_F_LSTM_%s"%str(c))
        sp = spm.SentencePieceProcessor(model_file='f_vocab/full_%s_new.model'%str(c))
        X = []
        Y = []

        data = pd.read_table("Classification_dataset.csv",delimiter='\t',header=0,names=["input", "output"],converters = {'input' : str,'output' : int})
        for i in range(0,len(data)):
            if data.at[i,'output'] == 1:
                X.append(np.array(sp.encode(data.at[i,'input']),dtype=np.int32))
                Y.append(1)
            if data.at[i,'output'] == 0 :
                X.append(data.at[i,'input'])
                Y.append(0) 
        Y = np.array(Y)



        X1_train, X1_test, y1_train,y1_test  = train_test_split(X,Y, test_size=0.1,shuffle=True)
        X1_train, X1_valid, y1_train,y1_valid  = train_test_split(X1_train,y1_train, test_size=0.11112,shuffle=True)

        X1_train = tf.ragged.constant(X1_train)
        X1_test = tf.ragged.constant(X1_test)
        X1_valid = tf.ragged.constant(X1_valid)

        y1_train = np.array(y1_train)
        y1_test = np.array(y1_test)
        y1_valid = np.array(y1_valid)
        def train():
        
        
        
            config_defaults = {
                'dims':8,
                'lay_2':8,
                'lay_3':8,
                'importance':1,
                'activation1':'relu',

                'optimiz':'adam',
                'batch_size': 512,
                'activation': 'relu',
                'optimizer': 'nadam',

                'seed': 42,

                'act_dim': 8,
                'mol_dim': 8,
                'dropout1':0.1

            }

            # Initialize a new wandb run
            wandb.init(config=config_defaults)

            # Config is a variable that holds and saves hyperparameters and inputs
            config = wandb.config

            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=[None],dtype=np.int32),
                tf.keras.layers.Embedding(
                    input_dim=c+1,
                    output_dim=config.dims,
                    mask_zero=False),
                tf.keras.layers.Dropout(config.dropout1),
                tf.keras.layers.LSTM(config.lay_2),

                tf.keras.layers.Dense(config.lay_3, activation=config.activation1),
                tf.keras.layers.Dense(1,activation='sigmoid')
            ])


            model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                      optimizer=config.optimiz,
                      metrics=['accuracy'])

            history = model.fit(X1_train,y1_train,batch_size=config.batch_size,
                      epochs=35,
                      validation_data=(X1_valid, y1_valid),class_weight = {0: 1.,1: config.importance},
                      callbacks=[WandbCallback(monitor='val_accuracy'),
                                  EarlyStopping(patience=7, restore_best_weights=True)])
        
        wandb.agent(sweep_id, function=train,count=35)

