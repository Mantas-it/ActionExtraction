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
            'values': [32, 64, 128]
        },
        "importance" :{
          'distribution': 'uniform',
          "min": 1.0,
          "max": 3.0
        },
        
        'dims':{
            'values': [1,4,8,12,16]
        },
        'heads':{
            'values': [2,4,8]
        },
        'layerz':{
            'values': [1,2,3,4]
        },
        'lay_2':{
            'values': [4,8,16,32,64]
        },
        'activation1': {
            'values': ['relu', 'elu', 'selu','gelu']
        },
        'optimiz':{
            'values': ['adam','RMSprop','nadam','adamax']
        }
        
    }
}


MAX_SEQ_LENGTH = 600

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

        unique, counts = np.unique(Y, return_counts=True)
        di = dict(zip(unique, counts))
        print(di)
        print(round(di[1]/di[0],3)*100,'%')
        max_length = 600

        too_long = np.array([len(x) > max_length for x in X])

        # Remove the too-long instances from X and Y
        X = [x for x in X if len(x) <= max_length]
        Y = Y[~too_long]
        print(len(X))
        # Pad the remaining instances with zeros
        X_padded = np.zeros((len(X), max_length))
        for i, x in enumerate(X):
            X_padded[i, :len(x)] = x


        X1_train, X1_test, y1_train,y1_test  = train_test_split(X_padded,Y, test_size=0.1,shuffle=True)
        X1_train, X1_valid, y1_train,y1_valid  = train_test_split(X1_train,y1_train, test_size=0.11112,shuffle=True)



        #X1_train = tf.ragged.constant(X1_train)
        #X1_test = tf.ragged.constant(X1_test)
        #X1_valid = tf.ragged.constant(X1_valid)

        y1_train = np.array(y1_train)
        y1_test = np.array(y1_test)
        y1_valid = np.array(y1_valid)
        
        def train():
        
        
        
            config_defaults = {
                'dims':8,
                'lay_2':8,
                'lay_3':8,
                'importance':1,
                'heads':2,
                'activation1':'relu',
                'layerz':2,

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
            
            MAX_SEQ_LENGTH = 600
            
            NUM_LAYERS = 2
            
            VOCAB_SIZE = c + 1
            input_shape = (MAX_SEQ_LENGTH,)

            # Define the input layer
            inputs = tf.keras.layers.Input(shape=input_shape, dtype='int32')

            # Define the embedding layer
            embedding_layer = tf.keras.layers.Embedding(VOCAB_SIZE, config.dims)
            embeddings = embedding_layer(inputs)

            # Define the positional encoding layer
            pos_encoding_layer = tf.keras.layers.Embedding(MAX_SEQ_LENGTH, config.dims)
            pos_encoding = pos_encoding_layer(tf.range(start=0, limit=MAX_SEQ_LENGTH, delta=1))

            # Add the positional encoding to the embeddings
            embeddings += pos_encoding

            # Define the transformer layers
            for i in range(config.layerz):
                # Define the multi-head attention layer
                attention = tf.keras.layers.MultiHeadAttention(num_heads=config.heads, key_dim=config.dims)
                attn_output = attention(embeddings, embeddings)
                
                # Define the feedforward layer
                feedforward = tf.keras.Sequential([
                    tf.keras.layers.Dense(config.lay_2, activation=config.activation1),
                    tf.keras.layers.Dense(config.dims)
                ])
                feedforward_output = feedforward(attn_output)
                
                # Add residual connections and layer normalization
                layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
                layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
                attention_output = layer_norm1(embeddings + attn_output)
                feedforward_output = layer_norm2(attention_output + feedforward_output)
                embeddings = feedforward_output

            # Take the mean across the sequence dimension
            pooled_output = tf.keras.layers.GlobalAveragePooling1D()(embeddings)

            # Define the output layer
            outputs = tf.keras.layers.Dense(1, activation='sigmoid')(pooled_output)

            model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
            
          
            model.compile(optimizer=config.optimiz, loss='binary_crossentropy', metrics=['accuracy'])

            

            history = model.fit(X1_train,y1_train,batch_size=config.batch_size,
                      epochs=35,
                      validation_data=(X1_valid, y1_valid),class_weight = {0: 1.,1: config.importance},
                      callbacks=[WandbCallback(monitor='val_accuracy'),
                                  EarlyStopping(patience=7, restore_best_weights=True)])
        
        wandb.agent(sweep_id, function=train,count=35)

