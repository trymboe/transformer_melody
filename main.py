import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import logging
import sys


from transformer import create_transformer_model_ml_nodes
from data_process import prepare_data, piano_roll_to_pretty_midi
from generate import eval_model

np.set_printoptions(threshold=sys.maxsize)

logging.getLogger('tensorflow').setLevel(logging.ERROR)

EPOCHS = 250
BATCH_SIZE = 64
VALIDATION_SIZE = 0.15
LEARNING_RATE = 0.005
TEMPERATURE = 0.5

NUM_PREDICTIONS = 60 
INPUT_LENGTH = 40

FS = 12

def plot(history, path): 
    plt.plot(history.epoch, history.history['loss'], label='total training loss')
    plt.savefig(path+'_training_loss.png')
    plt.figure()
    plt.plot(history.epoch, history.history['val_loss'], label='total val loss')
    plt.savefig(path+'_validation_loss.png') 
    plt.figure()
    plt.plot(history.epoch, history.history['accuracy'], label='total accuracy')
    plt.savefig(path+'_accuracy.png') 

    plt.figure()
    plt.plot(history.epoch, history.history['val_accuracy'], label='total val accuracy')
    plt.savefig(path+'_validation_accuracy.png') 


  

def transformer(train_dataset, val_dataset, save_model_path):
  # Define hyperparameters

  sequence_length = 40
  input_shape = (sequence_length, 129)
  output_dim = 129

  d_model = 129
  num_heads = 4
  num_layers = 2
  dff = 256

  # Create the model
  model = create_transformer_model_ml_nodes(input_shape, d_model, num_heads, num_layers, dff, output_dim)

  # Compile the model
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

  # Fit the model
  history = model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset)
  
  callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        min_delta=0,
        patience=50,
        verbose=1,
        restore_best_weights=True),
    ]
  plot(history, load_model_path)
  model.save_weights(save_model_path)
  # Evaluate the model
  evaluation = model.evaluate(val_dataset)

  print('Validation Loss:', evaluation[0])
  print('Validation Accuracy:', evaluation[1])

if __name__  == "__main__":
  train = True
  sequence = False
  big_model = True
  dataset = "x_small"

  model_name = "model1"


  #Correct path based on the parameters
  load_model_path = f'models/{model_name}/{dataset}/e_{EPOCHS}_{INPUT_LENGTH}'
  out_file = f"results/{model_name}/{dataset}/e_{EPOCHS}_{INPUT_LENGTH}"
  
  train_ds, val_ds = prepare_data(f"data/melody/{dataset}", INPUT_LENGTH, 1, FS, VALIDATION_SIZE, BATCH_SIZE)

  if train:
    transformer(train_ds, val_ds, load_model_path)
  else:
    sequence_length = 40
    input_shape = (sequence_length, 129)
    output_dim = 129

    d_model = 129
    num_heads = 4
    num_layers = 2
    dff = 256
    model = create_transformer_model_ml_nodes(input_shape, d_model, num_heads, num_layers, dff, output_dim)
    model.load_weights(load_model_path)
    generated_notes = eval_model(model, train_ds, INPUT_LENGTH, num_predictions=NUM_PREDICTIONS, sequence=sequence, temp=TEMPERATURE)
    pm = piano_roll_to_pretty_midi(generated_notes.transpose(), FS/2)
    pm.write(out_file+".mid")
  
  
  plt.show()