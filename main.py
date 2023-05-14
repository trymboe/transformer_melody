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

EPOCHS = 500
BATCH_SIZE = 64
VALIDATION_SIZE = 0.15
LEARNING_RATE = 0.005
TEMPERATURE = 0.5

NUM_PREDICTIONS = 60 
INPUT_LENGTH = 40

FS = 12


if __name__  == "__main__":
  train = True
  sequence = False
  big_model = True
  dataset = "medium"

  model_name = "model1"


  #Correct path based on the parameters
  load_model_path = f'models/{model_name}/{dataset}/e_{EPOCHS}_{INPUT_LENGTH}'
  out_file = f"results/{model_name}/{dataset}/e_{EPOCHS}_{INPUT_LENGTH}"
  
  train_ds, val_ds = prepare_data(f"data/melody/{dataset}", INPUT_LENGTH, 1, FS, VALIDATION_SIZE, BATCH_SIZE)

  if train:
    transformer(train_ds, val_ds, load_model_path)
    retraining(train_ds, val_ds, load_model_path)
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