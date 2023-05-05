import sys
import numpy as np
import logging

from data_process import prepare_data, piano_roll_to_pretty_midi
from generate import eval_model
from model import create_model, create_model_sequence, train_model
from transformer import create_transformer_model

np.set_printoptions(threshold=sys.maxsize)

logging.getLogger('tensorflow').setLevel(logging.ERROR)

EPOCHS = 25
BATCH_SIZE = 64
VALIDATION_SIZE = 0.15
LEARNING_RATE = 0.005
TEMPERATURE = 0.5

NUM_PREDICTIONS = 60 
INPUT_LENGTH = 40

FS = 12

def transformer(train_dataset, val_dataset, save_model_path):
  save_model_path = save_model_path + "trans_binary"
  # Define hyperparameters
  batch_size = 32
  sequence_length = 40
  input_shape = (sequence_length, 129)
  output_dim = 129

  d_model = 129
  num_heads = 4
  num_layers = 2
  dff = 256

  # Create the model
  model = create_transformer_model(input_shape, d_model, num_heads, num_layers, dff, output_dim)

  # Compile the model
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

  # Fit the model
  history = model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset)
  model.save_weights(save_model_path)
  # Evaluate the model
  evaluation = model.evaluate(val_dataset)

  print('Validation Loss:', evaluation[0])
  print('Validation Accuracy:', evaluation[1])

if __name__  == "__main__":
  train = False
  sequence = False
  big_model = True
  dataset = "xx_small"
  #Choose the model_name that fits the parameters
  if not sequence and not big_model:
    model_name = "model1"
  elif sequence and not big_model:
    model_name = "model2"
  elif not sequence and big_model:
    model_name = "model3"
  elif sequence and big_model:
    model_name = "model4"

  #Correct path based on the parameters
  load_model_path = f'models/{model_name}/{dataset}/e_{EPOCHS}_{INPUT_LENGTH}'
  out_file = f"results/{model_name}/{dataset}/e_{EPOCHS}_{INPUT_LENGTH}"
  
  train_ds, val_ds = prepare_data(f"data/melody/test", INPUT_LENGTH, 1, FS, VALIDATION_SIZE, BATCH_SIZE)

  if train:
    transformer(train_ds, val_ds, load_model_path)
  else:
    batch_size = 32
    sequence_length = 40
    input_shape = (sequence_length, 129)
    output_dim = 129

    d_model = 129
    num_heads = 4
    num_layers = 2
    dff = 256
    model = create_transformer_model(input_shape, d_model, num_heads, num_layers, dff, output_dim)
    model.load_weights(load_model_path)
    generated_notes = eval_model(model, train_ds, INPUT_LENGTH, num_predictions=NUM_PREDICTIONS, sequence=sequence, temp=TEMPERATURE)
    pm = piano_roll_to_pretty_midi(generated_notes.transpose(), FS/2)
    pm.write(out_file+".mid")
  
  exit()

  #Prepare data and create model, either as a sequence model (not used), or a normal model
  if sequence:
    if train:
      train_ds, val_ds = prepare_data(f"data/melody/{dataset}", INPUT_LENGTH, INPUT_LENGTH, FS, VALIDATION_SIZE, BATCH_SIZE)
    else:
      train_ds, val_ds = prepare_data(f"data/melody/test", INPUT_LENGTH, INPUT_LENGTH, FS, VALIDATION_SIZE, BATCH_SIZE)
    model, loss, _ = create_model_sequence(INPUT_LENGTH, LEARNING_RATE, model_name)
  else:
    if train:
      train_ds, val_ds = prepare_data(f"data/melody/{dataset}", INPUT_LENGTH, 1, FS, VALIDATION_SIZE, BATCH_SIZE)
    else:
      train_ds, val_ds = prepare_data(f"data/melody/test", INPUT_LENGTH, 1, FS, VALIDATION_SIZE, BATCH_SIZE)
    model, loss, _ = create_model(INPUT_LENGTH, LEARNING_RATE, model_name)

  #Load model for evaluation
  if not train:
    model.load_weights(load_model_path)
  else:
      train_model(model, train_ds, val_ds, f"models/{model_name}/{dataset}/e_{EPOCHS}_{INPUT_LENGTH}", EPOCHS)

  #Generate new midi-track
  if not train:
    generated_notes = eval_model(model, train_ds, INPUT_LENGTH, num_predictions=NUM_PREDICTIONS, sequence=sequence, temp=TEMPERATURE)
    pm = piano_roll_to_pretty_midi(generated_notes.transpose(), FS/2)

    pm.write(out_file+".mid")


  # plt.show()




'''
for input_seq, label_seq in dataset.take(5):
    print("Input sequence:\n", input_seq.numpy())
    print("Label sequence:\n", label_seq.numpy())
    print()
'''