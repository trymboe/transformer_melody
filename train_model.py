import matplotlib.pyplot as plt
import tensorflow as tf


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


def retraining(train_dataset, val_dataset, load_model_path):
  loaded_model = tf.keras.models.load_model(load_model_path)
  history = loaded_model.history
  num_epochs_completed = len(history.history['loss'])
  
  loaded_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  
  callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        min_delta=0,
        patience=50,
        verbose=1,
        restore_best_weights=True),
    ]
  
  history = loaded_model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset, callbacks=callbacks,)
  plot(history, load_model_path)



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

  callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        min_delta=0,
        patience=50,
        verbose=1,
        restore_best_weights=True),
    ]
  
  # Fit the model
  history = model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset, callbacks=callbacks,)
  
  plot(history, load_model_path)
  model.save_weights(save_model_path)
  # Evaluate the model
  evaluation = model.evaluate(val_dataset)

  print('Validation Loss:', evaluation[0])
  print('Validation Accuracy:', evaluation[1])
