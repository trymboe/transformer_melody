import numpy as np
import tensorflow as tf

def eval_model(model, dataset, input_length, num_predictions=120, sequence=False, temp=1):
    """
    Evaluates a trained model by generating new notes based on the given input sequence.

    Parameters:
    model (tf.keras.Model): A trained Keras model to generate new notes.
    dataset (tf.data.Dataset): A Tensorflow dataset containing the input sequences to generate new notes from.
    input_length (int): The length of the input sequence.
    num_predictions (int): The number of notes to generate.
    sequence (bool): If True, generates notes using a sequence-based approach. If False, generates notes using a non-sequence-based approach.
    temp (float): A temperature value to control the randomness of note generation.

    Returns:
    generated_notes (np.ndarray): An array containing the generated notes.

    """

    for input_seq, _ in dataset.take(1):
        input_notes = input_seq.numpy()[0]


    generated_notes = np.empty((num_predictions + input_length,129))

    generated_notes[:input_length, :] = input_notes

    for i in range(num_predictions):
        # print("input notes", generated_notes)
        if sequence:
            next_note = predict_next_note_sequence(input_notes, model)
            generated_notes = np.concatenate((generated_notes, next_note), axis=0)
            input_notes = next_note
        else:
            next_note = predict_next_note(input_notes, model, temp)
            generated_notes[i+input_length] = next_note
            input_notes = np.delete(input_notes, 0, axis=0)
            # print(input_notes)
            input_notes = np.append(input_notes, next_note, axis=0)

    generated_notes = generated_notes[:, :-1]
    generated_notes[generated_notes != 0] = 127

    return generated_notes

def predict_next_note(notes: np.ndarray, model: tf.keras.Model, temperature: float) -> int:
    """
    Predicts the next note in a sequence of notes based on a trained model.

    Args:
    notes (np.ndarray): A numpy array containing the input sequence of notes.
    The shape should be (sequence_length, num_unique_notes).
    model (tf.keras.Model): The trained model to use for prediction.
    temperature (float): A scalar value controlling the randomness of the prediction.
    Higher temperature will lead to more random predictions.

    Returns:
    int: The index of the predicted next note in the vocabulary.
    """    
    assert temperature >= 0

    # Add batch dimension
    inputs = tf.expand_dims(notes, 0)

    predictions_logits = model.predict(inputs)
    # print(predictions_logits)

    max_idx = get_index(predictions_logits, temperature)
    print(max_idx)
    # Create a new array with 1 at the index of the maximum value
    next_note = np.zeros_like(predictions_logits)
    next_note[0][max_idx] = 1

    return next_note

#Discontinued
def predict_next_note_sequence(notes: np.ndarray, model: tf.keras.Model, temperature: float = 0) -> int:
    """Generates a note IDs using a trained sequence model."""
    assert temperature >= 0

    # Add batch dimension
    inputs = tf.expand_dims(notes, 0)

    predictions_logits = np.squeeze(model.predict(inputs), axis=0)
    
    index = get_index(predictions_logits, temperature)

    max_idx = np.argmax(predictions_logits, axis=1)

    print("generated_notes", max_idx)
    
    predictions = np.zeros_like(predictions_logits)
    predictions[np.arange(predictions_logits.shape[0]), index] = 1
    print(predictions.shape)

    return predictions

def get_index(prediction_logits, epsilon):
    """
    This function takes two inputs, prediction_logits, and epsilon, and returns an index based on a probabilistic selection process.

    Args:
    - prediction_logits: A 2D numpy array, where the first dimension represents the number of distributions and the second dimension represents the number of values in each distribution.
    - epsilon: A float value between 0 and 1 representing the probability of choosing the index greedily or randomly.

    Returns:
    - An integer representing the index selected based on the probabilistic selection process.
    """
    # # Find the indices of the k highest probabilities
    # prediction_logits = prediction_logits[0]
    # k = 5
    # top_k_indices = np.argsort(prediction_logits)[::-1][:k]

    # # Get the corresponding probabilities
    # top_k_probabilities = [prediction_logits[i] for i in top_k_indices]

    # top_k_probabilities_normalized = np.array([p / sum(top_k_probabilities) for p in top_k_probabilities])


    # # Get the index of the highest probability
    # highest_index = np.argmax(top_k_probabilities_normalized)

    # # Create a one-hot vector with the same length as the original list
    # one_hot = np.zeros_like(top_k_probabilities_normalized)
    # for i in range(len(one_hot)):
    #     one_hot[i] = 1/len(one_hot)
    # # one_hot[highest_index] = 1

    # # Interpolate between the original probabilities and the one-hot vector
    # new_probabilities = epsilon * top_k_probabilities_normalized + (1 - epsilon) * one_hot

    # # Normalize the new probabilities to make them add up to 1
    # new_probabilities /= np.sum(new_probabilities)

    # # exit()
    # # Randomly choose an index based on the weights
    # chosen_index = np.random.choice(top_k_indices, p=new_probabilities)
    
    # print(chosen_index)


    # return chosen_index






    # Create an array of random values to compare with epsilon
    rand_vals = np.random.rand(prediction_logits.shape[0])

    # Create an array of indices from 0 to 127
    indices = np.arange(prediction_logits.shape[1])

    # Determine whether to choose the index greedily or randomly
    greedy = rand_vals > epsilon

    # Get the index with the highest probability for each of the 120 distributions
    max_indices = np.argmax(prediction_logits, axis=1)

    sorted_indices = np.argsort(prediction_logits)[0]
    # print(sorted_indices)
    # Choose the first index with probability p, and the second index with probability 1-p
    print(sorted_indices[-1], sorted_indices[-2],sorted_indices[-3], sorted_indices[-4])
    if np.random.random() > epsilon:
        return sorted_indices[-2]
    else:
        if np.random.random() > epsilon:
            return sorted_indices[-1]
        else:
            if np.random.random() > epsilon:
                return sorted_indices[-3]
            else:
                return sorted_indices[-4]

