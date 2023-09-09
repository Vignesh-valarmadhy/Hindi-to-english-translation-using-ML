# Hindi-to-english-translation-using-ML



---

# Machine Translation with TensorFlow

This code is a TensorFlow implementation of a machine translation model for translating Hinglish (a mix of Hindi and English) to English. It involves the following major components:

## Data Preparation

1. **Data Import**: The code begins by importing the dataset from a file named `hin.txt`.

2. **Text Preprocessing**: The code preprocesses the text data by tokenizing words, removing punctuation, and performing spell correction using the Autocorrect library.

3. **Vocabulary Building**: It builds separate vocabularies for English and Hindi words and associates them with pre-trained word embeddings from GloVe (50-dimensional).

4. **Data Conversion**: English and Hindi sentences are tokenized and converted into numerical representations. Hindi words are integer-encoded using LabelEncoder.

## Model Architecture

The core of the code involves building a sequence-to-sequence model with an attention mechanism using TensorFlow. Key components include:

- **Encoder**: A basic LSTM encoder processes the input Hindi sentences.
- **Decoder**: The decoder is an LSTM-based raw RNN with attention, translating the encoded context to English.
- **Loss Function**: Mean Square Loss is used for training.

## Training and Evaluation

- The model is trained using an Adam optimizer for a specified number of epochs.
- Training is done in batches with batch size (`bt`).
- The code computes and displays the mean square loss during training.
- After training, the code can be used to predict English translations for Hinglish sentences.

## Analysis

- The code performs analysis on the trained model by calculating the Euclidean distance between predicted and actual embeddings for a set of words.
- It also includes an analysis of gradients with respect to word embeddings.

## Usage

- Ensure you have the required libraries and data files (e.g., GloVe embeddings, 'hin.txt') in the appropriate directories.
- Adjust hyperparameters and settings as needed.
- Run the code, and it will train the machine translation model and display training progress.

## Dependencies

- TensorFlow
- Autocorrect
- NumPy
- NLTK
- scikit-learn

## Author

VIGNESH S

---

