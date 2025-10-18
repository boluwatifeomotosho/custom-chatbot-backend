import json
import pickle
import os
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import random

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Define paths for data and model files
INTENTS_FILE = 'intents.json'
MODELS_DIR = '../Models'
WORDS_FILE = os.path.join(MODELS_DIR, 'words.pkl')
CLASSES_FILE = os.path.join(MODELS_DIR, 'classes.pkl')
MODEL_FILE = os.path.join(MODELS_DIR, 'chatbot_model.h5')

def train_chatbot_model():
    """Train the chatbot model using the intents data"""

    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Load intents file
    with open(INTENTS_FILE, 'r', encoding='utf-8') as file:
        intents = json.load(file)

    words = []
    classes = []
    documents = []
    ignore_words = ['?', '.', ',', '!']

    # Process each intent
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            # Tokenize each word
            w = nltk.word_tokenize(pattern)
            words.extend(w)
            # Add documents in the corpus
            documents.append((w, intent['tag']))
            # Add to our classes list
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    # Lemmatize and lower each word and remove duplicates
    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
    words = sorted(list(set(words)))

    # Sort classes
    classes = sorted(list(set(classes)))

    print(f"{len(documents)} documents")
    print(f"{len(classes)} classes: {classes}")
    print(f"{len(words)} unique lemmatized words: {words}")

    # Create the Models directory if it doesn't exist
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    # Save words and classes
    pickle.dump(words, open(WORDS_FILE, 'wb'))
    pickle.dump(classes, open(CLASSES_FILE, 'wb'))

    # Create training data
    training = []
    output_empty = [0] * len(classes)

    for doc in documents:
        # Initialize bag of words
        bag = []
        # List of tokenized words for the pattern
        pattern_words = doc[0]
        # Lemmatize each word - create base word, in attempt to represent related words
        pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

        # Create bag of words array with 1, if word match found in current pattern
        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)

        # Output is a '0' for each tag and '1' for current tag (for each pattern)
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1

        training.append([bag, output_row])

    # Shuffle features and turn into np.array
    random.shuffle(training)
    training = np.array(training, dtype=object)

    # Split data into features (X) and labels (Y)
    train_x = np.array(list(training[:, 0]))
    train_y = np.array(list(training[:, 1]))

    # Split data into training and validation sets
    train_x, val_x, train_y, val_y = train_test_split(
        train_x, train_y, test_size=0.2, random_state=42, stratify=train_y
    )
    print("Training data created")

    # Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
    # equal to number of intents to predict output intent with softmax
    model = keras.Sequential([
        layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(train_y[0]), activation='softmax')
    ])

    # Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
    sgd = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # Fitting and saving the model
    hist = model.fit(
        train_x,
        train_y,
        validation_data=(val_x, val_y),
        epochs=200,
        batch_size=5,
        verbose=1
    )
    model.save(MODEL_FILE)

    print("Model created and saved successfully!")
    return model

if __name__ == "__main__":
    train_chatbot_model()
