import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import img_to_array, load_img

def load_data(file_path):
    data = pd.read_csv(file_path)
    texts = data['text'].tolist()  # Get the text data
    labels = data['label'].tolist()  # Get the labels
    return texts, labels

def load_tokenizer():
    tokenizer = Tokenizer()  # No arguments required for initialization
    return tokenizer

def preprocess_texts(texts, tokenizer, max_length=100):
    tokenizer.fit_on_texts(texts)  # Fit the tokenizer on the texts
    sequences = tokenizer.texts_to_sequences(texts)  # Convert texts to sequences
    padded_sequences = pad_sequences(sequences, maxlen=max_length)  # Pad sequences
    return padded_sequences

def load_images(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpeg') or filename.endswith('.png'):  # Adjust based on your image formats
            img_path = os.path.join(folder, filename)
            img = load_img(img_path, target_size=(64, 64))  # Resize to the input size of your model
            img = img_to_array(img) / 255.0  # Normalize pixel values
            images.append(img)

            # Extract label from filename (or create a mapping based on your use case)
            if 'happy' in filename:
                labels.append(0)  # Happy
            elif 'sad' in filename:
                labels.append(1)  # Sad
            elif 'angry' in filename:
                labels.append(2)  # Angry

    return np.array(images), np.array(labels)
