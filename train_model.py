import os
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


# Load images and labels
def load_data(image_folder):
    images = []
    labels = []
    label_map = {'happy': 0, 'sad': 1, 'angry': 2}

    for label in os.listdir(image_folder):
        for image_file in os.listdir(os.path.join(image_folder, label)):
            image_path = os.path.join(image_folder, label, image_file)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (48, 48))  # Resize to match input size
            images.append(image)
            labels.append(label_map[label])

    return np.array(images), np.array(labels)


# Prepare dataset
image_folder = 'images'  # Your folder containing happy, sad, angry images
X, y = load_data(image_folder)
X = X.astype('float32') / 255.0  # Normalize images
y = to_categorical(y, num_classes=3)  # One-hot encoding of labels

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))  # Output layer for 3 classes

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save model
model.save('model.h5')
