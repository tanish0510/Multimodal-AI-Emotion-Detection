import tensorflow as tf
from PIL import Image
import numpy as np


def load_model(model_path):
    # Load your trained model
    return tf.keras.models.load_model(model_path)


def preprocess_image(file):
    try:
        # Open the image and convert it to RGB
        img = Image.open(file).convert('RGB')
        img = img.resize((48, 48))  # Resize to expected dimensions
        img_array = np.array(img)  # Convert to numpy array

        img_array = img_array / 255.0  # Normalize to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        print(f"Error in preprocess_image: {str(e)}")
        return None  # Return None if processing fails


def predict_emotion(model, file_path):
    img_array = preprocess_image(file_path)
    if img_array is None:
        return "Error processing image"

    # Check the shape of the processed image
    print(f"Image shape: {img_array.shape}")

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]  # Get the predicted class index
    return predicted_class  # Return the predicted class index or its label
