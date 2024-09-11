from flask import Flask, request, render_template
import os
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from PIL import Image

app = Flask(__name__)

# Load your trained model (make sure to provide the correct path)
model = load_model('model.h5')

# Emotion labels corresponding to the model's output
emotion_labels = ['Happy', 'Sad', 'Angry']

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        selected_emotion = request.form['emotion']

        if file:
            file_path = os.path.join('static/uploads', file.filename)
            file.save(file_path)

            # Process the image for prediction
            img = load_img(file_path, target_size=(48, 48), color_mode='grayscale')
            img_array = img_to_array(img)

            # Convert grayscale image to RGB
            img_rgb = Image.fromarray(img_array.squeeze()).convert('RGB')
            img_array = img_to_array(img_rgb)
            img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize to [0, 1]

            # Make prediction
            prediction = model.predict(img_array)
            predicted_emotion_index = np.argmax(prediction)
            predicted_emotion = emotion_labels[predicted_emotion_index]
            confidence = prediction[0][predicted_emotion_index] * 100  # Get confidence percentage

            # Log the prediction details
            print(f"Uploaded Emotion: {selected_emotion}, Predicted Emotion: {predicted_emotion}, Confidence: {confidence:.2f}%, Prediction: {prediction}")

            # Render results with confidence statistics
            return render_template('result.html',
                                   uploaded_image=file.filename,
                                   predicted_emotion=predicted_emotion,
                                   selected_emotion=selected_emotion,
                                   confidence=confidence,
                                   predictions=prediction[0])  # Pass the full prediction array for details

if __name__ == '__main__':
    app.run(debug=True)
