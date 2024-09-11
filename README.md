# MultiModal Emotion Prediction 🌟🔮

![MultiModal Emotion Prediction](https://img.shields.io/badge/MultiModal-Emotion%20Prediction-blue?style=for-the-badge)

## Overview

MultiModal Emotion Prediction is a deep learning-based web application that predicts human emotions using **text**, **images**, and **emojis**. This project combines multiple modes of input to classify emotions with high accuracy, presenting detailed predictions alongside confidence scores and graphical representations.

---

## Features ✨

- **Upload Images**: Upload an image for emotion prediction.
- **Emotion Selection**: Users can manually select their predicted emotion for comparison.
- **Confidence Scores**: Detailed confidence breakdown of predicted emotions.
- **Visual Graphs**: Interactive bar charts using Chart.js to visualize prediction results.
- **Responsive UI**: Modern, minimalistic design with animations and gradient effects.
  
---

## Tech Stack 🛠️

- **Flask**: Python-based web framework for backend development.
- **TensorFlow/Keras**: Used for model training and emotion classification.
- **Chart.js**: For dynamic graph rendering of emotion prediction results.
- **HTML/CSS**: For responsive and visually appealing frontend.
- **JavaScript**: To add interactivity to the application.
  
---

## Screenshots 📸

### Emotion Upload Page
![Emotion Upload](screenshots/upload_page.png)

### Results Page with Confidence Graph
![Results Page](screenshots/results_page.png)

---

## Installation & Setup ⚙️

1. **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/emotion-prediction.git
    cd emotion-prediction
    ```

2. **Create a Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Application**
    ```bash
    flask run
    ```

5. **Access the App**
    Open your browser and navigate to: `http://127.0.0.1:5000`

---

## Project Structure 📁


---

## Usage Guide 📖

1. **Upload an Image**: Select an image of a person expressing one of the emotions (happy, sad, angry).
2. **Select Emotion**: Choose your perceived emotion from the dropdown menu.
3. **View Results**: After submitting, the application will display the predicted emotion, your selection, confidence scores, and a bar chart.

---

## Future Enhancements 🔮

- **Additional Emotions**: Expand the model to detect more nuanced emotions like surprise or disgust.
- **Live Emotion Tracking**: Incorporate live video analysis for real-time emotion detection.
- **Text + Image Prediction**: Combine both text and image inputs for even more accurate emotion classification.

---

## Contributing 🤝

Contributions are welcome! Feel free to open issues or submit pull requests to enhance this project. Please make sure to follow the contribution guidelines.

---

## License 📄

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements 🙌

- **TensorFlow** for providing excellent deep learning frameworks.
- **Chart.js** for the interactive and easy-to-use graphing library.
- Special thanks to all contributors who made this project possible!

---


