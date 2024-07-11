from flask import Flask, render_template, request, jsonify
from keras.models import load_model
import librosa
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('modelspechh.keras')  # Replace 'modelspechh.keras' with the path to your trained model

# Define function to extract features from audio file
def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

# Define endpoint to handle file upload and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Check if the request contains a file
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    # Check if the file is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Check if the file is allowed
    allowed_extensions = {'wav', 'mp3'}  # Add more extensions if needed
    if '.' not in file.filename or file.filename.split('.')[-1].lower() not in allowed_extensions:
        return jsonify({'error': 'Unsupported file format'})
    
    # Save the file to a temporary location
    file_path = 'C:/Users/anilk/jup/ml speech/OAF_back_angry.wav'  # Temporary file path
    file.save(file_path)
    
    # Extract features from the uploaded audio file
    features = extract_mfcc(file_path)
    features = np.expand_dims(features, axis=0)  # Add batch dimension
    features = np.expand_dims(features, axis=-1)
    # Make predictions using the model
    predictions = model.predict(features)
    predicted_class = np.argmax(predictions)
    
    # Define class labels
    class_labels = ['fear', 'angry', 'disgust', 'neutral', 'sad', 'ps', 'happy']
    
    # Return the predicted class
    return jsonify({'predicted_class': class_labels[predicted_class]})

# Define endpoint to render the HTML form
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
