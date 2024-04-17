from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model
MODEL_FILE = 'mainmodel.h5'
model = load_model(MODEL_FILE)

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi'}

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for image classification
@app.route('/classify', methods=['POST'])
def classify_image():
    if 'file' not in request.files:
        return render_template('result.html', result="No file part")
    
    file = request.files['file']

    if file.filename == '':
        return render_template('result.html', result="No selected file")

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Perform image classification and draw green box
        result, processed_image_path = classify_image_with_model(file_path)
        print("CHeck here")
        print(processed_image_path)

        return render_template('result.html', result=result, image_path=processed_image_path)
    else:
        return render_template('result.html', result="Invalid file format")

# Function to classify image using the loaded model and draw green box
def classify_image_with_model(file_path):
    # Check if the file is a video
    if file_path.lower().endswith('.mp4') or file_path.endswith('.avi'):
        # Open video file
        cap = cv2.VideoCapture(file_path)
        
        # Read frames until a frame with a face is found
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return "Error: Unable to read video file or insufficient frames", None
            
            # Convert frame to grayscale for face detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Load Haar cascade for face detection
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            # If faces are detected, proceed with classification
            if len(faces) > 0:
                # Convert frame to RGB format
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize frame to match model input size
                resized_frame = cv2.resize(frame_rgb, (224, 224))
                
                # Preprocess input
                img_array = tf.keras.applications.mobilenet_v2.preprocess_input(np.expand_dims(resized_frame, axis=0))
                
                # Perform inference
                predictions = model.predict(img_array)
                
                # Extract probability of being a deepfake
                probability = predictions[0][0]

                # Decode predictions
                if probability > 0.5:
                    result = "The video is classified as a deepfake with a probability of {:.2f}%.".format(probability * 100)
                else:
                    result = "The video is classified as a real video with a probability of {:.2f}%.".format((1 - probability) * 100)
                
                # Draw green box around the face
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        #/Users/steveaugustine/Desktop/deepfake_detection/static/uploads/processed_uploads_Screenshot_2024-04-15_at_7.30.47_PM.png
                # Save the processed image static/uploads/processed_uploads_Screenshot_2024-04-14_at_12.11.18_AM.png
                processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + secure_filename(file_path)+'.png')
                cv2.imwrite(processed_image_path, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # Release video capture
                cap.release()
                
                return result, processed_image_path
        
        # If no faces are found
        cap.release()
        return "Error: No face detected in the video", None
    
    else:
        # If file is an image, use existing image classification code
        img = tf.keras.preprocessing.image.load_img(file_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)  # Preprocess input
        
        # Perform inference
        predictions = model.predict(img_array)
        
        # Extract probability of being a deepfake
        probability = predictions[0][0]

        # Decode predictions
        if probability > 0.5:
            result = "The image is classified as a deepfake with a probability of {:.2f}%.".format(probability * 100)
        else:
            result = "The image is classified as a real image with a probability of {:.2f}%.".format((1 - probability) * 100)
        
        # Face detection and draw green box
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Draw green rectangles around detected faces
        processed_img = np.array(img)
        for (x, y, w, h) in faces:
            cv2.rectangle(processed_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Save the processed image
        processed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + secure_filename(file_path))
        cv2.imwrite(processed_image_path, cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))
        print(processed_image_path)
        return result, processed_image_path

if __name__ == '__main__':
    app.run(debug=True)
