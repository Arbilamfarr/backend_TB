from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from flask_cors import CORS
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

CORS(app)
# Configurer le dossier de stockage des fichiers uploadés
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model
model = load_model('c:/Users/HP/Desktop/projet deep learning/model.h5')

# Fonction pour vérifier les extensions d'image valides
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png', 'gif'}

@app.route("/upload-image", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image = request.files["image"]
    if image.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Check if the file is a valid image
    if not allowed_file(image.filename):
        return jsonify({'error': 'Invalid file type. Only images are allowed.'}), 400

    # Process the image
    img = Image.open(image)
    img = img.resize((224, 224))  # Resize to match model input
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make predictions
    predictions = model.predict(img_array)
    # Assuming the model outputs a single value or class
    result = np.argmax(predictions, axis=1).tolist()  # Change this as per your model output

    # Sécuriser et sauvegarder l'image
    filename = secure_filename(image.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(filepath)

    return jsonify({"message": "Image uploaded successfully", "filename": filename, "result": result}), 200

if __name__ == "__main__":
    app.run(debug=True)
