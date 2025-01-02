from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from flask_cors import CORS
import base64


app = Flask(__name__)

CORS(app)
# Configurer le dossier de stockage des fichiers uploadés
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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

    # Vérifier si le fichier est une image valide
    if not allowed_file(image.filename):
        return jsonify({'error': 'Invalid file type. Only images are allowed.'}), 400

    # Sécuriser et sauvegarder l'image
    filename = secure_filename(image.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(filepath)

    # Exemple : retourner une réponse avec des métadonnées de l'image
    loss = 0.5  # Actual loss value
    accuracy = 0.8  # Actual accuracy value
    with open(os.path.join(app.config['UPLOAD_FOLDER'], 'res.jpg'), 'rb') as img_file:
        image_data = base64.b64encode(img_file.read()).decode('utf-8')  # Encode image to base64
    return jsonify({"message": "C e personne a ete detectee", "filename": filename, "loss": loss, "accuracy": accuracy, "image": image_data}), 200

if __name__ == "__main__":
    app.run(debug=True)
