from PIL.ImageOps import grayscale
from flask import Flask, render_template, request, url_for
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import gdown
from infected import visualize_infection, display_predicted_mask  # your infection overlay function

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def download_models():
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)

    models_to_download = {
        '1a3gOW9J71SW6JpDpIISZrKuu0OfTQRlQ': 'FILE_ID_1',
        '1DukDCDS99H6c8s4ikbTqu2TnkVBPHmPY': 'FILE_ID_2',
        '1A7aHqY4_Y_IoXpf8owDmV_mhvHcTqhVq': 'FILE_ID_3'
    }

    for model_name, file_id in models_to_download.items():
        model_path = os.path.join(model_dir, model_name)
        if not os.path.exists(model_path):
            print(f"Downloading {model_name}...")
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, model_path, quiet=False)
        else:
            print(f"{model_name} already exists.")



# Load models
tb_pneumonia_model = load_model('models/TBPneumoniaClassificationCNN.h5')
lung_cancer_model = load_model('models/NoduleDetectionModel.h5')
main_model = load_model('models/6LungDiseasedetectionmodel.h5')

# Preprocess function
def preprocess_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read grayscale
    img = cv2.resize(img, (256, 256))  # Resize to match model input
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img.reshape(1, 256, 256, 1)  # Shape: (1, 256, 256, 1)

# Homepage
@app.route('/')
def index():
    return render_template('index.html')

# Main model prediction
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    filename = secure_filename(file.filename)
    upload_folder = os.path.join('static', 'uploads')
    os.makedirs(upload_folder, exist_ok=True)

    filepath = os.path.join(upload_folder, filename)
    file.save(filepath)

    img = preprocess_image(filepath)
    pred = main_model.predict(img)[0]
    classes = ['Cancerous(nodule)', 'Normal','Pneumonia', 'TB']
    result = classes[np.argmax(pred)]
    confidence = pred[np.argmax(pred)] * 100

    return render_template("result.html",
                           prediction=result,
                           confidence=f"{confidence:.2f}%",
                           image_path=url_for('static', filename=f"uploads/{filename}"),
                           extra_msg="Try specific models for better results.")



# TB / Pneumonia model route
@app.route('/tb_pneumonia', methods=['GET', 'POST'])
def tb_pneumonia():
    if request.method == 'POST':
        file = request.files['image']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img = preprocess_image(filepath)
        pred = tb_pneumonia_model.predict(img)[0]
        classes = ['Normal', 'Pneumonia', 'TB']
        result = classes[np.argmax(pred)]
        confidence = pred[np.argmax(pred)] * 100

        return render_template("result.html",
                               prediction=result,
                               confidence=f"{confidence:.2f}%",
                               image_path=url_for('static', filename=f"uploads/{filename}"))

    return render_template("tb_pneumonia.html")

# Lung cancer model route
@app.route('/lung_cancer', methods=['GET', 'POST'])
def lung_cancer():
    if request.method == 'POST':
        file = request.files['image']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img = preprocess_image(filepath)
        pred = lung_cancer_model.predict(img)[0]
        classes = ['Nodule(cancerous)' ,'Normal']
        result = classes[np.argmax(pred)]
        confidence = pred[np.argmax(pred)] * 100

        return render_template("result.html",
                               prediction=result,
                               confidence=f"{confidence:.2f}%",
                               image_path=url_for('static', filename=f"uploads/{filename}"))

    return render_template("lung_cancer.html")

if __name__ == '__main__':  
  port = int(os.environ.get("PORT", 5000))
  app.run(host="0.0.0.0", port=port)

