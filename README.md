
This is a Flask-based web application for detecting lung diseases such as **TB**, **Pneumonia**, **Lung Cancer (Nodule)**, and **Normal** conditions using chest X-ray images. It uses trained deep learning models and includes infection visualization.

---

##  Features

- Predicts whether an X-ray shows Normal, TB, Pneumonia, or Cancerous Nodules.
- Uses 3 different models:
  - Main Combined Model
  - TB/Pneumonia Classification Model
  - Lung Cancer (Nodule) Detection Model
-  Visualizes infected areas on the X-ray in collab still not updated in app.py
-  Models are automatically downloaded from Google Drive using `gdown`.

---

##  Tech Stack

- Python 3
- Flask
- TensorFlow / Keras
- OpenCV
- NumPy
- Google Drive + gdown for model loading

---

# Dataset
- download it from the drive link given below
https://drive.google.com/drive/folders/1qVO5ZTHXiS0Spb0JcTL5KQ3LDTaVOghf?usp=sharing

# Install dependencies
pip install -r requirements.txt

# Run the flask app in terminal
python app.py
- Visit http://127.0.0.1:5000 in your browser

#  Author
Developed by Bharath Kumar Taddi
 GitHub - https://github.com/bharathkumar1207 
 LinkedIn - www.linkedin.com/in/bharath-kumar-81a3632ba

# License
This project is for educational purposes only. All rights reserved.






