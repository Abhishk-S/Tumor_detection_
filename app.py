import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
#from flask_ngrok import run_with_ngrok

app = Flask(__name__)
#run_with_ngrok(app) 

model = load_model('Tumor10Epochs.h5')

print ('Model loaded.')

def get_class(classNo):
    if classNo==0: return 'No Tumor detected'
    elif classNo==1: return 'Tumor detected'

def get_result(img):
    image = cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64,64))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    return  model.predict(image)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value=get_result(file_path)
        result=get_class(value) 
        return result
    return None


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)