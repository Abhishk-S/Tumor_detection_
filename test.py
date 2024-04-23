import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('Tumor10Epochs.h5')
image = cv2.imread('C:\\Users\\abhis\\Downloads\\Tumor_detection\\pred\\pred0.jpg')
image = Image.fromarray(image, 'RGB')
image = image.resize((64,64))
image = np.array(image)
image = np.expand_dims(image, axis = 0)

y_pred = model.predict(image)
#result = np.where(y_pred > 0.5, 1,0)
print (y_pred)#,result)

