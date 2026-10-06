from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image

#flask for web app
app = Flask(__name__)

#load the ML knowledge
model = tf.keras.models.load_model('model.h5')

#route enpoint API
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file=request.files['image']
    img=Image.open(file).convert('RGB')
    img=img.resize((180,180))
    #NumPy pixels array
    img_array=np.array(img,dtype=np.float32)
    img_array=np.expand_dims(img_array, axis=0)
    #sigmoid 0 or 1 dog or cat
    prediction=model.predict(img_array)
    if(prediction[0][0]>0.5):
        prediction='Dog'
    else:
        prediction='Cat'
    #rendering to HTML page
    return render_template('index.html', prediction=prediction)

if __name__=='__main__':
    app.run(debug=True)
    