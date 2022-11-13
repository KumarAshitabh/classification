from flask import Flask
from flask import request
from joblib import load
import pickle
from PIL import Image
import numpy
from skimage import io, color
from numpy import asarray

app = Flask(__name__)
model_path = "model/svm.sav"
model = pickle.load(open(model_path, 'rb'))

@app.route("/")
def hello_world():
    return "<!-- hello --> <b> Hello, World!</b>"


# get x and y somehow    
#     - query parameter
#     - get call / methods
#     - post call / methods ** 

@app.route("/sum", methods=['POST'])
def sum():
    x = request.json['x']
    y = request.json['y']
    z = x + y 
    return {'sum':z}



@app.route("/predict", methods=['POST'])
def predict_digit():
    file = request.files['image']
    # Read the image via file.stream
    image = Image.open(file.stream)
    size = 64, 64
    image = image.resize(size, Image.ANTIALIAS)
    print("done loading")
    image = asarray(image)[:,:,0]
    predicted = model.predict(image)
    return {"y_predicted":int(predicted[0])}

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)