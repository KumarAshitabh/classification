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
    #Process first image
    file = request.files['image1']
    image = Image.open(file.stream)
    size = 64, 64
    image = image.resize(size, Image.ANTIALIAS)
    print("done loading")
    image = asarray(image)[:,:,0]
    #data = Image.fromarray(image)
    #data.save('gfg_dummy_pic.png')
    predicted1 = model.predict(image)

    #Process 2nd image
    file = request.files['image2']
    image = Image.open(file.stream)
    size = 64, 64
    image = image.resize(size, Image.ANTIALIAS)
    print("done loading")
    image = asarray(image)[:,:,0]
    #data = Image.fromarray(image)
    #data.save('gfg_dummy_pic.png')
    predicted2 = model.predict(image)

    #Compare Predicted1 and predicted2
    if int(predicted1[0]) == int(predicted2[0]):
        return {"Message":"Both images not same"}
    else:
        return {"Message":"Both images not same"}

    

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)