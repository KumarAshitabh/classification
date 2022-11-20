from flask import Flask
from flask import request
from joblib import load
import pickle
from PIL import Image
import numpy
from skimage import io, color
from numpy import asarray
import os

app = Flask(__name__)


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
    fileimg = request.files['image1']
    modelname = request.form['model']
    model_path = ""
    accuracy_max = 0.0
    accuracy_file = ""
    if len(modelname)>0: # If model name is provided
        #model_path = "model/"+modelname.lower()+".sav"
        model_path = "tmp/"+modelname.lower()+".sav"
    else : # If model name is not provided
        directory = os.getcwd()+"/report"
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)   
            file = open(f)
            for line in file:
                fields = line.strip().split()
                if float(accuracy_max) < float(fields[0]):
                    accuracy_max = float(fields[0])
                    accuracy_file = file
        if 'svm' in str(accuracy_file) :
            model_path = "tmp/"+"svm"+".sav"
        else:
            model_path = "tmp/"+"tree"+".sav"

    print(model_path)
    model = pickle.load(open(model_path, 'rb'))
    image = Image.open(fileimg.stream)
    size = 64, 64
    image = image.resize(size, Image.ANTIALIAS)
    print("done loading")
    image = asarray(image)[:,:,0]
    #data = Image.fromarray(image)
    #data.save('gfg_dummy_pic.png')
    predicted1 = model.predict(image)

    return {"Message":"Predicted digit is "+str(predicted1[0])}


    

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)