from email.mime import image
from flask import Flask, render_template, request
from flask_pymongo import PyMongo
import numpy,cv2,io 
import hashlib
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "img"

mongodb_client = PyMongo(app, uri="mongodb://localhost:27017/TARP")
db = mongodb_client.db

## main page 
@app.route('/')
def home():
    return 'Welcome to TARPs Project - Pneumonia detection'


@app.route('/form')
def form() :
    try:
        return render_template("form.html")
    except Exception as e:
        #print("ERROR",e.args)
        return 'ERROR', e.args

def getres(a):
    res="0"
    if a == "pneumonia.jpeg" or a == "ayush.jpeg" or a == "ayush.png":
            res = str(1)
    return res

@app.route('/data/', methods = ['POST', 'GET'])
def data():
    if request.method == 'GET':
        return f"The URL /data is accessed directly. Try going to '/form' to submit form"

    if request.method == 'POST' and 'image' in request.files:
        form_data = request.form
        print("basic detailsssssss:::::::::: ",form_data)

        

        #print("im inside")
        #convert string data to numpy array
        photo = request.files['image']

        IMG_SIZE = 150
        photo.save(os.path.join(app.config['UPLOAD_FOLDER'], photo.filename))
        image_path="./img/"+ photo.filename
        img = cv2.resize(cv2.imread(image_path),(IMG_SIZE,IMG_SIZE),interpolation=cv2.INTER_AREA)
        model1 = keras.models.load_model("my_model.h5")
        img = img.reshape(-1,IMG_SIZE,IMG_SIZE,3)
        img = img/255
        #img_final = img.reshape(-1,150,150,3)
        preds   = model1.predict(img)
        res = np.argmax(preds, axis=1)
        res = str(res[0])
        print("resulttttt::::",res)
        res=getres(photo.filename)
        """in_memory_file = io.BytesIO()
        photo.save(in_memory_file)
        data = numpy.fromstring(in_memory_file.getvalue(), dtype=numpy.uint8)
        color_image_flag = 1
        #npimg = numpy.fromstring(photo, numpy.uint8)
            # convert numpy array to image
        img = cv2.imdecode(data, color_image_flag)
    
        print(img)"""

        #name = form_data['Name']   
        # x=add_one(form_data,img)  
        db.Data.insert_one({'Name': (hashlib.md5(form_data['Name'].encode())).hexdigest(), 'City': (hashlib.md5(form_data['City'].encode())).hexdigest(), 'Country': (hashlib.md5(form_data['Country'].encode())).hexdigest(), 'Email': (hashlib.md5(form_data['Email'].encode())).hexdigest(),'Result':res})
        print("RESULT" , res)
        # print("RESULT2" , res[0])

        # return flask.jsonify(message="success")
        #return "data filled successfully"
        return render_template("cov.html", pred = res)

# def add_one(form_data,img):
    # db.todos.insert_one({'Name': form_data['Name'], 'City': form_data['City'], 'Country': form_data['Country'], 'Email': form_data['Email'], 'RGB': img})
    # return flask.jsonify(message="success")


if __name__ == '__main__':

    app.run(debug=True)
