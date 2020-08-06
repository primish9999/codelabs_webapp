import os
import uuid
from flask import Flask, jsonify, request
import datetime
import base64
import json
from werkzeug.utils import secure_filename
import joblib
from flask import Flask, request, render_template,jsonify
from tensorflow import keras
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
# import app 
application = Flask(__name__)
global_file_name=""
@application.route('/', methods=["GET","POST"])
def index():
    print()
    return render_template('index.html')
@application.route("/health", methods=["GET"])
def health():
    return jsonify({"Message": "Service OK"}), 200
@application.route("/upload", methods=["GET","POST"])
def upload():
    f = request.files['image']
    filename="img-" + str(uuid.uuid4())
    f.save(secure_filename(filename))
    return render_template('index.html',image=filename)
    # return jsonify({"Message":"Image Uploaded","FileName":filename}),200
@application.route("/predict", methods=["POST"])
def predict():
    data = {
    "temperature":request.form["temperature"],
    "pO2_saturation":request.form["pO2_saturation"],
    "leukocyte_count":request.form["leukocyte_count"],
    "neutrophil_count": request.form["neutrophil_count"],
    "lymphocyte_count": request.form["lymphocyte_count"],
    "file_name":request.form["file_name"]
    }
    print(data)
    result,score,status = load_and_predict(data)
    return jsonify({"Status":status, "Prediction":result,"Confidence":score}),200
def load_and_predict(data):
    #This function is not tested
    # try:
        tmp = data["temperature"]
        po2 = data["pO2_saturation"]
        leu_cnt = data["leukocyte_count"]
        neu_cnt = data["neutrophil_count"]
        lym_cnt = data["lymphocyte_count"]
        img_name = data["file_name"]
        rf_model_name="finalized_RFmodel.sav"   # add filename
        tf_model_name="chest_xrayimage_covidmodel_4.h5"   #  add filename
        rf_model = joblib.load(rf_model_name)
        trnsfr_learning_model = keras.models.load_model(tf_model_name)
        rf_probs = rf_model.predict(np.array([tmp,po2,leu_cnt,neu_cnt,lym_cnt]).reshape(1,-1))
        IMG_DIM = (300, 300)
        img = [img_to_array(load_img(img_name, target_size = IMG_DIM)) ]
        # print('Image Shape: ',img.shape)
        img = np.array(img)
        img = img.astype('float32')
        img /= 255
        tf_probs = trnsfr_learning_model.predict([img]) #might need to pre-process it maybe
        print('Image Shape After: ',img.shape)
        if rf_probs[0]==1:
            if tf_probs[0][0]>0.4:
                return "postive",str(tf_probs[0][0]) ,"Success"
            else:
                return "negative",str(tf_probs[0][0]),"Success"
        else:
            if tf_probs[0][0]<0.6:
                return "negative",str(tf_probs[0][0]) ,"Success"
            else:
                return "postive",str(tf_probs[0][0]) ,"Success"
    # except:
        # return None,None,"Error"
if __name__ == '__main__':
    application.run(host='0.0.0.0', port=8889)
## Route declaration##
	# from flask import current_app as app
	# from flask import render_template
	# 
	# 
	# # @app.route('/')
	# # def home():
	# #    ##Landing page## 
	# #     return render_template('index.html',
	# #                            title="ISCovid Site",
	# #                            description="Upload X-Ray")
	# Collapse
