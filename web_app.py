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

application = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@application.route("/health", methods=["GET"])
def health():
    return jsonify({"Message": "Service OK"}), 200


@application.route("/upload", methods=["POST"])
def upload():
    f = request.files['image']
    filename="  img-" + str(uuid.uuid4())
    f.save(secure_filename(filename))
    return jsonify({"Message":"Image Uploaded","FileName":filename}),200

@application.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    result,score,status = load_and_predict(data)
    return jsonify({"Status":status, "Prediction":result,"Confidence":score}),200

def load_and_predict(data):
    #This function is not tested
    try:
        tmp = data["temperature"]
        p02 = data["pO2_saturation"]
        leu_cnt = data["leukocyte_count"]
        neu_cnt = data["neutrophil_count"]
        lym_cnt = data["lymphocyte_count"]
        img_name = data["file_name"]
        rf_model_name=""   # add filename
        tf_model_name=""   #  add filename
        rf_model = joblib.load(rf_model_name)
        trnsfr_learning_model = keras.models.load_model(tf_model_name)
        rf_probs = rf_model.predict_proba([tmp,po2,leu_cnt,neu_cnt,lym_cnt])
        tf_probs = trnsfr_learning_model.predict_proba([img_name]) #might need to pre-process it maybe
        rf_probs*tf_probs
        final_probs = [x*y for x,y in zip(rf_probs,tf_probs)]
        label = np.argmax(final_probs)
        if label:
            return "postive",final_probs[label] ,"Success"
        else:
            return "negative",final_probs[label] ,"Success"
    except:
        return None,None,"Error"

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=8889)

 
app = Flask(__name__)
def do_something(text1,text2):
   Health = Health.upper()
   Predict = Predict.upper()
   combine = Health + Predict 
   return combine
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/join', methods=['GET','POST'])
def my_form_post():
    Health = request.form['Health']
    word = request.args.get('text1')
    Predict = request.form['Predict']
    combine = do_something(Health,Predict)
    result = {
        "output": combine
    }
    result = {str(key): value for key, value in result.items()}
    return jsonify(result=result)
if __name__ == '__main__':
    app.run(debug=True)
