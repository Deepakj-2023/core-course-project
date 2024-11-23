import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load CSV data
precautions_df = pd.read_csv("C:/Users/Deepak/Desktop/core_course_project/datasets/added dataset/precautions_df.csv")
workout_df = pd.read_csv("C:/Users/Deepak/Desktop/core_course_project/datasets/added dataset/workout_df.csv")
medications_df = pd.read_csv("C:/Users/Deepak/Desktop/core_course_project/datasets/added dataset/medications.csv")
description_df = pd.read_csv("C:/Users/Deepak/Desktop/core_course_project/datasets/added dataset/description.csv")
diets_df = pd.read_csv("C:/Users/Deepak/Desktop/core_course_project/datasets/added dataset/diets.csv")
hospitals_df = pd.read_csv("C:/Users/Deepak/Desktop/core_course_project/datasets/added dataset/hospitals.csv")  # Assuming hospital dataset has ratings



app = Flask(__name__)

def predict(values, dic):
    if len(values) == 8:
        model = pickle.load(open('models/diabetes.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
    elif len(values) == 26:
        model = pickle.load(open('models/breast_cancer.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
    elif len(values) == 13:
        model = pickle.load(open('models/heart.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
    elif len(values) == 18:
        model = pickle.load(open('models/kidney.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
    elif len(values) == 10:
        model = pickle.load(open('models/liver.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/diabetes", methods=['GET', 'POST'])
def diabetesPage():
    return render_template('diabetes.html')

@app.route("/cancer", methods=['GET', 'POST'])
def cancerPage():
    return render_template('breast_cancer.html')

@app.route("/heart", methods=['GET', 'POST'])
def heartPage():
    return render_template('heart.html')

@app.route("/kidney", methods=['GET', 'POST'])
def kidneyPage():
    return render_template('kidney.html')

@app.route("/liver", methods=['GET', 'POST'])
def liverPage():
    return render_template('liver.html')

@app.route("/malaria", methods=['GET', 'POST'])
def malariaPage():
    return render_template('malaria.html')

@app.route("/pneumonia", methods=['GET', 'POST'])
def pneumoniaPage():
    return render_template('pneumonia.html')
#----------------------------------------------------------------------------------------------
def get_additional_info(disease_name):
    # Retrieve and process description
    desc_series = description_df[description_df['Disease'] == disease_name]['Description']
    desc = desc_series.tolist()[0] if not desc_series.empty else "No description available."
    
    # Retrieve and process precautions
    pre_df = precautions_df[precautions_df['Disease'] == disease_name][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    if not pre_df.empty:
        pre = pre_df.values.flatten()
        pre = [precaution for precaution in pre if pd.notna(precaution)]
    else:
        pre = ["No precautions available."]
    
    # Retrieve and process medications
    med_series = medications_df[medications_df['Disease'] == disease_name]['Medication']
    med = med_series.tolist() if not med_series.empty else ["No medications available."]
    
    # Retrieve and process diets
    die_series = diets_df[diets_df['Disease'] == disease_name]['Diet']
    die = die_series.tolist() if not die_series.empty else ["No diets available."]
    
    # Retrieve and process workouts
    wrkout_series = workout_df[workout_df['disease'] == disease_name]['workout']
    wrkout = wrkout_series.tolist() if not wrkout_series.empty else ["No workout information available."]
    
    # Ensure sorting and selecting top hospitals are correct
    top_hospitals = hospitals_df.sort_values(by='rating', ascending=False).head(5)
    
    return {
        "description": desc,
        "diets": die,
        "medications": med,
        "precautions": pre,
        "workout": wrkout,
        "hospitals": top_hospitals
    }




#------------------------------------------------------------------------------------------
@app.route("/predict", methods=['POST', 'GET'])
def predictPage():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            pred = predict(to_predict_list, to_predict_dict)

            # Map the prediction to a disease name
            disease_name_mapping = {8: "Diabetes", 26: "Breast Cancer", 13: "Heart Disease", 18: "Kidney Disease", 10: "Liver Disease"}
            disease_name = disease_name_mapping.get(len(to_predict_list), "Healthy")

            if disease_name != "Healthy":
                additional_info = get_additional_info(disease_name)
                return render_template('predict.html', pred=pred, additional_info=additional_info)
            else:
                return render_template('predict.html', pred=pred)

    except Exception as e:
        message = f"Error: {str(e)}. Please enter valid data."
        return render_template("home.html", message=message)


#---------------------------------------------------------------------------------------------------------
# @app.route("/malariapredict", methods = ['POST', 'GET'])
# def malariapredictPage():
#     if request.method == 'POST':
#         try:
#             if 'image' in request.files:
#                 img = Image.open(request.files['image'])
#                 img = img.resize((36,36))
#                 img = np.asarray(img)
#                 img = img.reshape((1,36,36,3))
#                 img = img.astype(np.float64)
#                 model = load_model("models/malaria.h5")
#                 pred = np.argmax(model.predict(img)[0])
#         except:
#             message = "Please upload an Image"
#             return render_template('malaria.html', message = message)
#     return render_template('malaria_predict.html', pred = pred)

# @app.route("/pneumoniapredict", methods = ['POST', 'GET'])
# def pneumoniapredictPage():
#     pred = 0
#     if request.method == 'POST':
#         try:
#             if 'image' in request.files:
#                 img = Image.open(request.files['image'])
#                 img.show()
#                 img = img.convert('L')
#                 img = img.resize((36,36))
#                 img = np.asarray(img)
#                 img = img.reshape((1,36,36,1))
#                 # img.show()
#                 # print(img)
#                 img = img / 255.0
#                 model = load_model("models/pneumonia.h5")
#                 print(img.shape)
#                 pred = np.argmax(model.predict(img)[0])
#         except Exception as e:
#             message = "Please upload an Image"
#             # message = e
#             print(e)
#             return render_template('pneumonia.html', message = message)
#     return render_template('pneumonia_predict.html', pred = pred)

if __name__ == '__main__':
	app.run(debug = True)
