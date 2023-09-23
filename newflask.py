from flask import Flask, render_template, request, json, jsonify, send_file
import json
import pandas as pd
import numpy as np
import joblib
import gender_guesser.detector as gender
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load the model
model = joblib.load('fake_pro_model.pkl')

# Load the language mapping
lang_list = list(enumerate(np.unique(['lang1', 'lang2', 'lang3'])))  # Replace with actual languages
lang_dict = {name: i for i, name in lang_list}

# Function to predict user gender
def predict_user_sex(name):
    d = gender.Detector(case_sensitive=False)
    first_name = str(name).split(' ')[0]
    gen = d.get_gender(u"{}".format(first_name))

    gender_code_dict = {'female': -2, 'mostly_female': -1, 'unknown': 0, 'andy': 0, 'mostly_male': 1, 'male': 2}
    code = gender_code_dict[gen]

    return code

# Function to extract features from user input
def extract_user_features(input_data):
    lang_code = lang_dict.get(input_data['lang'], -1)
    gender_code = predict_user_sex(input_data['name'])

    features = {
        'statuses_count': float(input_data['statuses_count']),
        'followers_count': float(input_data['followers_count']),
        'friends_count': float(input_data['friends_count']),
        'favourites_count': float(input_data['favourites_count']),
        'listed_count': float(input_data['listed_count']),
        'gender_code': gender_code,
        'lang_code': lang_code
    }

    return features



# Route for the home page
@app.route('/')
def home():
    return render_template('landing_page.html')

# Route for the newhtml page
@app.route('/newhtml.html')
def new_html():
    return render_template('newhtml.html')

@app.route('/twitter.jpeg')
def showtwitter():
    image_path = 'C:/Users/rupaa/PycharmProjects/helloWorld/templates/twitter.jpeg'
    return send_file(image_path, mimetype='image/jpeg')

@app.route('/fb.jpeg')
def showfb():
    image_path = 'C:/Users/rupaa/PycharmProjects/helloWorld/templates/fb.jpeg'
    return send_file(image_path, mimetype='image/jpeg')

@app.route('/insta.jpeg')
def showinsta():
    image_path = 'C:/Users/rupaa/PycharmProjects/helloWorld/templates/insta.jpeg'
    return send_file(image_path, mimetype='image/jpeg')

@app.route('/others.jpeg')
def showothers():
    image_path = 'C:/Users/rupaa/PycharmProjects/helloWorld/templates/others.jpeg'
    return send_file(image_path, mimetype='image/jpeg')


# Route to handle prediction

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input
    user_input = request.form.to_dict()

    # Extract features from input
    features = extract_user_features(user_input)

    # Predict using the model
    prediction = model.predict([list(features.values())])[0]

    result = 'Fake' if prediction == 1 else 'Real'

    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
