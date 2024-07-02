from flask import Flask, render_template, request,jsonify    #'jsonify': unknown word.
import pandas as pd
import pickle

app = Flask(__name__)
data = pd.read_csv('final_dataset.csv')
pipe = pickle.load(open("RidgeModel.pkl",'rb'))

@app.route('/')
def index():
    area = sorted(data['area'].unique())
    bedrooms = sorted(data['bedrooms'].unique())
    bathrooms = sorted(data['bathrooms'].unique())
    stories = sorted(data['stories'].unique())
    mainroad = sorted(data['mainroad'].unique())
    guestroom = sorted(data['guestroom'].unique())
    basement = sorted(data['basement'].unique())
    hotwaterheating = sorted(data['hotwaterheating'].unique())
    airconditioning = sorted(data['airconditioning'].unique())
    parking = sorted(data['parking'].unique())
    prefarea = sorted(data['prefarea'].unique())
    return render_template('index.html', area = area, bedrooms = bedrooms, bathrooms = bathrooms, stories = stories, mainroad = mainroad, guestroom = guestroom, basement = basement, hotwaterheating = hotwaterheating, airconditioning = airconditioning, parking = parking, prefarea = prefarea)
    

@app.route('/predict', methods=['POST'])
def predict():
    area = request.form.get('area')
    bedrooms = request.form.get('bedrooms')
    bathrooms = request.form.get('bathrooms')
    stories = request.form.get('stories')
    mainroad = request.form.get('mainroad')
    guestroom = request.form.get('guestroom')
    basement = request.form.get('basement')
    hotwaterheating = request.form.get('hotwaterheating')
    airconditioning = request.form.get('airconditioning')
    parking = request.form.get('parking')
    prefarea = request.form.get('prefarea')
    
    # Create a DataFrame with the input data
    input_data = pd.DataFrame([[area,bedrooms,bathrooms,stories,mainroad,guestroom,basement,hotwaterheating,airconditioning,parking,prefarea]],
                               columns=['area','bedrooms','bathrooms','stories','mainroad','guestroom','basement','hotwaterheating','airconditioning','parking','prefarea'])

    print("Input Data:")
    print(input_data)

    # Convert 'baths' column to numeric with errors='coerce'
    input_data['bathrooms'] = pd.to_numeric(input_data['bathrooms'], errors='coerce')

    # Convert input data to numeric types
    input_data = input_data.astype({'bedrooms': int, 'bathrooms': int, 'area': int, 'stories': int, 'mainroad' :int,'guestroom': int,'basement': int,'hotwaterheating': int,'airconditioning': int,'parking': int,'prefarea': int})

    # Handle unknown categories in the input data
    for column in input_data.columns:
        unknown_categories = set(input_data[column]) - set(data[column].unique())
        if unknown_categories:
            print(f"Unknown categories in {column}: {unknown_categories}")
            # Handle unknown categories (e.g., replace with a default value)
            input_data[column] = input_data[column].replace(unknown_categories, data[column].mode()[0])

    print("Processed Input Data:")
    print(input_data)

    # Predict the price
    prediction = pipe.predict(input_data)[0]
    print(prediction)
    return str(prediction)
    

if __name__ == "__main__":
    app.run(debug=True, port=5000)


if __name__ == "__main__":
    app.run(debug=True, port=5000)