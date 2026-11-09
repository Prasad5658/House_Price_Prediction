
from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Loading the trained Random Forest model package
with open("RandomForest_HousePrice.pkl", "rb") as f:
    model_package = pickle.load(f)

# Extracting model and saved information
model = model_package['model']
category_mappings = model_package['category_mappings']
feature_names = model_package['feature_names']


@app.route("/")
def fun():
    return render_template("index.html")


@app.route("/predict", methods=['GET', 'POST'])
def prediction():

    if request.method == 'POST':

        try:
            # Getting numerical inputs
            numeric_features = [
                'bedrooms',
                'bathrooms',
                'sqft_living',
                'sqft_lot',
                'floors',
                'waterfront',
                'view',
                'condition',
                'sqft_above',
                'sqft_basement',
                'yr_built',
                'yr_renovated'
            ]

            value = []

            for name in numeric_features:
                value.append(float(request.form[name]))

            # Getting categorical inputs
            city = request.form['city']
            statezip = request.form['statezip']
            country = request.form['country']

            # Converting categories into numerical values
            city_code = category_mappings['city'].get(city)
            statezip_code = category_mappings['statezip'].get(statezip)
            country_code = category_mappings['country'].get(country)

            # Checking for unknown categories
            if city_code is None or statezip_code is None or country_code is None:
                return render_template(
                    "index.html",
                    prediction_text="Please enter valid City, State ZIP and Country values."
                )

            # Adding categorical values
            value.extend([
                city_code,
                statezip_code,
                country_code
            ])

            # Creating DataFrame
            value = pd.DataFrame(
                [value],
                columns=feature_names
            )

            # Predicting house price
            sol = model.predict(value)[0]

            return render_template(
                "index.html",
                prediction_text=f"Predicted House Price: {sol:,.2f}"
            )

        except (ValueError, KeyError) as e:
            return render_template(
                "index.html",
                prediction_text="Please enter valid values in all fields."
            )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
