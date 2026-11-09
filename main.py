
import pandas as pd
import numpy as np
import pickle
import sys

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error


class RandomForest:

    def __init__(self, path):
        try:
            # Reading dataset
            self.df = pd.read_csv(path)

            # Dropping date and street columns
            self.df = self.df.drop(['date', 'street'], axis=1)

            # Creating category mappings
            self.category_mappings = {}

            # Factorizing categorical columns
            for col in ['city', 'statezip', 'country']:

                codes, categories = pd.factorize(self.df[col])

                self.category_mappings[col] = {
                    str(category): int(i)
                    for i, category in enumerate(categories)
                }

                self.df[col] = codes

            # Removing houses with zero price
            self.df = self.df[self.df['price'] > 0]

            # Removing top 0.1% price records
            self.price_limit = self.df['price'].quantile(0.999)

            self.df = self.df[
                self.df['price'] < self.price_limit
            ]

            # Independent and dependent variables
            self.X = self.df.iloc[:, 1:]
            self.y = self.df.iloc[:, 0]

            # Saving feature names
            self.feature_names = list(self.X.columns)

            # Splitting dataset
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X,
                self.y,
                test_size=0.2,
                random_state=42
            )

            print("Training records:", len(self.X_train))
            print("Testing records:", len(self.X_test))
            print("Total features:", len(self.feature_names))

        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            print(
                f"Error in Line no: {er_line} "
                f"Due to {er_ty}: {er_msg}"
            )

    def training(self):
        try:
            # Creating Random Forest model
            self.reg = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                random_state=42
            )

            # Training model
            self.reg.fit(self.X_train, self.y_train)

            # Training predictions
            self.train_predictions = self.reg.predict(
                self.X_train
            )

            print("\nTraining Performance")

            print(
                "Training RMSE:",
                root_mean_squared_error(
                    self.y_train,
                    self.train_predictions
                )
            )

            print(
                "Training R2 Score:",
                r2_score(
                    self.y_train,
                    self.train_predictions
                )
            )

        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            print(
                f"Error in Line no: {er_line} "
                f"Due to {er_ty}: {er_msg}"
            )

    def testing(self):
        try:
            # Testing predictions
            self.test_predictions = self.reg.predict(
                self.X_test
            )

            print("\nTesting Performance")

            print(
                "Testing RMSE:",
                root_mean_squared_error(
                    self.y_test,
                    self.test_predictions
                )
            )

            print(
                "Testing R2 Score:",
                r2_score(
                    self.y_test,
                    self.test_predictions
                )
            )

        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            print(
                f"Error in Line no: {er_line} "
                f"Due to {er_ty}: {er_msg}"
            )

    def model_saving(self):
        try:
            # Creating model package
            model_package = {
                'model': self.reg,
                'category_mappings': self.category_mappings,
                'feature_names': self.feature_names,
                'price_limit': self.price_limit
            }

            # Saving model package using Pickle
            with open("RandomForest_HousePrice.pkl", "wb") as f:
                pickle.dump(model_package, f)

            print("\nModel package saved successfully!")

            # Loading saved model package
            with open("RandomForest_HousePrice.pkl", "rb") as p:
                loaded_package = pickle.load(p)

            # Extracting model
            loaded_model = loaded_package['model']

            # Checking prediction after loading
            result = loaded_model.predict(
                self.X_test.iloc[[0]]
            )[0]

            print(
                "Prediction after loading model:",
                result
            )

        except Exception as e:
            er_ty, er_msg, er_line = sys.exc_info()
            print(
                f"Error in Line no: {er_line} "
                f"Due to {er_ty}: {er_msg}"
            )


if __name__ == "__main__":

    obj = RandomForest("data.csv")

    obj.training()

    obj.testing()

    obj.model_saving()
