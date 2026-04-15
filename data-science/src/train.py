# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Trains ML model using training dataset and evaluates using test dataset. Saves trained model.
"""

import argparse
import mlflow
import mlflow.sklearn

import pandas as pd

from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("train")
    
    # -------- WRITE YOUR CODE HERE --------
    
    # Step 1: Define arguments for train data, test data, model output, and RandomForest hyperparameters. Specify their types and defaults.  

    parser.add_argument("--train_data", type=str, help = "Path to the training data")
    parser.add_argument("--test_data", type=str, help = "Path to the test data")
    parser.add_argument("--n_estimators", type=int, default=100, help = "Number of trees in the random forest regressor")
    parser.add_argument("--max_depth", type=int, default=None, help = "Maximum depth of each tree in the random forest.  If set to None, nodes are expanded until all leaves are pure or contain less than min_samples_split samples.")
    parser.add_argument("--model_output", type=str, help = "Path to the output model")

    args = parser.parse_args()

    return args

def main(args):
    '''Read train and test datasets, train model, evaluate model, save trained model'''

    # -------- WRITE YOUR CODE HERE --------

    # Step 2: Read the train and test datasets from the provided paths using pandas. Replace '_______' with appropriate file paths and methods.

    # Load datasets
    train_df = pd.read_csv(Path(args.train_data)/"data.csv")
    test_df = pd.read_csv(Path(args.test_data)/"data.csv")

    
    # Step 3: Split the data into features (X) and target (y) for both train and test datasets. Specify the target column name.

    # Assign the price column to y_train and y_test, then drop the price column from the dataframes to create X_train and X_test
    y_train = train_df["price"].values
    X_train = train_df.drop("price", axis=1).values
    y_test = test_df["price"].values
    X_test = test_df.drop("price", axis=1).values

    
    # Step 4: Initialize the RandomForest Regressor with specified hyperparameters, and train the model using the training data.

    # Initialize and train a random forest regressor, assigning a random_state value for reproducibility
    # First, convert the sentinel value of -1 for max_depth to None if it is in args.max_depth
    max_depth = args.max_depth
    if max_depth == -1:
        max_depth = None
    rfr_model = RandomForestRegressor(n_estimators=args.n_estimators, max_depth=max_depth, random_state=42)
    rfr_model.fit(X_train, y_train)

    
    # Step 5: Log model hyperparameters like 'n_estimators' and 'max_depth' for tracking purposes in MLflow.
    
    # Log model hyperparameters
    mlflow.log_param("model", "RandomForestRegressor")  # Provide the model name
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)

    
    # Step 6: Predict target values on the test dataset using the trained model, and calculate the mean squared error.

    # Predict price values for the test data and compute the MSE
    rfr_predictions = rfr_model.predict(X_test)

    mse = mean_squared_error(y_test, rfr_predictions)
    print('Mean squared error on test data: {mse:.2f}')
    

    
    # Step 7: Log the MSE metric in MLflow for model evaluation, and save the trained model to the specified output path.

    # log the MSE
    mlflow.log_metric("MSE", float(mse))

    # Save the trained model
    mlflow.sklearn.save_model(sk_model=rfr_model, path=args.model_output)


if __name__ == "__main__":
    
    # Parse Arguments
    args = parse_args()

    lines = [
        f"Train dataset input path: {args.train_data}",
        f"Test dataset input path: {args.test_data}",
        f"Model output path: {args.model_output}",
        f"Number of Estimators: {args.n_estimators}",
        f"Max Depth: {args.max_depth}"
    ]

    for line in lines:
        print(line)

    with mlflow.start_run():
        main(args)

    

