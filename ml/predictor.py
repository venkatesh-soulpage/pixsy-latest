import os
import dask.dataframe as dd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler
from sklearn import metrics
import pickle as pk
import pandas as pd
import argparse
import time
import json


def predict_data(photo_data, matches_data):
    photo_query_feature = pd.read_json(photo_data)
    match_query_features = pd.read_json(matches_data)
    # Creating a dataframe in required format for classifier by combining photo and match features
    reps = match_query_features.shape[0]
    photo_query_features = pd.concat([photo_query_feature] * reps, axis=0).reset_index(
        drop=True
    )
    query_df = photo_query_features.merge(
        match_query_features,
        how="outer",
        left_index=True,
        right_index=True,
        suffixes=("_photo", "_match"),
    )

    query_array = query_df.drop(["photo_id", "match_id"], axis="columns")

    # Load model and scaler from pickle files
    model_filename = os.path.join("ml", "models", "pickled_mlp_model.pkl")
    scaler_filename = os.path.join("ml", "models", "pickled_scaler.pkl")

    with open(model_filename, "rb") as model_file, open(
        scaler_filename, "rb"
    ) as scaler_file:
        mlp = pk.load(model_file)
        scaler = pk.load(scaler_file)

    start = time.time()
    # query_array_scaled = scaler.fit_transform(query_array)
    query_pred = mlp.predict(query_array)
    probabilities = mlp.predict_proba(query_array)

    print("saving predictions to csv")
    prediction_df = query_df[["photo_id", "match_id"]]
    prediction_df["FP_Prediction"] = query_pred
    prediction_df["FP_Prob_Score"] = probabilities[:, 1]
    return prediction_df