import os
import warnings
import pandas as pd
import pickle as pk
from sklearn.metrics.pairwise import cosine_similarity


def get_sim_score(row):
    """
    Calculates similarity score between 2048 sized photo and match vectors
    input: a pandas dataframe row
    returns the cosine similarity score in floating points between -1 and 1
    """
    photo_feat = row.filter(like="photo").values.reshape(1, -1)
    match_feat = row.filter(like="match").values.reshape(1, -1)
    return cosine_similarity(photo_feat, match_feat)[0][0]


def predict_data(photo_data, matches_data):
    """
    Predicting, using the classification model, each match pair into one of three classes - 10 (FP), 11 (similar), and 12 (exact match).

    The function calls the extract_features function to extract photo and match features and combines them into
    a single dataframe so that inference can be carried out using the classifier model.
    """

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

    # adding two additional columns with cosine similarity score between the match and photo features.
    # One column for hand crafted features and another for deep learning model features
    query_array["sim_score_hand"] = query_array.filter(like="hand", axis=1).apply(
        get_sim_score, axis=1
    )
    query_array["sim_score_deep"] = query_array.filter(like="deep", axis=1).apply(
        get_sim_score, axis=1
    )

    # Load model from pickled file
    model_filename = os.path.join("ml", "models", "pickled_xgb_3label_model.pkl")
    with open(model_filename, "rb") as model_file:
        xgb_model = pk.load(model_file)

    # Inference
    query_pred = xgb_model.predict(query_array)
    proba = xgb_model.predict_proba(query_array)

    # Calculating the score using probabilities assigned to each label by the model
    prob = (proba[:, 0] * 0.3 + proba[:, 1] * 1 + proba[:, 2] * 2) / 2

    # Return predictions
    prediction_df = query_df[["photo_id", "match_id"]]
    prediction_df["FP_Prediction"] = query_pred
    prediction_df["Score"] = prob
    return prediction_df
