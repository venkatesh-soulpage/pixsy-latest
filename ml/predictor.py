import os
import dask.dataframe as dd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler
from sklearn import metrics
import pickle
import pandas as pd
import argparse
import time
import json


import warnings

# from kafkaservice import kafkaproducer


warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--Photo", "-P", help="path to file containig photo id")
parser.add_argument(
    "--Matches", "-M", help="path to file with list of matches returned by crawler"
)

args = parser.parse_args()

# parsing photo and match query ids
with open(args.Photo, "r") as p_id, open(args.Matches, "r") as m_ids:
    photo_query = p_id.read()
    match_queries = m_ids.read().split(",")


photo_queryset = Photo.query.filter_by(name=photo_query).first()
formatted_data = {"photo_id": photo_queryset.id}
formatted_data.update(photo_queryset.data.items())

match_queryset = Match.query.filter(Match.name.in_(match_queries)).all()
match_formatted_data = []
for match in match_queryset:
    data = {"match_id": match.id}
    data.update(match.data.items())
    match_formatted_data.append(data)


photo_query_feature = pd.DataFrame.from_dict(formatted_data, orient="index")
match_query_features = pd.DataFrame(data=match_formatted_data)


photo_query_feature = photo_query_feature.transpose()
photo_query_feature.reset_index(drop=True)


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
model_filename = os.path.join("models", "pickled_mlp_model.pkl")
scaler_filename = os.path.join("models", "pickled_scaler.pkl")

with open(model_filename, "rb") as model_file, open(
    scaler_filename, "rb"
) as scaler_file:
    mlp = pickle.load(model_file)
    scaler = pickle.load(scaler_file)

start = time.time()
# query_array_scaled = scaler.fit_transform(query_array)
query_pred = mlp.predict(query_array)
probabilities = mlp.predict_proba(query_array)
end = time.time()
print("Prediction done")
print(
    "Time taken for inference on {} pairs is {} seconds".format(
        query_array.shape[0], end - start
    )
)

query_df.rename(columns={"photo_id": "photo", "match_id": "match"}, inplace=True)
prediction_df = query_df[["photo", "match"]]
prediction_df["status"] = query_pred
prediction_df["score"] = probabilities[:, 1]
prediction_df.set_index("photo", inplace=True)

# print(prediction_df.head())
prediction_data = json.loads(prediction_df.to_json(orient="table"))
prediction_data = prediction_data["data"]
# kafkaproducer(prediction_data, "prediction")

print("Done!")
