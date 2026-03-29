import pandas as pd
import pickle

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from src.features.preprocessing import feature_engineering


def run_pipeline():

    df = pd.read_csv("train.csv")

    df = feature_engineering(df)

    y = df["Transported"].astype(int)
    X = df.drop("Transported", axis=1)

    categorical_cols = [
        "HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side"
    ]

    numerical_cols = [
        "Age", "RoomService", "FoodCourt", "ShoppingMall",
        "Spa", "VRDeck", "Cabin_num", "Group_size", "TotalSpending"
    ]

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    preprocessor = ColumnTransformer([
        ("cat", categorical_pipeline, categorical_cols),
        ("num", numerical_pipeline, numerical_cols)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X, y)

    with open("model_pipeline.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    print("Pipeline trained and saved!")

    return pipeline