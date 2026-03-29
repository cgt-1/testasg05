import streamlit as st
import pandas as pd
import pickle
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.preprocessing import feature_engineering

st.title("ASG 05 MD - CallistaGianna - Spaceship Titanic")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(BASE_DIR, "model_pipeline.pkl")

with open(model_path, "rb") as f:
    pipeline = pickle.load(f)

# Inputs
Age = st.number_input("Age", value=25)
HomePlanet = st.selectbox("HomePlanet", ["Earth","Europa","Mars"])
CryoSleep = st.selectbox("CryoSleep", [True, False])
Destination = st.selectbox("Destination", ["TRAPPIST-1e","55 Cancri e","PSO J318.5-22"])
VIP = st.selectbox("VIP", [True, False])
RoomService = st.number_input("RoomService", 0)
FoodCourt = st.number_input("FoodCourt", 0)
ShoppingMall = st.number_input("ShoppingMall", 0)
Spa = st.number_input("Spa", 0)
VRDeck = st.number_input("VRDeck", 0)
Deck = st.selectbox("Deck", ["A","B","C","D","E","F","G"])
Side = st.selectbox("Side", ["P","S"])
PassengerId = st.text_input("PassengerId", "0001_01")

input_data = pd.DataFrame({
    "PassengerId": [PassengerId],
    "Name": ["Unknown"],
    "Age": [Age],
    "HomePlanet": [HomePlanet],
    "CryoSleep": [CryoSleep],
    "Destination": [Destination],
    "VIP": [VIP],
    "RoomService": [RoomService],
    "FoodCourt": [FoodCourt],
    "ShoppingMall": [ShoppingMall],
    "Spa": [Spa],
    "VRDeck": [VRDeck],
    "Cabin": [f"{Deck}/0/{Side}"]
})

if st.button("Predict"):
    input_data = feature_engineering(input_data)
    prediction = pipeline.predict(input_data)

    if prediction[0] == 1:
        st.success("Passenger Transported")
    else:
        st.error("Passenger Not Transported")