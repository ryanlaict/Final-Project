# app.py

import streamlit as st
import pandas as pd
from recommender_utils import (
    calculate_per, preprocess_data, cluster_players,
    build_similarity_matrix, recommend_wnba_players_by_input
)

@st.cache_data
def load_and_prepare_data():
    data = pd.read_csv("combined_data_encoded.csv")
    data = calculate_per(data)
    data, feature_cols = preprocess_data(data)
    data = cluster_players(data, feature_cols)
    similarity_matrix = build_similarity_matrix(data, feature_cols)
    return data, similarity_matrix

# Load data
combined_data_encoded, similarity_matrix = load_and_prepare_data()

# App UI
st.title("🏀 WNBA Player Recommender")
st.write("Get a WNBA player recommendation based on your NBA preferences!")

search_mode = st.selectbox("Recommend based on:", ["NBA Player", "NBA Team", "City"])

if search_mode == "NBA Player":
    user_input = st.text_input("Enter your favorite NBA player:")
elif search_mode == "NBA Team":
    user_input = st.text_input("Enter your favorite NBA team:")
else:
    user_input = st.text_input("Enter your city:")

if user_input:
    recommendations = recommend_wnba_players_by_input(
        user_input, combined_data_encoded, similarity_matrix, search_mode
    )
    if isinstance(recommendations, str):
        st.warning(recommendations)
    else:
        st.success("Recommended WNBA Players:")
        st.dataframe(recommendations)
