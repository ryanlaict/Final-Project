# app.py

import streamlit as st
import pandas as pd
from recommender_utils import (
    calculate_per, preprocess_data, cluster_players,
    build_similarity_matrix, recommend_wnba_players
)

@st.cache_data
def load_and_prepare_data():
    data = pd.read_csv("combined_data_encoded.csv)
    data = calculate_per(data)
    data, feature_cols = preprocess_data(data)
    data = cluster_players(data, feature_cols)
    similarity_matrix = build_similarity_matrix(data, feature_cols)
    return data, similarity_matrix

# Load processed data
combined_data_encoded, similarity_matrix = load_and_prepare_data()

# UI
st.title("🏀 WNBA Player Recommender")
st.write("Enter your favorite NBA player and get WNBA recommendations!")

player_input = st.text_input("Favorite NBA Player:", "")

if player_input:
    result = recommend_wnba_players(player_input, combined_data_encoded, similarity_matrix)
    if isinstance(result, str):
        st.warning(result)
    else:
        st.success("Here are your WNBA recommendations:")
        st.dataframe(result)