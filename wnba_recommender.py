import streamlit as st
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

combined_data = pd.read_csv("combined_data.csv")
combined_data_encoded = pd.read_csv("combined_data_encoded.csv")
def recommend_wnba_players(nba_player, top_n=5):
    if nba_player not in combined_data_encoded["Player"].values:
        return "Player not found in dataset."

    nba_cluster = combined_data_encoded.loc[combined_data_encoded["Player"] == nba_player, "Playstyle_Cluster"].values[0]
    nba_city_encoded = combined_data_encoded[combined_data_encoded["Player"] == nba_player][[col for col in combined_data_encoded.columns if col.startswith("City_")]].values[0]
    
    wnba_candidates = combined_data_encoded[(combined_data_encoded["League_WNBA"] == 1)]
    wnba_candidates['City_Match'] = wnba_candidates[[col for col in combined_data_encoded.columns if col.startswith("City_")]].apply(
        lambda x: int(any(x == nba_city_encoded)), axis=1)
    
    if "Starter" in wnba_candidates.columns:
        wnba_candidates = wnba_candidates.sort_values(by=["Starter", "City_Match", "PER"], ascending=[False, False, False])
    else:
        wnba_candidates = wnba_candidates.sort_values(by=["City_Match", "PER"], ascending=[False, False])
    
    similar_wnba_players = nba_to_wnba_similarity.loc[nba_player, wnba_candidates["Player"]].sort_values(ascending=False).head(top_n)

    team_columns = [col for col in combined_data_encoded.columns if col.startswith("Team Name_")]
    city_columns = [col for col in combined_data_encoded.columns if col.startswith("City_")]

    wnba_candidates = combined_data_encoded[combined_data_encoded["Player"].isin(similar_wnba_players.index)].copy()
    wnba_candidates['Team'] = wnba_candidates[team_columns].idxmax(axis=1).str.replace("Team Name_", "")
    wnba_candidates['City'] = wnba_candidates[city_columns].idxmax(axis=1).str.replace("City_", "")

    return wnba_candidates[["Player", "Team", "City"]].drop_duplicates()

# Streamlit interface
st.title("WNBA Recommender System")

# Input section for NBA player/team
favorite_nba_player = st.text_input("Enter your favorite NBA player:")
favorite_nba_team = st.text_input("Or enter your favorite NBA team:")

# Process the input and generate recommendation
if favorite_nba_player:
    wnba_recommendations = recommend_wnba_players(favorite_nba_player, top_n=3)
    st.subheader(f"Recommended WNBA Players for {favorite_nba_player}")
    st.write(wnba_recommendations)
elif favorite_nba_team:
    st.write(f"Recommendations for WNBA players from {favorite_nba_team} team coming soon!")
else:
    st.write("Please enter a favorite NBA player or team to get recommendations.")
