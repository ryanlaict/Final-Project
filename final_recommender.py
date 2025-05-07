#Final WNBA Recommender
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Load your data
combined_data_encoded = pd.read_csv("combined_data_encoded.csv")
# -------------------------------

# ---- Recommender Logic ----

def calculate_per(data):
    data["PER"] = (
        (data["PTS"] + data["AST"] * 1.5 + data["TRB"] * 1.2 + data["STL"] * 1.5 + data["BLK"] * 1.5) /
        (data["MP"] + 1)
    ) * 15
    return data

def preprocess_data(data):
    data = calculate_per(data)
    stat_weights = {
        "G": {"AST": 1.5, "3P": 1.2, "STL": 1.2, "TRB": 0.8, "BLK": 0.8},
        "F": {"TRB": 1.3, "BLK": 1.1, "AST": 1.1, "3P": 1.0},
        "C": {"TRB": 1.5, "BLK": 1.3, "AST": 0.7, "3P": 0.5},
    }
    def apply_weights(row):
        for pos, weights in stat_weights.items():
            if row.get(f"Pos_{pos}", 0) == 1:
                for stat, weight in weights.items():
                    if stat in row:
                        row[stat] *= weight
        return row
    data = data.copy().apply(apply_weights, axis=1)
    feature_cols = ['PTS', 'AST', 'TRB', 'STL', 'BLK', '3P', 'PER']
    scaler = StandardScaler()
    data[feature_cols] = scaler.fit_transform(data[feature_cols])
    return data, feature_cols

def cluster_players(data, feature_cols, n_clusters=5):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    data["Playstyle_Cluster"] = kmeans.fit_predict(data[feature_cols])
    return data

def build_similarity_matrix(data, feature_cols):
    similarity_matrix = cosine_similarity(data[feature_cols])
    return pd.DataFrame(similarity_matrix, index=data["Player"], columns=data["Player"])

def recommend_wnba_players(nba_player, combined_data_encoded, nba_to_wnba_similarity, top_n=5):
    if nba_player not in combined_data_encoded["Player"].values:
        return None
    nba_city_encoded = combined_data_encoded[combined_data_encoded["Player"] == nba_player][[col for col in combined_data_encoded.columns if col.startswith("City_")]].values[0]
    wnba_candidates = combined_data_encoded[combined_data_encoded["League_WNBA"] == 1].copy()
    wnba_candidates["City_Match"] = wnba_candidates[[col for col in combined_data_encoded.columns if col.startswith("City_")]].apply(
        lambda x: int(any(x == nba_city_encoded)), axis=1)
    if "Starter" in wnba_candidates.columns:
        wnba_candidates = wnba_candidates.sort_values(by=["Starter", "City_Match", "PER"], ascending=[False, False, False])
    else:
        wnba_candidates = wnba_candidates.sort_values(by=["City_Match", "PER"], ascending=[False, False])
    similar_wnba_players = nba_to_wnba_similarity.loc[nba_player, wnba_candidates["Player"]].sort_values(ascending=False).head(top_n)
    team_columns = [col for col in combined_data_encoded.columns if col.startswith("Team Name_")]
    city_columns = [col for col in combined_data_encoded.columns if col.startswith("City_")]
    final_recs = combined_data_encoded[combined_data_encoded["Player"].isin(similar_wnba_players.index)].copy()
    final_recs["Team"] = final_recs[team_columns].idxmax(axis=1).str.replace("Team Name_", "")
    final_recs["City"] = final_recs[city_columns].idxmax(axis=1).str.replace("City_", "")
    return final_recs[["Player", "Team", "City"]].drop_duplicates()

# ---- Run Preprocessing Once ----
combined_data_encoded = calculate_per(combined_data_encoded)
combined_data_encoded, feature_cols = preprocess_data(combined_data_encoded)
combined_data_encoded = cluster_players(combined_data_encoded, feature_cols)
nba_to_wnba_similarity = build_similarity_matrix(combined_data_encoded, feature_cols)

# ---- Streamlit UI ----
st.set_page_config(page_title="WNBA Recommender", layout="wide")

st.title("🏀 WNBA Player Recommender")
st.markdown("Get WNBA player recommendations based on your favorite NBA player.")

nba_players = sorted(combined_data_encoded[combined_data_encoded["League_WNBA"] == 0]["Player"].unique())
selected_nba_player = st.selectbox("Select an NBA Player", nba_players)

if st.button("Recommend WNBA Players"):
    recommendations = recommend_wnba_players(selected_nba_player, combined_data_encoded, nba_to_wnba_similarity, top_n=5)
    if recommendations is None:
        st.warning("Player not found in dataset.")
    else:
        st.subheader("Recommended WNBA Players:")
        st.table(recommendations)
