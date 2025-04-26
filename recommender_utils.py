# recommender_utils.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

def calculate_per(data):
    data["PER"] = (
        (data["PTS"] + data["AST"] * 1.5 + data["TRB"] * 1.2 + data["STL"] * 1.5 + data["BLK"] * 1.5) /
        (data["MP"] + 1)
    ) * 15
    return data

def preprocess_data(data):
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
    return pd.DataFrame(
        cosine_similarity(data[feature_cols]),
        index=data["Player"],
        columns=data["Player"]
    )

def recommend_wnba_players(nba_player, combined_data_encoded, similarity_matrix, top_n=5):
    if nba_player not in combined_data_encoded["Player"].values:
        return "Player not found in dataset."

    nba_cluster = combined_data_encoded.loc[combined_data_encoded["Player"] == nba_player, "Playstyle_Cluster"].values[0]
    nba_city_encoded = combined_data_encoded[combined_data_encoded["Player"] == nba_player][[col for col in combined_data_encoded.columns if col.startswith("City_")]].values[0]

    wnba_candidates = combined_data_encoded[combined_data_encoded["League_WNBA"] == 1].copy()
    wnba_candidates['City_Match'] = wnba_candidates[[col for col in combined_data_encoded.columns if col.startswith("City_")]].apply(
        lambda x: int(any(x == nba_city_encoded)), axis=1)

    if "Starter" in wnba_candidates.columns:
        wnba_candidates = wnba_candidates.sort_values(by=["Starter", "City_Match", "PER"], ascending=[False, False, False])
    else:
        wnba_candidates = wnba_candidates.sort_values(by=["City_Match", "PER"], ascending=[False, False])

    similar_wnba_players = similarity_matrix.loc[nba_player, wnba_candidates["Player"]].sort_values(ascending=False).head(top_n)

    team_columns = [col for col in combined_data_encoded.columns if col.startswith("Team Name_")]
    city_columns = [col for col in combined_data_encoded.columns if col.startswith("City_")]

    wnba_candidates = combined_data_encoded[combined_data_encoded["Player"].isin(similar_wnba_players.index)].copy()
    wnba_candidates['Team'] = wnba_candidates[team_columns].idxmax(axis=1).str.replace("Team Name_", "")
    wnba_candidates['City'] = wnba_candidates[city_columns].idxmax(axis=1).str.replace("City_", "")

    return wnba_candidates[["Player", "Team", "City"]].drop_duplicates()

def recommend_wnba_players_by_input(user_input, data, sim_matrix, mode, top_n=5):
    if mode == "NBA Player":
        nba_subset = data[(data["League_NBA"] == 1) & (data["Player"].str.lower() == user_input.lower())]
    elif mode == "NBA Team":
        team_cols = [col for col in data.columns if col.startswith("Team Name_")]
        nba_subset = data[(data["League_NBA"] == 1) & (
            data[team_cols].idxmax(axis=1).str.replace("Team Name_", "").str.lower() == user_input.lower()
        )]
    elif mode == "City":
        city_cols = [col for col in data.columns if col.startswith("City_")]
        nba_subset = data[(data["League_NBA"] == 1) & (
            data[city_cols].idxmax(axis=1).str.replace("City_", "").str.lower() == user_input.lower()
        )]
    else:
        return "Invalid search mode."

    if nba_subset.empty:
        return f"No NBA records found for {mode.lower()} '{user_input}'. Please check spelling."

    nba_city_vec = nba_subset[[col for col in data.columns if col.startswith("City_")]].values[0]
    nba_cluster = nba_subset["Playstyle_Cluster"].values[0]
    nba_player = nba_subset["Player"].values[0]

    wnba_candidates = data[data["League_WNBA"] == 1].copy()
    wnba_candidates["City_Match"] = wnba_candidates[[col for col in data.columns if col.startswith("City_")]].apply(
        lambda x: int(any(x == nba_city_vec)), axis=1
    )

    if "Starter" in wnba_candidates.columns:
        wnba_candidates = wnba_candidates.sort_values(by=["Starter", "City_Match", "PER"], ascending=[False, False, False])
    else:
        wnba_candidates = wnba_candidates.sort_values(by=["City_Match", "PER"], ascending=[False, False])

    top_players = sim_matrix.loc[nba_player, wnba_candidates["Player"]].sort_values(ascending=False).head(top_n)

    team_cols = [col for col in data.columns if col.startswith("Team Name_")]
    city_cols = [col for col in data.columns if col.startswith("City_")]

    recs = data[data["Player"].isin(top_players.index)].copy()
    recs["Team"] = recs[team_cols].idxmax(axis=1).str.replace("Team Name_", "")
    recs["City"] = recs[city_cols].idxmax(axis=1).str.replace("City_", "")

    return recs[["Player", "Team", "City"]].drop_duplicates()
