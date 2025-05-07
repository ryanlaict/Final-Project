#WNBA recommendor with customer data integrated 

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load and preprocess customer data
def preprocess_customer_data(customer_data):
    """Preprocess customer dataset: one-hot encode categorical features and normalize numerical ones."""
    categorical_cols = ["favorite_nba_team", "favorite_players", "wnba_team_interest", "merch_purchased"]
    numerical_cols = ["purchase_frequency", "game_attendance", "tv_viewing_hours", "social_media_engagement", "sports_news_engagement"]
    
    # One-hot encode categorical columns
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_cats = encoder.fit_transform(customer_data[categorical_cols])
    encoded_cats_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_cols))
    
    # Normalize numerical columns
    scaler = StandardScaler()
    customer_data[numerical_cols] = scaler.fit_transform(customer_data[numerical_cols])
    
    # Combine processed features
    customer_data = pd.concat([customer_data, encoded_cats_df], axis=1).drop(columns=categorical_cols)
    
    return customer_data

def calculate_per(data):
    """Calculate Player Efficiency Rating (PER)"""
    data["PER"] = (
        (data["PTS"] + data["AST"] * 1.5 + data["TRB"] * 1.2 + data["STL"] * 1.5 + data["BLK"] * 1.5) /
        (data["MP"] + 1)  # Avoid division by zero
    ) * 15  # Scale PER to an average of 15
    return data

def preprocess_data(data):
    """Apply position-based weightings and normalize stats"""
    data = calculate_per(data)
    
    stat_weights = {
        "G": {"AST": 1.5, "3P": 1.2, "STL": 1.2, "TRB": 0.8, "BLK": 0.8},
        "F": {"TRB": 1.3, "BLK": 1.1, "AST": 1.1, "3P": 1.0},
        "C": {"TRB": 1.5, "BLK": 1.3, "AST": 0.7, "3P": 0.5},
    }

    def apply_weights(row):
        for pos, weights in stat_weights.items():
            if row[f"Pos_{pos}"] == 1:
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
    """Group players into clusters based on playstyle"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    data["Playstyle_Cluster"] = kmeans.fit_predict(data[feature_cols])
    return data

def build_similarity_matrix(data, feature_cols):
    """Compute similarity scores"""
    similarity_matrix = cosine_similarity(data[feature_cols])
    return pd.DataFrame(similarity_matrix, index=data["Player"], columns=data["Player"])

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

# Load customer data and integrate into recommendations
def recommend_for_customer(customer_email, customer_data):
    # Locate the customer by email
    customer = customer_data[customer_data["email"] == customer_email]
    if customer.empty:
        return "Customer not found."

    # Extract one-hot encoded columns related to favorite NBA team
    favorite_team_columns = [col for col in customer_data.columns if col.startswith("favorite_nba_team_")]
    favorite_team = customer[favorite_team_columns].idxmax(axis=1).str[len("favorite_nba_team_"):].values[0]

    # Extract one-hot encoded columns related to favorite players
    favorite_players_columns = [col for col in customer_data.columns if col.startswith("favorite_players_")]
    favorite_players = customer[favorite_players_columns].idxmax(axis=1).str[len("favorite_players_"):].values[0]

    # Determine WNBA team suggestion
    wnba_team_columns = [col for col in customer_data.columns if col.startswith("wnba_team_interest_")]
    wnba_team_suggestion = customer[wnba_team_columns].idxmax(axis=1).str[len("wnba_team_interest_"):].values[0]

    # Get player recommendations
    if favorite_players != "unknown":
        player_recommendations = recommend_wnba_players(favorite_players, top_n=3)
    else:
        player_recommendations = "No player preference"

    return {
        "Suggested WNBA Team": wnba_team_suggestion,
        "Recommended WNBA Players": player_recommendations
    }

combined_data_encoded = pd.read_csv("combined_data_encoded.csv")
# Preprocess data
combined_data_encoded = calculate_per(combined_data_encoded)
combined_data_encoded, feature_cols = preprocess_data(combined_data_encoded)
combined_data_encoded = cluster_players(combined_data_encoded, feature_cols)
nba_to_wnba_similarity = build_similarity_matrix(combined_data_encoded, feature_cols)