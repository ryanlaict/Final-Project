Project / Goals
The goal of this project was to develop a recommendation system that connects NBA fans to WNBA players and merchandise. By understanding a fan's favorite NBA player, team, or city, the system recommends similar WNBA players and customized merchandise suggestions. This project aims to increase engagement with the WNBA by leveraging existing NBA fanbases.

Process
Step 1: Data Collection and Preparation

Compiled datasets of NBA and WNBA players, including team affiliation, city, position, and performance metrics (like Player Efficiency Rating - PER).

One-hot encoded categorical features such as city and team names.

Added a list of the Top 25 WNBA players to filter and prioritize high-profile recommendations.

Step 2: Building the Recommender

Constructed a hybrid similarity matrix combining playstyle features, city matching, and performance metrics.

Designed a recommendation function that allows users to input an NBA player, team, or city to receive WNBA player recommendations.

Integrated filters to prioritize "starter" players, city matches, and Top 25 WNBA players when desired.

Step 3: Merchandise Recommendation

Created a second model to classify customers by engagement levels (based on purchase frequency, TV viewing hours, social media engagement, etc.).

Developed a rank-based merchandise recommendation function: highly engaged fans are offered a wider range of merchandise, while less engaged fans are suggested basic options like game tickets or t-shirts.

Results
The WNBA player recommender accurately returned highly relevant matches based on NBA fan input, especially when applying the Top 25 player filter.

The merchandise recommender provided customized promotional strategies based on customer interaction data.

Early testing showed that adding the Top 25 filter made the recommendations more appealing to casual fans unfamiliar with the full WNBA roster.

Challenges
Aligning NBA and WNBA player profiles was challenging because of differences in available data and league structures.

Ensuring enough WNBA player matches for every NBA input, especially when filtering for Top 25 players, sometimes required fine-tuning thresholds.

Balancing recommendation diversity (city match vs playstyle match) was tricky — too strict on either metric led to fewer or less interesting recommendations.

Future Goals
Expand player feature sets with advanced stats like usage rate, defensive rating, and shot charts for deeper player similarity.

Integrate collaborative filtering based on fan behavior (e.g., fans who like LeBron also follow Breanna Stewart).

Deploy a full interactive web app using Streamlit with dynamic visualizations and customer input forms.

Launch A/B testing campaigns to measure the real-world impact of targeted merchandise promotions.