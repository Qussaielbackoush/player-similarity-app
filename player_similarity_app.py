import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go

# Load and clean data
df = pd.read_csv("Player_Data_Final.xls.csv")

# Define metrics by position
metrics_by_position = {
    "FW": ['Match Played', 'Goals', 'Assists', 'Succ. dribbles', 'Total shots', 'Goal conversion %'],
    "MF": ['Match Played', 'Goals', 'Assists', 'Ground duels won', 'Accurate passes %', 'Accurate final third passes', 'Acc. long balls'],
    "DF": ['Match Played', 'Ground duels won', 'Aerial duels won', 'Accurate passes %', 'Accurate final third passes', 'Acc. long balls', 'Clearances']
}

meta_columns = ['Name', 'Team', 'Pos', 'Age','Nationality', 'League']

# Drop missing values initially
df_clean = df.dropna(subset=meta_columns).copy()
df_clean = df_clean.reset_index(drop=True)

# Sidebar filters (apply only on results)
st.sidebar.title("🔎 Result Filters")
team_filter = st.sidebar.selectbox("Filter results by Team", ["All"] + sorted(df_clean['Team'].unique()))
pos_filter = st.sidebar.selectbox("Filter results by Position", ["All"] + sorted(df_clean['Pos'].unique()))
age_range = st.sidebar.slider("Filter results by Age", int(df_clean['Age'].min()), int(df_clean['Age'].max()), (18, 35))

# --- Main selection ---
st.title("⚽ Player Similarity Finder (2024–2025)")

col1, col2, col3, col4 = st.columns(4)
with col1:
    selected_player = st.selectbox("Choose a player (required):", df_clean["Name"])
with col2:
    selected_team = st.selectbox("Optional: Choose a team", ["Any"] + sorted(df_clean['Team'].unique()))
with col3:
    selected_league = st.selectbox("Optional: Choose a league", ["Any"] + sorted(df_clean['League'].unique()))
with col4:
    selected_position = st.selectbox("Optional: Choose a position", ["Any"] + sorted(df_clean['Pos'].unique()))

# --- Apply optional filters to restrict player pool ---
player_pool = df_clean.copy()
if selected_team != "Any":
    player_pool = player_pool[player_pool["Team"] == selected_team]
if selected_league != "Any":
    player_pool = player_pool[player_pool["League"] == selected_league]
if selected_position != "Any":
    player_pool = player_pool[player_pool["Pos"] == selected_position]

# If selected player not in filtered pool, re-add him
if selected_player not in player_pool["Name"].values:
    player_pool = pd.concat([player_pool, df_clean[df_clean["Name"] == selected_player]])

if selected_player:
    player_row = df_clean[df_clean['Name'] == selected_player].iloc[0]
    player_pos = player_row["Pos"]

    # Choose metrics by position
    if any(p in player_pos for p in ["FW", "ST", "ATT"]):
        comparison_columns = metrics_by_position["FW"]
    elif any(p in player_pos for p in ["MF", "MID"]):
        comparison_columns = metrics_by_position["MF"]
    elif any(p in player_pos for p in ["DF", "CB", "LB", "RB"]):
        comparison_columns = metrics_by_position["DF"]
    else:
        comparison_columns = metrics_by_position["FW"]

    # Drop missing values in metrics
    df_filtered_metrics = player_pool.dropna(subset=comparison_columns).reset_index(drop=True)

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_filtered_metrics[comparison_columns])

    # Similarity matrix
    similarity_matrix = cosine_similarity(X_scaled)

    # Number of results
    num_results = st.sidebar.slider("Number of similar players to show", 3, 30, 10)

    player_idx = df_filtered_metrics[df_filtered_metrics['Name'] == selected_player].index[0]
    selected_vector = similarity_matrix[player_idx]

    df_filtered_metrics["Similarity"] = selected_vector * 100
    similar_players = df_filtered_metrics.sort_values("Similarity", ascending=False)
    similar_players = similar_players[similar_players["Name"] != selected_player]

    # --- Apply result filters ---
    if team_filter != "All":
        similar_players = similar_players[similar_players['Team'] == team_filter]
    if pos_filter != "All":
        similar_players = similar_players[similar_players['Pos'] == pos_filter]
    similar_players = similar_players[
        (similar_players['Age'] >= age_range[0]) & (similar_players['Age'] <= age_range[1])
    ]

    # --- Player profile ---
    st.markdown(f"## 🧾 Player Profile: {selected_player}")
    st.write(f"**Age**: {player_row['Age']} | **Team**: {player_row['Team']} | **Nationality**: {player_row['Nationality']} | **Position**: {player_row['Pos']}")

    # Player stats
    st.markdown("### 📊 Performance Stats")
    st.dataframe(df_filtered_metrics.loc[[player_idx], comparison_columns].T.rename(columns={player_idx: selected_player}), use_container_width=True)

    # Similar players
    st.markdown(f"### 🧬 Similar Players to {selected_player}")
    st.dataframe(similar_players[["Name", "Team", "Pos", "Age", "Similarity"] + comparison_columns].head(num_results), use_container_width=True)
# Radar chart
if not similar_players.empty:
    st.markdown("---")
    st.markdown("## 📊 Radar Comparison")

    top_similar = similar_players.iloc[0]
    similar_player_name = top_similar["Name"]

    # Use a version of filtered_df that has the metrics safely
    df_metrics = filtered_df.dropna(subset=comparison_columns).reset_index(drop=True)

    # Find index again inside df_metrics
    player_idx_metrics = df_metrics[df_metrics["Name"] == selected_player].index[0]

    radar_df = pd.DataFrame({
        "Metric": comparison_columns,
        selected_player: [df_metrics.loc[player_idx_metrics, col] for col in comparison_columns],
        similar_player_name: [top_similar[col] for col in comparison_columns]
    })

    radar_df.set_index("Metric", inplace=True)

    # Normalize values into [0,100]
    radar_df_norm = pd.DataFrame(index=radar_df.index)

    for metric in radar_df.index:
        min_val = df_metrics[metric].min()
        max_val = df_metrics[metric].max()

        for col in radar_df.columns:
            val = radar_df.loc[metric, col]
            if max_val > min_val:
                norm_val = (val - min_val) / (max_val - min_val) * 100
                radar_df_norm.loc[metric, col] = norm_val
            else:
                radar_df_norm.loc[metric, col] = 50  # fallback if all values are identical

    radar_df_norm.reset_index(inplace=True)

    # Create radar chart
    fig = go.Figure()

    # Orange for selected player
    fig.add_trace(go.Scatterpolar(
        r=radar_df_norm[selected_player],
        theta=radar_df_norm["Metric"],
        fill='toself',
        name=selected_player,
        line=dict(color="orange"),
        fillcolor="rgba(255,165,0,0.3)"
    ))

    # Blue for top similar player
    fig.add_trace(go.Scatterpolar(
        r=radar_df_norm[similar_player_name],
        theta=radar_df_norm["Metric"],
        fill='toself',
        name=similar_player_name,
        line=dict(color="blue"),
        fillcolor="rgba(0,0,255,0.3)"
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100]),
        ),
        showlegend=True,
        title=dict(text=f"{selected_player} vs {similar_player_name}", x=0.5)
    )

    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align: center; font-size:14px;'>"
    "© 2025 Developed by <b>Qussai Elbackoush</b> – All Rights Reserved"
    "</div>",
    unsafe_allow_html=True
)

