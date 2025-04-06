import streamlit as st # type: ignore
import pandas as pd
import plotly.express as px # type: ignore
import plotly.graph_objects as go  # type: ignore
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage

# Set page configuration
st.set_page_config(
    page_title="Sabrina Carpenter Concert Analysis",
    page_icon="🎵",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size:2.5rem;
        font-weight:bold;
        color:#8349c9;
        text-align:center;
        margin-bottom:2rem;
    }
    .section-header {
        font-size:1.8rem;
        font-weight:bold;
        color:#8349c9;
        margin-top:1rem;
        margin-bottom:1rem;
    }
    .bio-card {
        background-color:#f8f9fa;
        padding:1.5rem;
        border-radius:10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stat-highlight {
        color:#8349c9;
        font-weight:bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='main-header'>Spotify Live Presents: A Night of Sweet Secrets</div>", unsafe_allow_html=True)
st.markdown("### Location: Capital One Arena, Washington DC\n")
st.markdown("""Early October 2025: Bridging the gap between Gracie Abrams' 'The Secret of Us' tour finale and Sabrina Carpenter's 'Short n’ Sweet' tour kickoff!""", unsafe_allow_html=True)

# Function to load data
@st.cache_data
def load_data():
    # Songs data
    sabrina_songs_data = {
        'song_title': ['Espresso', 'Please Please Please', 'Feather', 'Nonsense', 'Skin', 
                      'Because I Liked A Boy', 'Emails I Can\'t Send', 'Fast Times', 'Read Your Mind', 'Vicious'],
        'streams': [1200000000, 750000000, 650000000, 900000000, 450000000, 
                   350000000, 550000000, 480000000, 320000000, 380000000],
        'chart_position': [1, 3, 5, 2, 12, 18, 8, 15, 21, 16],
        'release_year': [2023, 2023, 2023, 2022, 2021, 2022, 2022, 2022, 2023, 2022],
        'album': ['Short n\' Sweet', 'Emails I Can\'t Send', 'Emails I Can\'t Send', 'Emails I Can\'t Send', 'Singular: Act I', 
                 'Emails I Can\'t Send', 'Emails I Can\'t Send', 'Emails I Can\'t Send', 'Short n\' Sweet', 'Emails I Can\'t Send'],
        'grammy_nominations': [1, 1, 0, 1, 0, 0, 1, 0, 0, 0]
    }

    gracie_songs_data = {
        'song_title': ['I miss you, I\'m sorry', 'Risk', 'Block me out', 'Where do we go now?', 
                    'Difficult', 'Feels Like', 'Close to you', 'Amelie', 'That Much', 'Right Now'],
        'streams': [220000000, 185000000, 75000000, 130000000, 105000000, 
                90000000, 65000000, 180000000, 48000000, 95000000],
        'chart_position': [35, 25, 62, 45, 50, 
                        55, 70, 30, 85, 52],
        'release_year': [2020, 2021, 2023, 2022, 2023, 
                        2020, 2021, 2023, 2020, 2022],
        'album': ['minor', 'This Is What It Feels Like', 'Good Riddance', 'This Is What It Feels Like', 'Good Riddance',
                'minor', 'This Is What It Feels Like', 'Good Riddance', 'minor', 'This Is What It Feels Like'],
        'grammy_nominations': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    }

    # Concert demographics data
    age_data = {
        'age_group': ['13-17', '18-24', '25-34', '35-44', '45-54', '55+'],
        'percentage': [10, 6.63, 22.36, 16.45, 10.71, 12.72]
    }
    
    # Artist similarity data
    artists_data = {
        'artist': ['Sabrina Carpenter', 'Gracie Abrams'],
        'genre': ['Pop', 'Pop'],
        'tempo': [112.96, 115.94],
        'danceability': [0.82, 0.55],
        'valence': [0.721, 0.338],
        'energy': [0.907, 0.808],
        'collaborations': [11, 2],
        'avg_stream_count': [510714286, 91935484],
        'social_media_followers': [46100000, 5000000]
    }
    
    return pd.DataFrame(sabrina_songs_data), pd.DataFrame(gracie_songs_data), pd.DataFrame(age_data), pd.DataFrame(artists_data)

# Load data
sabrina_songs_df, gracie_songs_df, age_df, artists_df = load_data()

# Create tabs for different sections
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Audience Demographics",
    "🎤 Sabrina Carpenter Selection Process",
    "🎶 Gracie Abrams Selection Process", 
    "👩‍🎤 Artist Biography", 
    "🎵 Hit Song Performance"
])

# Tab 1: Audience Demographics Analysis
with tab1:
    st.markdown("<div class='section-header'>Audience Demographic Analysis</div>", unsafe_allow_html=True)
    
    # Create a dropdown to select which visualization to display
    viz_option = st.selectbox(
        "Select Visualization:",
        ["Age Distribution", "DC Area Universities", "Music Genre Preferences"],
        index=0
    )
    
    # Display Age Distribution visualization when selected
    if viz_option == "Age Distribution":
        st.subheader("Concert Audience Age Distribution")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            chart_type = st.radio("Select Chart Type:", ["Bar Chart", "Line Chart"], horizontal=True)
            
            if chart_type == "Bar Chart":
                fig = px.bar(
                    age_df,
                    x="age_group",
                    y="percentage",
                    color="percentage",
                    color_continuous_scale="Agsunset",
                    labels={"age_group": "Age Group", "percentage": "Percentage (%)"}
                )
            else:
                fig = px.line(
                    age_df,
                    x="age_group",
                    y="percentage",
                    markers=True,
                    line_shape="spline",
                    labels={"age_group": "Age Group", "percentage": "Percentage (%)"}
                )
                
            fig.update_layout(
                height=500,
                xaxis_title="Age Group",
                yaxis_title="Percentage (%)",
                yaxis_range=[0, max(age_df["percentage"]) * 1.1]
            )
            
            # Add annotations highlighting Gen Z and Millennials
            fig.add_annotation(
                x="13-17",
                y=age_df[age_df["age_group"] == "13-17"]["percentage"].values[0] + 2,
                text="Gen Z",
                showarrow=True,
                arrowhead=1,
                font=dict(size=14, color="#FF78C4")
            )
            
            fig.add_annotation(
                x="18-24",
                y=age_df[age_df["age_group"] == "18-24"]["percentage"].values[0] + 2,
                text="Gen Z",
                showarrow=True,
                arrowhead=1,
                font=dict(size=14, color="#FF78C4")
            )
            
            fig.add_annotation(
                x="25-34",
                y=age_df[age_df["age_group"] == "25-34"]["percentage"].values[0] + 2,
                text="Millennials",
                showarrow=True,
                arrowhead=1,
                font=dict(size=14, color="#FF78C4")
            )
            
            fig.add_annotation(
                x="35-44",
                y=age_df[age_df["age_group"] == "35-44"]["percentage"].values[0] + 2,
                text="Millennials",
                showarrow=True,
                arrowhead=1,
                font=dict(size=14, color="#FF78C4")
            )

            fig.add_annotation(
                x="45-54",
                y=age_df[age_df["age_group"] == "45-54"]["percentage"].values[0] + 2,
                text="Gen X",
                showarrow=True,
                arrowhead=1,
                font=dict(size=14, color="#FF78C4")
            )

            fig.add_annotation(
                x="55+",
                y=age_df[age_df["age_group"] == "55+"]["percentage"].values[0] + 2,
                text="Boomers II",
                showarrow=True,
                arrowhead=1,
                font=dict(size=14, color="#FF78C4")
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("""
            <div class="bio-card">
                <h3>Youth Audience Insights</h3>
                <p>
                    The data shows that <span class="stat-highlight">55.44%</span> of concert attendees 
                    are between <span class="stat-highlight">13-44</span> years old (Gen Z and Millennials).
                </p>
                <p>
                    This demographic alignment makes Sabrina Carpenter an ideal headliner for a 
                    youth-focused concert at Capital One Arena, as her primary fanbase falls within this age range.
                </p>
                <p>
                    Additional factors supporting this choice:
                </p>
                <ul>
                    <li>Strong social media presence resonating with younger audiences</li>
                    <li>Recent viral TikTok trends featuring her music</li>
                    <li>Appeal across both teen and young adult demographics</li>
                </ul>
            </div> """, unsafe_allow_html=True)
    
    # Display DC Area Universities visualization when selected
    elif viz_option == "DC Area Universities":
        st.subheader("DC Area Universities and Colleges")
        
        # Sample data for DC area universities
        dc_universities = pd.DataFrame({
            'University': ['Georgetown University', 'George Washington University', 'American University',
                          'Howard University', 'Catholic University', 'Gallaudet University', 
                          'University of the District of Columbia', 'Trinity Washington University',
                          'George Mason University', 'University of Maryland College Park'],
            'Students': [19000, 26500, 14000, 9000, 5500, 1800, 4200, 2000, 38000, 41200],
            'lat': [38.9076, 38.9009, 38.9365, 38.9227, 38.9333, 38.9031, 38.9432, 38.9267, 38.8315, 38.9869],
            'lon': [-77.0723, -77.0491, -77.0891, -77.0194, -76.9994, -76.9986, -77.0162, -77.0019, -77.3103, -76.9426],
            'Gen_Z_Percentage': [75, 70, 72, 78, 71, 69, 65, 68, 73, 74]
        })
        
        # Display options for university visualization
        uni_viz_type = st.radio(
            "Select University View:",
            ["Map View", "Bar Chart View"],
            horizontal=True
        )
        
        if uni_viz_type == "Map View":
            # Create a map visualization with bubble size representing student population
            fig_map = px.scatter_mapbox(
                dc_universities,
                lat='lat',
                lon='lon',
                size='Students',
                color='Gen_Z_Percentage',
                color_continuous_scale='Viridis',
                hover_name='University',
                hover_data={'Students': True, 'Gen_Z_Percentage': True, 'lat': False, 'lon': False},
                zoom=10,
                mapbox_style='carto-positron',
                title='DC Area Universities by Student Population and Gen Z Percentage',
                size_max=25,
                opacity=0.8,
                labels={'Gen_Z_Percentage': 'Gen Z %', 'Students': 'Student Count'}
            )
            
            fig_map.update_layout(
                height=600,
                margin={"r":0,"t":50,"l":0,"b":0}
            )
            
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            # Create bar chart of university population
            fig_uni_bar = px.bar(
                dc_universities.sort_values('Students', ascending=False),
                x='University',
                y='Students',
                color='Gen_Z_Percentage',
                color_continuous_scale='Viridis',
                labels={'Students': 'Number of Students', 'University': 'University/College'},
                title='DC Area Universities by Student Population',
                height=600,
                text_auto=True
            )
            
            fig_uni_bar.update_layout(
                xaxis={'categoryorder':'total descending'},
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig_uni_bar, use_container_width=True)
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Total student count
            total_students = dc_universities['Students'].sum()
            gen_z_students = int(sum(dc_universities['Students'] * dc_universities['Gen_Z_Percentage'] / 100))
            
            # Create stats visualization
            fig_stats = go.Figure()
            
            fig_stats.add_trace(go.Indicator(
                mode = "number",
                value = total_students,
                title = {"text": "Total Students in DC Area"},
                domain = {'row': 0, 'column': 0}
            ))
            
            fig_stats.add_trace(go.Indicator(
                mode = "number+delta",
                value = gen_z_students,
                title = {"text": "Gen Z Students"},
                delta = {'reference': total_students, 'relative': True, 'valueformat': '.1%'},
                domain = {'row': 0, 'column': 1}
            ))
            
            fig_stats.update_layout(
                grid = {'rows': 1, 'columns': 2, 'pattern': "independent"},
                height = 200
            )
            
            st.plotly_chart(fig_stats, use_container_width=True)
        
        with col2:
            st.markdown("""
            <div class="info-card">
                <p>The Washington DC metro area is home to over <span class="stat-highlight">160,000</span> college students
                across major universities, with <span class="stat-highlight">~70%</span> falling within the Gen Z demographic.</p>
                <p>Capital One Arena's central location is within <span class="stat-highlight">5 miles</span> of 8 major universities,
                making it highly accessible to the student population that forms Sabrina Carpenter's core fanbase.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Display Music Genre Preferences visualization when selected
    elif viz_option == "Music Genre Preferences":
        st.subheader("Music Genre Preferences by Age Group")
        
        # Sample data for music preferences
        music_prefs = pd.DataFrame({
            'Age_Group': ['13-17', '18-24', '25-34', '35-44', '45-54', '55+'],
            'Pop': [85, 78, 65, 52, 45, 30],
            'Rock': [45, 50, 60, 70, 75, 65],
            'Hip_Hop': [80, 75, 60, 45, 30, 15],
            'Country': [30, 35, 40, 45, 55, 60],
            'Electronic': [65, 70, 55, 35, 20, 10],
            'Classical': [25, 30, 35, 40, 50, 65]
        })
        
        # Add tabs for different music preference visualizations
        music_tabs = st.tabs(["Gen Z Focus", "All Age Groups", "Trend Analysis"])
        
        with music_tabs[0]:
            # Create a radar chart for Gen Z music preferences
            gen_z_data = pd.melt(
                music_prefs[music_prefs['Age_Group'].isin(['13-17', '18-24'])],
                id_vars=['Age_Group'],
                var_name='Genre',
                value_name='Preference_Score'
            )
            
            fig_radar = px.line_polar(
                gen_z_data,
                r='Preference_Score',
                theta='Genre',
                color='Age_Group',
                line_close=True,
                labels={'Preference_Score': 'Popularity Score', 'Genre': 'Music Genre'},
                title='Gen Z Music Preferences (Age 13-24)',
                color_discrete_sequence=["#FF78C4", "#9D76C1"]
            )
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                height=600,
                showlegend=True
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
        
        with music_tabs[1]:
            # Create a grouped bar chart comparing all genres across age groups
            genre_comparison = px.bar(
                music_prefs,
                x='Age_Group',
                y=['Pop', 'Rock', 'Hip_Hop', 'Country', 'Electronic', 'Classical'],
                title='Music Genre Preferences by Age Group',
                barmode='group',
                color_discrete_sequence=["#FF78C4", "#9D76C1", "#7A89C2", "#6C9BCF", "#4ADEDE", "#797EF6"]
            )
            
            genre_comparison.update_layout(
                height=600,
                xaxis_title="Age Group",
                yaxis_title="Popularity Score (%)",
                legend_title="Music Genre"
            )
            
            # Add a highlight annotation for Sabrina's genre
            genre_comparison.add_annotation(
                x="18-24",
                y=music_prefs[music_prefs["Age_Group"] == "18-24"]["Pop"].values[0] + 5,
                text="Sabrina Carpenter's Genre",
                showarrow=True,
                arrowhead=1,
                font=dict(size=12, color="#FF78C4")
            )
            
            st.plotly_chart(genre_comparison, use_container_width=True)
        
        with music_tabs[2]:
            # Create line chart showing trends across age groups
            music_long = pd.melt(
                music_prefs, 
                id_vars=['Age_Group'], 
                var_name='Genre', 
                value_name='Preference_Score'
            )
            
            trend_chart = px.line(
                music_long,
                x='Age_Group',
                y='Preference_Score',
                color='Genre',
                markers=True,
                title='Music Preference Trends Across Age Groups',
                color_discrete_sequence=["#FF78C4", "#9D76C1", "#7A89C2", "#6C9BCF", "#4ADEDE", "#797EF6"]
            )
            
            trend_chart.update_layout(
                height=600,
                xaxis_title="Age Group",
                yaxis_title="Popularity Score (%)",
                legend_title="Music Genre"
            )
            
            # Add highlight area for Gen Z
            trend_chart.add_vrect(
                x0="13-17", 
                x1="18-24",
                fillcolor="#FF78C4", 
                opacity=0.15,
                layer="below",
                line_width=0,
                annotation_text="Gen Z Focus",
                annotation_position="top left"
            )
            
            st.plotly_chart(trend_chart, use_container_width=True)
        
        st.markdown("""
        <div class="insight-card">
            <h4>Pop Music Dominance in Target Demographics</h4>
            <p>Pop music shows the highest preference among the 13-24 age group, with
            <span class="stat-highlight">78-85%</span> popularity compared to other genres.</p>
            <p>Sabrina Carpenter's pop music focus aligns perfectly with the preferences of
            both high school and college-aged fans in the DC metro area, supporting the strategic
            decision to feature her as the headliner.</p>
        </div>
        """, unsafe_allow_html=True)

# Tab 2: Artist Selection Process
with tab2:
    st.markdown("<div class='section-header'>Artist Selection Process</div>", unsafe_allow_html=True)
    
    # Load artist data
    @st.cache_data
    def load_artist_data():
        df = pd.read_csv('Python_Files/artist_scores.csv')
        # Ensure proper data types
        df['frequency'] = df['frequency'].astype(int)
        df['ranking_score'] = df['ranking_score'].astype(float)
        df['rank_sum'] = df['rank_sum'].astype(float)
        return df
    
    artist_df = load_artist_data()
    
    # Create a dropdown to select which visualization to display
    artist_viz_option = st.selectbox(
        "Select Artist Analysis View:",
        ["Ranking Overview", "Artist Comparison", "Selection Criteria", "Final Decision Matrix"],
        index=0
    )
    
    # Display the Ranking Overview visualization
    if artist_viz_option == "Ranking Overview":
        st.subheader("Top 20 Artists by Ranking Score")
        
        # Filter to top 20 artists
        top_artists = artist_df.sort_values('ranking_score', ascending=False).head(20).copy()
        
        # Create columns for explanation and chart
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown("""
            <div class="bio-card">
                <h3>Ranking Methodology</h3>
                <p>
                    Artists were scored based on two primary factors:
                </p>
                <ul>
                    <li><span class="stat-highlight">Frequency</span>: Number of times the artist appeared in charts</li>
                    <li><span class="stat-highlight">Rank Sum</span>: Weighted sum of their chart positions</li>
                </ul>
                <p>
                    The combined <span class="stat-highlight">Ranking Score</span> represents the artist's overall popularity
                    and chart performance.
                </p>
                <p>
                    Sabrina Carpenter achieved the highest score with <span class="stat-highlight">814.86</span>, 
                    significantly ahead of other artists.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("")
            # Add a key metrics indicator
            st.metric(
                label="Sabrina's Score Lead",
                value=f"{(top_artists.iloc[0]['ranking_score'] - top_artists.iloc[1]['ranking_score']):.2f} points",
                delta=f"{((top_artists.iloc[0]['ranking_score'] / top_artists.iloc[1]['ranking_score']) - 1) * 100:.1f}%"
            )
            
        with col2:
            # Create a horizontal bar chart of the top artists
            fig = px.bar(
                top_artists,
                y='artists',
                x='ranking_score',
                color='ranking_score',
                color_continuous_scale='Agsunset',
                text='ranking_score',
                labels={'artists': 'Artist', 'ranking_score': 'Ranking Score'},
                height=600
            )
            
            fig.update_layout(
                yaxis={'categoryorder':'total ascending'},
                xaxis_title="Ranking Score",
                yaxis_title="Artist",
                coloraxis_showscale=False
            )
            
            # Format text to display values with 1 decimal place
            fig.update_traces(texttemplate='%{x:.1f}', textposition='outside')
            
            st.plotly_chart(fig, use_container_width=True)
            
    # Display the Artist Comparison visualization
    elif artist_viz_option == "Artist Comparison":
        st.subheader("Comparative Analysis of Top 10 Artists")
        
        # Filter to top 10 artists for comparison
        top10_artists = artist_df.sort_values('ranking_score', ascending=False).head(10).copy()
        
        # Create tabs for different comparison views
        comparison_tabs = st.tabs(["Score Breakdown", "Frequency vs. Rank", "Normalized Metrics"])
        
        with comparison_tabs[0]:
            # Create a grouped bar chart comparing rank_sum and frequency for top 10 artists
            fig_breakdown = px.bar(
                top10_artists,
                x='artists',
                y=['frequency', 'rank_sum'],
                title='Score Components for Top 10 Artists',
                barmode='group',
                color_discrete_sequence=["#FF78C4", "#9D76C1"],
                labels={'artists': 'Artist', 'value': 'Value', 'variable': 'Metric'}
            )
            
            fig_breakdown.update_layout(
                height=500,
                xaxis_title="Artist",
                yaxis_title="Value",
                xaxis_tickangle=-45,
                legend_title="Metric"
            )
            
            # Add annotation highlighting Sabrina's balanced performance
            fig_breakdown.add_annotation(
                x="Sabrina Carpenter",
                y=top10_artists[top10_artists['artists'] == 'Sabrina Carpenter']['frequency'].values[0] + 50,
                text="High Frequency + High Rank Sum",
                showarrow=True,
                arrowhead=1,
                font=dict(size=12, color="#FF78C4")
            )
            
            st.plotly_chart(fig_breakdown, use_container_width=True)
            
            # Add explanation text
            st.markdown("""
            <div class="insight-card">
                <p>Sabrina Carpenter stands out with both high frequency (1104 appearances) and the highest rank sum (175.24),
                indicating both widespread popularity and strong chart positions.</p>
                <p>While Taylor Swift has slightly higher frequency (1142), her lower rank sum results in a lower overall score.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with comparison_tabs[1]:
            # Create a scatter plot of frequency vs. rank_sum
            fig_scatter = px.scatter(
                top10_artists,
                x='frequency',
                y='rank_sum',
                color='ranking_score',
                size='ranking_score',
                text='artists',
                color_continuous_scale='Agsunset',
                labels={'frequency': 'Frequency (Appearances)', 'rank_sum': 'Rank Sum', 'ranking_score': 'Ranking Score'},
                title='Frequency vs. Rank Sum for Top 10 Artists'
            )
            
            fig_scatter.update_traces(
                textposition='top center',
                marker=dict(line=dict(width=1, color='DarkSlateGrey'))
            )
            
            fig_scatter.update_layout(
                height=600,
                xaxis_title="Frequency (Chart Appearances)",
                yaxis_title="Rank Sum (Position Weight)"
            )
            
            # Add quadrant labels
            fig_scatter.add_annotation(
                x=max(top10_artists['frequency']) * 0.85,
                y=max(top10_artists['rank_sum']) * 0.85,
                text="High Frequency + High Rank Sum<br>(Ideal Candidates)",
                showarrow=False,
                font=dict(size=10, color="#FF78C4")
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        with comparison_tabs[2]:
            # Create a normalized metrics radar chart
            radar_data = top10_artists[['artists', 'frequency_normalized', 'rank_sum_normalized']].copy()
            radar_data = pd.melt(
                radar_data,
                id_vars=['artists'],
                var_name='Metric',
                value_name='Normalized Score'
            )
            
            fig_radar = px.line_polar(
                radar_data,
                r='Normalized Score',
                theta='Metric',
                color='artists',
                line_close=True,
                labels={'Normalized Score': 'Score (0-1)', 'Metric': 'Metric'},
                range_r=[0, 1],
                title='Normalized Performance Metrics',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                height=600,
                showlegend=True
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
            
            st.markdown("""
            <div class="info-card">
                <p>This visualization compares artists using normalized metrics (0-1 scale):</p>
                <ul>
                    <li><span class="stat-highlight">Frequency Normalized</span>: Chart appearance frequency scaled to 0-1</li>
                    <li><span class="stat-highlight">Rank Sum Normalized</span>: Chart position weight scaled to 0-1</li>
                </ul>
                <p>Sabrina Carpenter shows the most balanced performance across both metrics,
                with near-perfect rank sum normalized (1.0) and very high frequency normalized (0.967).</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Display the Selection Criteria visualization
    elif artist_viz_option == "Selection Criteria":
        st.subheader("Artist Selection Criteria Analysis")
        
        # Create columns for criteria explanation and chart
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
            <div class="bio-card">
                <h3>Selection Factors</h3>
                <p>
                    Our headliner selection considered multiple factors:
                </p>
                <ol>
                    <li><span class="stat-highlight">Chart Performance</span>: Overall ranking score</li>
                    <li><span class="stat-highlight">Musical Genre</span>: Pop genre alignment with target demographics</li>
                    <li><span class="stat-highlight">Audience Match</span>: Appeal to Gen Z and Millennial audiences</li>
                    <li><span class="stat-highlight">Social Media Presence</span>: Engagement with younger demographics</li>
                    <li><span class="stat-highlight">Current Momentum</span>: Recent chart trends and viral potential</li>
                </ol>
                <p>
                    Sabrina Carpenter excelled across all criteria, making her the optimal choice for our audience.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Add a simple genre match visualization
            genre_match = pd.DataFrame({
                'Artist': ['Sabrina Carpenter', 'Taylor Swift', 'Chappell Roan', 'Zach Bryan', 'Morgan Wallen'],
                'Genre_Match': [95, 90, 85, 60, 55]
            })
            
            fig_genre = px.bar(
                genre_match,
                x='Artist',
                y='Genre_Match',
                color='Genre_Match',
                color_continuous_scale='Agsunset',
                labels={'Artist': 'Artist', 'Genre_Match': 'Pop Genre Match (%)'},
                title='Pop Genre Alignment with Target Demographics',
                height=300
            )
            
            fig_genre.update_layout(
                xaxis_tickangle=-45,
                coloraxis_showscale=False
            )
            
            st.plotly_chart(fig_genre, use_container_width=True)
        
        with col2:
            # Create a radar chart for multiple selection criteria
            selection_data = pd.DataFrame({
                'Artist': ['Sabrina Carpenter', 'Taylor Swift', 'Chappell Roan', 'Billie Eilish', 'Zach Bryan'],
                'Chart_Performance': [95, 90, 75, 70, 65],
                'Genre_Match': [95, 90, 85, 80, 60],
                'Age_Demo_Match': [90, 85, 90, 85, 70],
                'Social_Media': [95, 90, 85, 90, 70],
                'Current_Momentum': [98, 85, 90, 75, 80]
            })
            
            # Melt the dataframe for radar chart
            selection_melt = pd.melt(
                selection_data, 
                id_vars=['Artist'], 
                var_name='Criterion', 
                value_name='Score'
            )
            
            fig_criteria = px.line_polar(
                selection_melt,
                r='Score',
                theta='Criterion',
                color='Artist',
                line_close=True,
                labels={'Score': 'Score (0-100)', 'Criterion': 'Selection Criterion'},
                range_r=[0, 100],
                title='Multi-Factor Selection Criteria Comparison',
                color_discrete_sequence=["#FF78C4", "#9D76C1", "#7A89C2", "#6C9BCF", "#4ADEDE"]
            )
            
            fig_criteria.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                height=600,
                showlegend=True
            )
            
            st.plotly_chart(fig_criteria, use_container_width=True)
            
            # Add explanation
            st.markdown("""
            <div class="insight-card">
                <p>This radar chart visualizes how top artists compare across five key selection criteria.</p>
                <p><span class="stat-highlight">Sabrina Carpenter</span> demonstrates the most balanced and strong 
                performance across all criteria, particularly excelling in current momentum (98/100) and maintaining 
                scores above 90 in all categories.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Display the Final Decision Matrix visualization
    elif artist_viz_option == "Final Decision Matrix":
        st.subheader("Headliner Selection Decision Matrix")
        
        # Create a dataframe for the final decision matrix
        decision_matrix = pd.DataFrame({
            'Artist': ['Sabrina Carpenter', 'Taylor Swift', 'Chappell Roan', 'Billie Eilish', 'Zach Bryan'],
            'Ranking_Score': [814.86, 719.34, 565.93, 489.15, 547.12],
            'Genre_Match': [95, 90, 85, 80, 60],
            'Target_Demo_Match': [95, 85, 90, 90, 70],
            'Tour_Availability': [90, 60, 85, 70, 80],
            'Cost_Effectiveness': [85, 50, 90, 75, 80],
            'Social_Engagement': [95, 90, 85, 90, 75]
        })
        
        # Calculate weighted scores
        weights = {
            'Ranking_Score': 0.3,
            'Genre_Match': 0.2,
            'Target_Demo_Match': 0.2,
            'Tour_Availability': 0.1,
            'Cost_Effectiveness': 0.1,
            'Social_Engagement': 0.1
        }
        
        # Normalize ranking score to 0-100 scale for fair comparison
        max_score = decision_matrix['Ranking_Score'].max()
        decision_matrix['Ranking_Score'] = decision_matrix['Ranking_Score'] / max_score * 100
        
        # Calculate weighted total
        for column, weight in weights.items():
            decision_matrix[f'{column}_weighted'] = decision_matrix[column] * weight
        
        decision_matrix['Total_Score'] = sum(decision_matrix[f'{column}_weighted'] for column in weights.keys())
        
        # Create columns for visualization and explanation
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Create a heatmap of the decision matrix
            matrix_columns = ['Artist', 'Ranking_Score', 'Genre_Match', 'Target_Demo_Match', 
                             'Tour_Availability', 'Cost_Effectiveness', 'Social_Engagement', 'Total_Score']
            heatmap_data = decision_matrix[matrix_columns].copy()
            
            # Format column names for display
            heatmap_data.columns = [
                'Artist', 'Chart Performance', 'Pop Genre Match', 'Target Demo Match',
                'Tour Availability', 'Cost Effectiveness', 'Social Engagement', 'Total Score'
            ]
            
            # Create the heatmap figure
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=heatmap_data.iloc[:, 1:].values,
                x=heatmap_data.columns[1:],
                y=heatmap_data['Artist'],
                colorscale='Agsunset',
                text=[[f'{val:.1f}' for val in row] for row in heatmap_data.iloc[:, 1:].values],
                texttemplate='%{text}',
                textfont={"size":12}
            ))
            
            fig_heatmap.update_layout(
                title='Artist Selection Decision Matrix (0-100 Scale)',
                height=500,
                xaxis_title="Selection Criteria",
                yaxis_title="Artist",
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Create a bar chart for the final scores
            fig_final = px.bar(
                decision_matrix.sort_values('Total_Score', ascending=False),
                x='Artist',
                y='Total_Score',
                color='Total_Score',
                color_continuous_scale='Agsunset',
                text='Total_Score',
                labels={'Artist': 'Artist', 'Total_Score': 'Total Weighted Score'},
                title='Final Artist Ranking',
                height=400
            )
            
            fig_final.update_layout(
                xaxis_title="Artist",
                yaxis_title="Total Weighted Score",
                coloraxis_showscale=False
            )
            
            # Format text to display values with 1 decimal place
            fig_final.update_traces(texttemplate='%{y:.1f}', textposition='outside')
            
            # Add a threshold line for selection
            fig_final.add_hline(
                y=90,
                line_dash="dash",
                line_color="#FF78C4",
                annotation_text="Selection Threshold",
                annotation_position="right"
            )
            
            st.plotly_chart(fig_final, use_container_width=True)
        
        with col2:
            st.markdown("""
            <div class="bio-card">
                <h3>Final Selection Rationale</h3>
                <p>
                    The decision matrix evaluates artists across six weighted criteria:
                </p>
                <ul>
                    <li><span class="stat-highlight">Chart Performance</span>: 30%</li>
                    <li><span class="stat-highlight">Pop Genre Match</span>: 20%</li>
                    <li><span class="stat-highlight">Target Demo Match</span>: 20%</li>
                    <li><span class="stat-highlight">Tour Availability</span>: 10%</li>
                    <li><span class="stat-highlight">Cost Effectiveness</span>: 10%</li>
                    <li><span class="stat-highlight">Social Engagement</span>: 10%</li>
                </ul>
                <p>
                    <span class="stat-highlight">Sabrina Carpenter</span> achieved the highest total score of
                    <span class="stat-highlight">92.8</span> out of 100, confirming her as the optimal headliner
                    for our event.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show the key metrics
            st.metric(
                label="Sabrina's Final Score",
                value=f"{decision_matrix[decision_matrix['Artist'] == 'Sabrina Carpenter']['Total_Score'].values[0]:.1f}/100",
                delta=f"{decision_matrix[decision_matrix['Artist'] == 'Sabrina Carpenter']['Total_Score'].values[0] - decision_matrix[decision_matrix['Artist'] == 'Taylor Swift']['Total_Score'].values[0]:.1f} vs. 2nd place"
            )
            
            st.markdown("""
            <div class="insight-card">
                <h4>Additional Factors</h4>
                <p>
                    Beyond the quantitative analysis:
                </p>
                <ul>
                    <li>Current viral momentum on TikTok</li>
                    <li>Strong appeal to DC area college demographic</li>
                    <li>Recent album success and radio play</li>
                    <li>Tour routing compatibility with our event dates</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
    # Add a concluding section visible regardless of the visualization selected
    st.markdown("""
    <div class="conclusion-card" style="margin-top: 30px; padding: 20px; background-color: rgba(255, 120, 196, 0.1); border-radius: 10px; border-left: 5px solid #FF78C4;">
        <h3>Selection Conclusion</h3>
        <p>
            Based on comprehensive data analysis of chart performance, genre alignment, and demographic appeal,
            <span style="font-weight: bold; color: #FF78C4;">Sabrina Carpenter</span> was selected as the ideal headliner for our Capital One Arena concert.
        </p>
        <p>
            Her exceptional ranking score of <span style="font-weight: bold;">814.86</span> (13.3% higher than the second-ranked artist), 
            combined with perfect pop genre alignment for our target Gen Z and Millennial audience, makes her the optimal choice to maximize 
            attendance and audience satisfaction.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Tab 3: Gracie Selection Process
with tab3:
    st.markdown("<div class='section-header'>Gracie Selection Process</div>", unsafe_allow_html=True)
    
    st.subheader("Gracie's Selection Analysis")
    
    # Create columns for main content and explanation
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Create a dropdown to select which visualization to display
        viz_option = st.selectbox(
            "Select Visualization:",
            ["Cluster Analysis", "Selection Comparison", "Artist Similarity Network"],
            index=0
        )
        
        # Load artist data
        @st.cache_data
        def load_artist_data():
            try:
                # This is a placeholder - in a real app, you would load from a CSV
                artist_data = pd.read_csv('Python_Files/artist_scores.csv')
                
                # Get top 30 artists by rank plus Gracie
                top_artists = artist_data[
                    (artist_data['rank'] <= 30) | 
                    (artist_data['artists'] == "Gracie Abrams")
                ].sort_values('rank')
                
                # Assign clusters
                top_artists['cluster'] = 3  # Default to cluster 3
                top_artists.loc[
                    (top_artists['frequency_normalized'] > 0.7) & 
                    (top_artists['rank_sum_normalized'] > 0.4), 'cluster'] = 1  # High performers
                top_artists.loc[
                    (top_artists['frequency_normalized'] > 0.3) & 
                    (top_artists['cluster'] == 3), 'cluster'] = 2  # Medium popularity
                
                return top_artists
            except Exception as e:
                st.error(f"Error loading artist data: {e}")
                # Return sample data if file can't be loaded
                return create_sample_artist_data()
        
        # Create sample data if the real data can't be loaded
        def create_sample_artist_data():
            # Create sample data that mimics the structure of the real data
            sample_data = {
                'artists': [
                    "Sabrina Carpenter", "Taylor Swift", "Olivia Rodrigo", 
                    "Billie Eilish", "Ariana Grande", "Dua Lipa", 
                    "SZA", "Gracie Abrams", "Chappell Roan", "Charli XCX"
                ],
                'rank': [1, 2, 3, 4, 5, 6, 7, 31, 8, 9],
                'frequency_normalized': [0.95, 0.92, 0.88, 0.85, 0.82, 0.78, 0.75, 0.25, 0.72, 0.68],
                'rank_sum_normalized': [0.90, 0.88, 0.85, 0.82, 0.79, 0.76, 0.73, 0.20, 0.70, 0.67],
                'ranking_score': [92.5, 90.0, 86.5, 83.5, 80.5, 77.0, 74.0, 22.5, 71.0, 67.5]
            }
            
            df = pd.DataFrame(sample_data)
            
            # Assign clusters
            df['cluster'] = 3
            df.loc[
                (df['frequency_normalized'] > 0.7) & 
                (df['rank_sum_normalized'] > 0.4), 'cluster'] = 1  # High performers
            df.loc[
                (df['frequency_normalized'] > 0.3) & 
                (df['cluster'] == 3), 'cluster'] = 2  # Medium popularity
            
            return df
        
        # Load or create artist data
        artist_df = load_artist_data()
        
        # Function to load Spotify music features data (similar to the KNN analysis in the notebook)
        @st.cache_data
        def load_spotify_features():
            try:
                # Sample data from the Dataset
                artists = ['Sabrina Carpenter', 'Gracie Abrams', 'Olivia Rodrigo', 'Ariana Grande', 'Dua Lipa', 'Taylor Swift']
                
                # Create sample features data
                features_data = {
                    'Artists': artists,
                    'danceability': [0.72, 0.68, 0.65, 0.71, 0.75, 0.63],
                    'energy': [0.65, 0.61, 0.67, 0.68, 0.80, 0.62],
                    'speechiness': [0.12, 0.11, 0.14, 0.10, 0.08, 0.15],
                    'acousticness': [0.25, 0.35, 0.20, 0.15, 0.10, 0.30],
                    'liveness': [0.15, 0.14, 0.20, 0.18, 0.16, 0.12],
                    'valence': [0.70, 0.65, 0.55, 0.68, 0.72, 0.60],
                    'tempo': [118, 115, 120, 125, 122, 110],
                    'popularity': [85, 80, 92, 90, 88, 95],
                    'daily_rank': [10, 28, 5, 8, 7, 3],
                    'daily_movement': [1, 2, 0, -1, 1, 0],
                    'weekly_movement': [3, 5, -1, 0, 2, -2],
                    'Minimum Fees (in Dollars)': [500000, 150000, 450000, 650000, 550000, 800000]
                }
                
                features_df = pd.DataFrame(features_data)
                return features_df
                
            except Exception as e:
                st.error(f"Error loading Spotify features data: {e}")
                return pd.DataFrame()  # Return empty frame on error
        
        # Load Spotify features
        spotify_features = load_spotify_features()
        
        # Display Cluster Analysis visualization when selected
        if viz_option == "Cluster Analysis":
            # Create a scatter plot for clusters
            cluster_colors = {1: "#73a7bf", 2: "#8eb535", 3: "#8e6bc7"}
            
            # Create a copy of the dataframe to add jitter for dense areas
            plot_df = artist_df.copy()
            
            # Add small random offsets to points to reduce overlapping
            # This is a manual implementation of jitter
            np.random.seed(42)  # For reproducibility
            jitter_amount = 0.01
            plot_df["frequency_normalized"] = plot_df["frequency_normalized"] + np.random.uniform(-jitter_amount, jitter_amount, len(plot_df))
            plot_df["rank_sum_normalized"] = plot_df["rank_sum_normalized"] + np.random.uniform(-jitter_amount, jitter_amount, len(plot_df))
            
            # Ensure values stay within valid range (0-1)
            plot_df["frequency_normalized"] = plot_df["frequency_normalized"].clip(0, 1)
            plot_df["rank_sum_normalized"] = plot_df["rank_sum_normalized"].clip(0, 1)
            
            # Adjust size based on data density - smaller points in dense areas
            plot_df["point_size"] = 10  # Base size
            
            fig = px.scatter(
                plot_df,
                x="frequency_normalized",
                y="rank_sum_normalized",
                color="cluster",
                color_discrete_map=cluster_colors,
                hover_name="artists",
                size="point_size",
                size_max=15,  # Limit maximum size
                opacity=0.6,  # Add transparency to see overlapping points
                labels={
                    "frequency_normalized": "Normalized Frequency",
                    "rank_sum_normalized": "Normalized Rank Sum",
                    "cluster": "Clusters"
                },
                title="Artist Clustering Analysis"
            )
            
            # Improve hover information
            fig.update_traces(
                hovertemplate="<b>%{hovertext}</b><br>Frequency: %{x:.2f}<br>Rank Sum: %{y:.2f}<extra></extra>"
            )
            
            # Highlight Sabrina and Gracie with special markers
            sabrina_data = artist_df[artist_df["artists"] == "Sabrina Carpenter"]
            gracie_data = artist_df[artist_df["artists"] == "Gracie Abrams"]
            
            if not sabrina_data.empty and not gracie_data.empty:
                highlight_df = pd.concat([sabrina_data, gracie_data])
                
                # Add highlighted points
                fig.add_trace(
                    go.Scatter(
                        x=highlight_df["frequency_normalized"],
                        y=highlight_df["rank_sum_normalized"],
                        mode="markers+text",
                        marker=dict(
                            size=18,
                            color="red",
                            line=dict(width=2, color="black")
                        ),
                        text=highlight_df["artists"],
                        textposition="bottom center",
                        name="Selected Artists",
                        hovertemplate="<b>%{text}</b><br>Frequency: %{x:.2f}<br>Rank Sum: %{y:.2f}<extra></extra>"
                    )
                )
            
            # Add cluster descriptions
            fig.update_layout(
                height=550,  # Slightly taller for better spacing
                xaxis=dict(range=[0, 1.05]),
                yaxis=dict(range=[0, 1.05]),
                legend_title="Clusters",
                legend=dict(
                    orientation="h",  # Horizontal legend
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                annotations=[
                    dict(
                        x=0.5,
                        y=1.05,
                        xref="paper",
                        yref="paper",
                        text="Cluster 1: Top performers | Cluster 2: Popular artists | Cluster 3: Emerging artists",
                        showarrow=False,
                        font=dict(size=12)
                    )
                ]
            )
            
            # Improve axis appearance
            fig.update_xaxes(
                gridcolor='lightgray',
                zerolinecolor='lightgray',
                tickformat='.1f'
            )
            fig.update_yaxes(
                gridcolor='lightgray',
                zerolinecolor='lightgray',
                tickformat='.1f'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="bio-card">
                <h3>Cluster Analysis Insights</h3>
                <p>
                    We performed clustering based on genre-related features and identified three distinct clusters:
                </p>
                <ul>
                    <li><span class="stat-highlight">Cluster 1:</span> Top performing artists with high frequency and rank metrics</li>
                    <li><span class="stat-highlight">Cluster 2:</span> Popular artists with moderate frequency metrics</li>
                    <li><span class="stat-highlight">Cluster 3:</span> Emerging artists with high growth potential</li>
                </ul>
                <p>
                    Sabrina Carpenter belongs to Cluster 1 (top performers), while we found Gracie Abrams in 
                    Cluster 3 (emerging artists with high potential), providing complementary balance to the lineup.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        # Display Selection Comparison visualization when selected
        elif viz_option == "Selection Comparison":
            # Get Sabrina, Gracie and top 5 alternatives
            sabrina = artist_df[artist_df["artists"] == "Sabrina Carpenter"]
            gracie = artist_df[artist_df["artists"] == "Gracie Abrams"]
            
            # Get alternatives (excluding Sabrina and Gracie)
            alternatives = artist_df[
                ~artist_df["artists"].isin(["Sabrina Carpenter", "Gracie Abrams"])
            ].sort_values("ranking_score", ascending=False).head(5)
            
            # Combine for comparison
            comparison_df = pd.concat([sabrina, gracie, alternatives])
            
            # Create bar chart
            fig = px.bar(
                comparison_df,
                x="artists",
                y="ranking_score",
                color="artists",
                color_discrete_sequence=[
                    "#36b1b5" if artist in ["Sabrina Carpenter", "Gracie Abrams"] else "#125fb8"
                    for artist in comparison_df["artists"]
                ],
                labels={
                    "artists": "Artist",
                    "ranking_score": "Ranking Score"
                },
                title="Artist Selection Comparison"
            )
            
            # Add annotation for Gracie
            if not gracie.empty:
                fig.add_annotation(
                    x=gracie["artists"].values[0],
                    y=gracie["ranking_score"].values[0] + 5,
                    text="Selected as second artist",
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=-30
                )
            
            fig.update_layout(
                height=500,
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class="bio-card">
                <h3>Selection Rationale</h3>
                <p>
                    From our analysis, we selected Gracie Abrams as the second main artist because:
                </p>
                <ul>
                    <li>Strong ranking score compared to other artists in her cluster</li>
                    <li>Growing popularity with significant audience overlap with Sabrina's fanbase</li>
                    <li>Complementary musical style that aligns with our target demographic</li>
                    <li>Offers demographic diversity while maintaining genre coherence</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # NEW OPTION: Artist Similarity Network based on the Python notebook
        elif viz_option == "Artist Similarity Network":
            if not spotify_features.empty:
                st.write("### Spotify Music Features Analysis")
                st.write("Using K-Nearest Neighbors to find artists similar to Sabrina Carpenter based on musical features")
                
                # Display music features with descriptions
                with st.expander("About Music Features"):
                    st.markdown("""
                    - **Danceability**: How suitable a track is for dancing (0-1)
                    - **Energy**: Measure of intensity and activity (0-1)
                    - **Speechiness**: Presence of spoken words (0-1)
                    - **Acousticness**: Amount of acoustic sound (0-1)
                    - **Liveness**: Presence of audience in recording (0-1)
                    - **Valence**: Musical positiveness/mood (0-1)
                    - **Tempo**: Beats per minute
                    """)
                
                # Create KNN analysis similar to the notebook
                # Select features for analysis
                features_for_knn = ['danceability', 'energy', 'speechiness', 'acousticness', 'liveness', 'valence', 'tempo', 'popularity']
                X = spotify_features[features_for_knn]
                
                # Check if we have enough data
                if len(X) >= 2:
                    # Standardize features
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # KNN model
                    from sklearn.neighbors import NearestNeighbors
                    knn_model = NearestNeighbors(n_neighbors=len(X), metric='euclidean')
                    knn_model.fit(X_scaled)
                    
                    # Find Sabrina Carpenter
                    try:
                        sabrina_index = spotify_features[spotify_features['Artists'] == 'Sabrina Carpenter'].index[0]
                        distances, indices = knn_model.kneighbors(X_scaled[sabrina_index].reshape(1, -1))
                        
                        # Create network graph showing artist similarities (similar to the notebook)
                        fig = go.Figure()
                        
                        # Create a circle layout for the network
                        num_nodes = min(6, len(indices[0]))  # Limit to 6 artists for clarity
                        angles = np.linspace(0, 2*np.pi, num_nodes, endpoint=False).tolist()
                        
                        # Central node (Sabrina)
                        center_x, center_y = 0, 0
                        radius = 1
                        
                        # Add edges from center to each artist
                        for i in range(1, num_nodes):
                            idx = indices[0][i]
                            angle = angles[i]
                            x = radius * np.cos(angle)
                            y = radius * np.sin(angle)
                            
                            # Add edge
                            edge_width = 5 * (1 - distances[0][i] / distances[0].max())  # Scale width by similarity
                            
                            fig.add_trace(go.Scatter(
                                x=[center_x, x, None], 
                                y=[center_y, y, None],
                                mode='lines',
                                line=dict(width=edge_width, color='gray'),
                                hoverinfo='none',
                                showlegend=False
                            ))
                            
                            # Add node for this artist
                            artist_name = spotify_features.iloc[idx]['Artists']
                            fee = spotify_features.iloc[idx]['Minimum Fees (in Dollars)']
                            
                            # Use different color/size for Gracie
                            color = 'purple' if artist_name == 'Gracie Abrams' else '#43afe0'
                            size = 20 if artist_name == 'Gracie Abrams' else 15
                            
                            fig.add_trace(go.Scatter(
                                x=[x], 
                                y=[y],
                                mode='markers+text',
                                marker=dict(size=size, color=color),
                                text=artist_name,
                                textposition="top left",
                                hovertemplate=f"{artist_name}<br>Fee: ${fee:,.0f}<extra></extra>",
                                showlegend=False
                            ))
                            
                            # Add special annotation for Gracie Abrams
                            if artist_name == 'Gracie Abrams':
                                fig.add_annotation(
                                    x=x,
                                    y=y+0.15,
                                    text="Best Choice",
                                    showarrow=True,
                                    arrowhead=2,
                                    ax=0,
                                    ay=-30,
                                    font=dict(color='purple')
                                )
                        
                        # Add center node (Sabrina)
                        fig.add_trace(go.Scatter(
                            x=[center_x], 
                            y=[center_y],
                            mode='markers+text',
                            marker=dict(size=25, color='#1db949'),
                            text='Sabrina Carpenter',
                            textposition="bottom right",
                            hovertemplate=f"Sabrina Carpenter<br>Fee: ${spotify_features.iloc[sabrina_index]['Minimum Fees (in Dollars)']:,.0f}<extra></extra>",
                            showlegend=False
                        ))
                        
                        # Update layout
                        fig.update_layout(
                            title="Artists Similar to Sabrina Carpenter (Based on Music Features)",
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            height=600,
                            margin=dict(l=20, r=20, t=40, b=20),
                            plot_bgcolor='rgba(240,240,240,0.8)'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Create a radar chart comparing Sabrina and Gracie's features
                        gracie_index = spotify_features[spotify_features['Artists'] == 'Gracie Abrams'].index[0]
                        
                        # Prepare data for radar chart
                        radar_features = ['danceability', 'energy', 'speechiness', 'acousticness', 'liveness', 'valence']
                        
                        # Create radar chart
                        radar_fig = go.Figure()
                        
                        # Add Sabrina
                        radar_fig.add_trace(go.Scatterpolar(
                            r=spotify_features.iloc[sabrina_index][radar_features].tolist(),
                            theta=radar_features,
                            fill='toself',
                            name='Sabrina Carpenter',
                            line_color='#1DB954'
                        ))
                        
                        # Add Gracie
                        radar_fig.add_trace(go.Scatterpolar(
                            r=spotify_features.iloc[gracie_index][radar_features].tolist(),
                            theta=radar_features,
                            fill='toself',
                            name='Gracie Abrams',
                            line_color='red'
                        ))
                        
                        radar_fig.update_layout(
                            polar=dict(
                                radialaxis=dict(visible=True)
                            ),
                            title="Music Feature Comparison",
                            height=400
                        )
                        
                        st.plotly_chart(radar_fig, use_container_width=True)
                        
                        # Display similarity and fee comparison table
                        similar_artists = ['Sabrina Carpenter']
                        
                        # Ensure we don't get out of bounds with indices
                        similar_indices = indices[0][1:min(6, len(indices[0]))]
                        for idx in similar_indices:
                            similar_artists.append(spotify_features.iloc[idx]['Artists'])
                        
                        # Extract data for these artists
                        compare_df = spotify_features[spotify_features['Artists'].isin(similar_artists)].copy()
                        
                        # Add similarity score (1 - normalized distance)
                        # Calculate for all artists except Sabrina (who has similarity 1.0 to herself)
                        similarity_scores = [1.0]  # Sabrina's self-similarity
                        max_dist = distances[0][1:].max()  # Max distance excluding self
                        
                        for i in range(1, len(distances[0])):
                            if i < len(compare_df):
                                similarity = 1 - (distances[0][i] / max_dist)
                                similarity_scores.append(round(similarity, 2))
                        
                        # Ensure we have scores for all rows
                        while len(similarity_scores) < len(compare_df):
                            similarity_scores.append(0)
                            
                        compare_df['Similarity Score'] = similarity_scores
                        
                        # Format the fee column for display
                        compare_df['Fee'] = compare_df['Minimum Fees (in Dollars)'].apply(lambda x: f"${x:,.0f}")
                        
                        # Sort by similarity
                        compare_df = compare_df.sort_values('Similarity Score', ascending=False)
                        
                        # Display just what we need
                        display_cols = ['Artists', 'Similarity Score', 'Fee', 'popularity']
                        st.write("### Artist Similarity Comparison")
                        
                        # Create styled dataframe
                        def highlight_gracie(s):
                            return ['background-color: rgba(255,0,0,0.2)' if x == 'Gracie Abrams' else '' for x in s]
                        
                        st.dataframe(
                            compare_df[display_cols].style.apply(highlight_gracie, subset=['Artists']), 
                            use_container_width=True,
                            height=250
                        )
                        
                        # Add explanation
                        st.markdown("""
                        <div class="bio-card mt-4">
                            <h3>KNN Analysis Results</h3>
                            <p>
                                Using K-Nearest Neighbors algorithm on Spotify's music features, we identified artists most similar to 
                                Sabrina Carpenter in terms of musical style and audience appeal. The algorithm considered factors like:
                            </p>
                            <ul>
                                <li>Acoustic properties of songs (danceability, energy, etc.)</li>
                                <li>Popularity metrics</li>
                                <li>Chart performance</li>
                            </ul>
                            <p>
                                <span class="stat-highlight">Gracie Abrams</span> was identified as an optimal match that balances 
                                musical similarity with Sabrina while offering a significantly lower booking fee. This provides excellent 
                                value while maintaining artistic coherence.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    except IndexError:
                        st.error("Sabrina Carpenter not found in the dataset")
                else:
                    st.error("Not enough data for KNN analysis")
            else:
                st.error("Could not load Spotify features data")
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>Analysis Methodology</h3>
            <p>Our artist selection process incorporated multiple factors:</p>
            <ul>
                <li>Social media presence</li>
                <li>Streaming platform popularity</li>
                <li>Billboard chart performance</li>
                <li>Audience demographic overlap</li>
                <li>Genre compatibility</li>
            </ul>
            <p>Each factor was normalized and weighted to create our final ranking scores.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card">    
            <h3>Key Findings</h3>
            <p>While Gracie wasn't in the same cluster as Sabrina, her metrics and audience alignment made her
            the optimal choice for our second main artist.</p>
            <p>This selection provides a balanced portfolio of established and emerging talent to maximize 
            appeal across our target demographics.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add KNN method explanation when artist similarity is selected
        if viz_option == "Artist Similarity Network":
            st.markdown("""
            <div class="info-card mt-4">
                <h3>KNN Analysis Method</h3>
                <p>Our K-Nearest Neighbors analysis:</p>
                <ol>
                    <li>Collected Spotify audio features for top artists</li>
                    <li>Standardized features to ensure equal weighting</li>
                    <li>Applied KNN to find artists most similar to Sabrina</li>
                    <li>Ranked results by similarity score</li>
                    <li>Combined with fee data to identify optimal value</li>
                </ol>
                <p>The red highlight indicates Gracie Abrams as our selected artist based on both similarity and cost-effectiveness.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Add conclusion section
    st.markdown("""
    <div class="conclusion-card bg-gray-100 p-4 rounded-lg mt-4">
        <h3 class="font-semibold">Conclusion</h3>
        <p>
            The data-driven selection of Gracie Abrams provides an optimal complement to Sabrina Carpenter for the
            Capital One Arena concert. While not in the same performance cluster, Gracie's growing popularity among
            the same target demographic and complementary musical style creates a balanced lineup that maximizes
            audience appeal and engagement potential.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Tab 4: Artist Justification Dashboard
with tab4:
    st.markdown("<div class='section-header'>Artist Biography & Justification</div>", unsafe_allow_html=True)

    # Create nested tabs within tab2
    nested_tab1, nested_tab2 = st.tabs(["Sabrina Carpenter", "Gracie Abrams"])

    # Timeline data with USA sales figures added
    timeline_data = {
        'Year': [2015, 2016, 2018, 2019, 2021, 2022, 2023, 2024],
        'Album/Milestone': ['Eyes Wide Open', 'EVOLution', 'Singular: Act I', 'Singular: Act II', 
                        'Skin (Breakthrough Single)', 'Emails I Can\'t Send', 'Nonsense (Viral Hit)', 'Short n\' Sweet & Espresso'],
        'USA_Sales': [30000, 35000, 25000, 20000, None, 35000, None, 135000],
        'Global_Sales': [40000, 45000, 40000, 35000, None, 75000, None, 315000] 
    }

    # Create DataFrame
    timeline_df = pd.DataFrame(timeline_data)

    # Add End_Year for timeline visualization
    timeline_df['End_Year'] = timeline_df['Year'] + 1

    with nested_tab1:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("""
            <div class="bio-card">
                <h3>Sabrina Carpenter</h3>
                <p>
                    American singer-songwriter, actress and content creator known for her catchy pop melodies and relatable lyrics. 
                    She gained initial fame through Disney Channel's "Girl Meets World" before focusing on her music career.
                </p>
                <h4>Career Highlights:</h4>
                <ul>
                    <li><span class="stat-highlight">6</span> studio albums</li>
                    <li><span class="stat-highlight">29</span> singles</li>
                    <li><span class="stat-highlight">34</span> music videos</li>
                    <li><span class="stat-highlight">3</span> Grammy nominations</li>
                    <li>Multiple Billboard Hot 100 chart appearances</li>
                    <li>Successful world tours (Emails I Can't Send Tour, Short n' Sweet Tour)</li>
                </ul>
                <p>
                    Her recent album "Short n' Sweet" (2023) has propelled her to mainstream success, 
                    especially with viral hits like "Espresso" and "Please Please Please".
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("")
            st.markdown("")
            st.markdown("")
            st.markdown("### Career Timeline")

            # Create the timeline chart using a different approach with go.Bar
            fig = go.Figure()

            # Add a bar for each album/milestone
            for i, row in timeline_df.iterrows():
                fig.add_trace(go.Bar(
                    x=[row['End_Year'] - row['Year']],  # Width of the bar
                    y=[row['Album/Milestone']],
                    orientation='h',
                    base=row['Year'],  # Start position
                    marker_color=px.colors.qualitative.Pastel[i % len(px.colors.qualitative.Pastel)],
                    name=row['Album/Milestone'],
                    hovertemplate=f"{row['Album/Milestone']}<br>Year: {row['Year']}<br>Global_Sales: {row['Global_Sales'] if pd.notna(row['Global_Sales']) else 'N/A'}"
                ))

            # Update layout
            fig.update_layout(
                height=350,
                showlegend=False,
                xaxis_title="",
                yaxis_title="",
                title="Sabrina Carpenter Global_Sales Timeline",
                xaxis=dict(
                    tickmode='array',
                    tickvals=list(range(2015, 2025)),
                    ticktext=[str(year) for year in range(2015, 2025)],
                ),
                barmode='overlay'
            )

            # Show timeline chart
            st.plotly_chart(fig, use_container_width=True)

            # Create a sales bar chart
            albums_with_sales = timeline_df.dropna(subset=['USA_Sales']).copy()
            albums_with_sales['USA_Sales_formatted'] = albums_with_sales['USA_Sales'].apply(lambda x: f"{int(x):,}")

            fig_sales = px.bar(
                albums_with_sales,
                x='Album/Milestone',
                y='USA_Sales',
                color='Album/Milestone',
                color_discrete_sequence=px.colors.qualitative.Pastel,
                text='USA_Sales_formatted'
            )

            fig_sales.update_layout(
                title="USA Sales by Album",
                xaxis_title="",
                yaxis_title="Units Sold",
                showlegend=False,
            )

            fig_sales.update_traces(textposition='outside')

            # Show sales chart
            st.plotly_chart(fig_sales, use_container_width=True)
        
        with col2:
            st.markdown("### Viral Success Metrics")
            
            # Metrics row
            metric1, metric2, metric3 = st.columns(3)
            with metric1:
                st.metric("Streams on Spotify", "4.2B+")
            with metric2:
                st.metric("TikTok Videos", "3.5M+")
            with metric3:
                st.metric("Instagram Followers", "31M+")
            
            # Social engagement
            st.markdown("### Social Media Engagement")
            
            social_data = {
                'Platform': ['TikTok', 'Instagram', 'YouTube', 'Twitter'],
                'Followers (M)': [20.5, 31.0, 7.8, 4.2],
                'Engagement Rate (%)': [8.5, 7.2, 6.4, 4.8]
            }
            
            social_df = pd.DataFrame(social_data)
            
            fig = px.scatter(
                social_df,
                x="Followers (M)",
                y="Engagement Rate (%)",
                size="Followers (M)",
                color="Platform",
                text="Platform",
                size_max=50,
                title="Social Media Presence"
            )
            
            fig.update_traces(textposition='top center')
            fig.update_layout(height=400)
            
            st.plotly_chart(fig, use_container_width=True)

            # Display total USA sales
            total_sales = int(albums_with_sales['USA_Sales'].sum())
            st.metric("Total USA Sales", f"{total_sales:,} units")

            # Create a pie chart showing sales distribution
            fig_pie = px.pie(
                albums_with_sales,
                values='USA_Sales',
                names='Album/Milestone',
                color='Album/Milestone',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )

            fig_pie.update_layout(
                title="Distribution of USA Album Sales"
            )

            # Show pie chart
            st.plotly_chart(fig_pie, use_container_width=True)

    with nested_tab2:
        col1, col2 = st.columns([1, 1])
        
        # Timeline data for Gracie Abrams
        gracie_timeline_data = {
            'Year': [2019, 2020, 2021, 2022, 2023, 2024],
            'Album/Milestone': ['Minor (EP)', 'This Is What It Feels Like (EP)', 
                            'Mess It Up (Single)', 'Good Riddance', 
                            'The Secret of Us (Single)', 'The Secret of Us'],
            'USA_Sales': [15000, 18000, None, 42000, None, 65000],
            'Global_Sales': [25000, 30000, None, 92000, None, 130000] 
        }

        # Create DataFrame
        gracie_timeline_df = pd.DataFrame(gracie_timeline_data)

        # Add End_Year for timeline visualization
        gracie_timeline_df['End_Year'] = gracie_timeline_df['Year'] + 1
        
        with col1:
            st.markdown("""
            <div class="bio-card">
                <h3>Gracie Abrams</h3>
                <p>
                    American singer-songwriter known for her introspective lyrics and intimate indie-pop sound.
                    She gained recognition through social media platforms before releasing her first EP.
                </p>
                <h4>Career Highlights:</h4>
                <ul>
                    <li><span class="stat-highlight">2</span> studio albums</li>
                    <li><span class="stat-highlight">2</span> EPs</li>
                    <li><span class="stat-highlight">15</span> singles</li>
                    <li><span class="stat-highlight">1</span> Grammy nomination</li>
                    <li>Collaborated with Taylor Swift on The Eras Tour</li>
                    <li>Successful tours (This Is What It Feels Like Tour, Good Riddance Tour)</li>
                </ul>
                <p>
                    Her debut album "Good Riddance" (2023) produced by Aaron Dessner established her as a 
                    significant voice in the indie-pop scene, followed by her sophomore album "The Secret of Us" (2024).
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("")
            st.markdown("")
            st.markdown("")
            st.markdown("### Career Timeline")

            # Create the timeline chart using a different approach with go.Bar
            fig = go.Figure()

            # Add a bar for each album/milestone
            for i, row in gracie_timeline_df.iterrows():
                fig.add_trace(go.Bar(
                    x=[row['End_Year'] - row['Year']],  # Width of the bar
                    y=[row['Album/Milestone']],
                    orientation='h',
                    base=row['Year'],  # Start position
                    marker_color=px.colors.qualitative.Pastel[i % len(px.colors.qualitative.Pastel)],
                    name=row['Album/Milestone'],
                    hovertemplate=f"{row['Album/Milestone']}<br>Year: {row['Year']}<br>Global_Sales: {row['Global_Sales'] if pd.notna(row['Global_Sales']) else 'N/A'}"
                ))

            # Update layout
            fig.update_layout(
                height=350,
                showlegend=False,
                xaxis_title="",
                yaxis_title="",
                title="Gracie Abrams Global Sales Timeline",
                xaxis=dict(
                    tickmode='array',
                    tickvals=list(range(2019, 2025)),
                    ticktext=[str(year) for year in range(2019, 2025)],
                ),
                barmode='overlay'
            )

            # Show timeline chart
            st.plotly_chart(fig, use_container_width=True)

            # Create a sales bar chart
            gracie_albums_with_sales = gracie_timeline_df.dropna(subset=['USA_Sales']).copy()
            gracie_albums_with_sales['USA_Sales_formatted'] = gracie_albums_with_sales['USA_Sales'].apply(lambda x: f"{int(x):,}")

            fig_sales = px.bar(
                gracie_albums_with_sales,
                x='Album/Milestone',
                y='USA_Sales',
                color='Album/Milestone',
                color_discrete_sequence=px.colors.qualitative.Pastel,
                text='USA_Sales_formatted'
            )

            fig_sales.update_layout(
                title="USA Sales by Album",
                xaxis_title="",
                yaxis_title="Units Sold",
                showlegend=False,
            )

            fig_sales.update_traces(textposition='outside')

            # Show sales chart
            st.plotly_chart(fig_sales, use_container_width=True)
        
        with col2:
            st.markdown("### Viral Success Metrics")
            
            # Metrics row
            metric1, metric2, metric3 = st.columns(3)
            with metric1:
                st.metric("Streams on Spotify", "1.8B+")
            with metric2:
                st.metric("TikTok Videos", "1.2M+")
            with metric3:
                st.metric("Instagram Followers", "1.9M+")
            
            # Social engagement
            st.markdown("### Social Media Engagement")
            
            social_data = {
                'Platform': ['TikTok', 'Instagram', 'YouTube', 'Twitter'],
                'Followers (M)': [0.8, 1.9, 0.5, 0.4],
                'Engagement Rate (%)': [9.2, 7.8, 5.3, 3.9]
            }
            
            social_df = pd.DataFrame(social_data)
            
            fig = px.scatter(
                social_df,
                x="Followers (M)",
                y="Engagement Rate (%)",
                size="Followers (M)",
                color="Platform",
                text="Platform",
                size_max=50,
                title="Social Media Presence"
            )
            
            fig.update_traces(textposition='top center')
            fig.update_layout(height=400)
            
            st.plotly_chart(fig, use_container_width=True)

            # Display total USA sales
            gracie_total_sales = int(gracie_albums_with_sales['USA_Sales'].sum())
            st.metric("Total USA Sales", f"{gracie_total_sales:,} units")

            # Create a pie chart showing sales distribution
            fig_pie = px.pie(
                gracie_albums_with_sales,
                values='USA_Sales',
                names='Album/Milestone',
                color='Album/Milestone',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )

            fig_pie.update_layout(
                title="Distribution of USA Album Sales"
            )

            # Show pie chart
            st.plotly_chart(fig_pie, use_container_width=True)

# Tab 5: Hit Song Performance
with tab5:
    st.markdown("<div class='section-header'>Hit Song Performance Analysis</div>", unsafe_allow_html=True)
    nested_tab1, nested_tab2 = st.tabs(["Sabrina Carpenter", "Gracie Abrams"])

    with nested_tab1:
        col1, col2 = st.columns([3, 1])
        with col1:
            # Create a dropdown for different metrics - with unique key
            metric = st.selectbox(
                "Select Performance Metric:",
                ["Streams", "Chart Position", "Grammy Nominations"],
                key="sabrina_metric_selector"  # Added unique key
            )
            if metric == "Streams":
                fig = px.bar(
                    sabrina_songs_df.sort_values(by="streams", ascending=False).head(10),
                    y="song_title",
                    x="streams",
                    color="album",
                    labels={"song_title": "Song", "streams": "Stream Count", "album": "Album"},
                    title="Top 10 Songs by Stream Count",
                    orientation='h',
                    text_auto='.2s'
                )
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                
            elif metric == "Chart Position":
                # For chart position, lower is better, so we need to sort differently
                chart_df = sabrina_songs_df.sort_values(by="chart_position", ascending=True).head(10)
                fig = px.bar(
                    chart_df,
                    y="song_title",
                    x="chart_position",
                    color="album",
                    labels={"song_title": "Song", "chart_position": "Billboard Hot 100 Position", "album": "Album"},
                    title="Top 10 Songs by Chart Position",
                    orientation='h',
                    text=chart_df["chart_position"]
                )
                fig.update_layout(yaxis={'categoryorder':'total descending'})
                fig.update_xaxes(autorange="reversed")  # Lower numbers (better positions) should be longer bars
                
            else:  # Grammy Nominations
                fig = px.bar(
                    sabrina_songs_df.sort_values(by="grammy_nominations", ascending=False).head(10),
                    y="song_title",
                    x="grammy_nominations",
                    color="album",
                    labels={"song_title": "Song", "grammy_nominations": "Grammy Nominations", "album": "Album"},
                    title="Songs by Grammy Nominations",
                    orientation='h',
                    text=sabrina_songs_df.sort_values(by="grammy_nominations", ascending=False).head(10)["grammy_nominations"]
                )
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        # Add song filtering by album - with unique key
            st.markdown("### Filter by Album")
            album_choice = st.selectbox(
                "Select Album:",
                ["All Albums"] + list(sabrina_songs_df["album"].unique()),
                key="sabrina_album_selector"  # Added unique key
            )
            if album_choice != "All Albums":
                filtered_songs = sabrina_songs_df[sabrina_songs_df["album"] == album_choice]["song_title"].tolist()
                st.write(f"Songs from {album_choice}:")
                for song in filtered_songs:
                    st.markdown(f"- {song}")
            else:
                st.write("Select an album to see its songs")
        
        with col2:
            st.markdown("### Performance Highlights")
            
            st.markdown("""
            <div class="bio-card">
                <h4>Stream Milestones</h4>
                <p><span class="stat-highlight">Espresso</span>: 1.2B+ streams across platforms</p>
                <p><span class="stat-highlight">Nonsense</span>: 900M+ streams with viral TikTok trend</p>
                <p><span class="stat-highlight">Please Please Please</span>: 750M+ streams</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(" ")

            st.markdown("""
            <div class="bio-card mt-4">    
                <h4>Chart Achievements</h4>
                <p><span class="stat-highlight">Billboard Hot 100</span>: Multiple top 10 entries</p>
                <p><span class="stat-highlight">Pop Airplay</span>: Strong radio performance</p>
                <p><span class="stat-highlight">Global Charts</span>: International appeal</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(" ")
            st.markdown("""
            <div class="bio-card mt-4">    
                <h4>Awards & Recognition</h4>
                <p>Grammy nominations for breakthrough performances</p>
                <p>MTV Video Music Awards nominations</p>
                <p>Billboard Women in Music recognition</p>
            </div>
            """, unsafe_allow_html=True)

    with nested_tab2:
        col1, col2 = st.columns([3, 1])
        with col1:
            metric = st.selectbox(
                "Select Performance Metric:",
                ["Streams", "Chart Position", "Grammy Nominations"],
                key="gracie_metric_selector"
            )
            
            if metric == "Streams":
                fig = px.bar(
                    gracie_songs_df.sort_values(by="streams", ascending=False).head(10),
                    y="song_title",
                    x="streams",
                    color="album",
                    labels={"song_title": "Song", "streams": "Stream Count", "album": "Album"},
                    title="Top 10 Songs by Stream Count",
                    orientation='h',
                    text_auto='.2s'
                )
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                
            elif metric == "Chart Position":

                chart_df = gracie_songs_df.sort_values(by="chart_position", ascending=True).head(10)
                fig = px.bar(
                    chart_df,
                    y="song_title",
                    x="chart_position",
                    color="album",
                    labels={"song_title": "Song", "chart_position": "Billboard Hot 100 Position", "album": "Album"},
                    title="Top 10 Songs by Chart Position",
                    orientation='h',
                    text=chart_df["chart_position"]
                )
                fig.update_layout(yaxis={'categoryorder':'total descending'})
                fig.update_xaxes(autorange="reversed")
                
            else:  # Grammy Nominations
                fig = px.bar(
                    gracie_songs_df.sort_values(by="grammy_nominations", ascending=False).head(10),
                    y="song_title",
                    x="grammy_nominations",
                    color="album",
                    labels={"song_title": "Song", "grammy_nominations": "Grammy Nominations", "album": "Album"},
                    title="Songs by Grammy Nominations",
                    orientation='h',
                    text=gracie_songs_df.sort_values(by="grammy_nominations", ascending=False).head(10)["grammy_nominations"]
                )
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Filter by Album")
            album_choice = st.selectbox(
                "Select Album:",
                ["All Albums"] + list(gracie_songs_df["album"].unique()),
                key="gracie_album_selector"
            )

            if album_choice != "All Albums":
                filtered_songs = gracie_songs_df[gracie_songs_df["album"] == album_choice]["song_title"].tolist()
                st.write(f"Songs from {album_choice}:")
                for song in filtered_songs:
                    st.markdown(f"- {song}")
            else:
                st.write("Select an album to see its songs")
        
        with col2:
            st.markdown("### Performance Highlights")
            
            st.markdown("""
            <div class="bio-card">
                <h4>Stream Milestones</h4>
                <p><span class="stat-highlight">Risk</span>: 650M+ streams across platforms</p>
                <p><span class="stat-highlight">I miss you, I'm sorry</span>: 450M+ streams</p>
                <p><span class="stat-highlight">Block me out</span>: 380M+ streams</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(" ")
            st.markdown("""
            <div class="bio-card mt-4">
                <h4>Chart Achievements</h4>
                <p><span class="stat-highlight">Billboard Hot 100</span>: First entry with "Where do we go now?"</p>
                <p><span class="stat-highlight">Alternative Charts</span>: Multiple top 20 entries</p>
                <p><span class="stat-highlight">UK Charts</span>: Growing international presence</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(" ")
            st.markdown("""
            <div class="bio-card mt-4">    
                <h4>Awards & Recognition</h4>
                <p>Grammy nomination for Best New Artist</p>
                <p>Acclaimed Taylor Swift tour opener</p>
                <p>iHeartRadio Music Awards nomination</p>
            </div>
            """, unsafe_allow_html=True)
