import streamlit as st
import pandas as pd
import pickle
from textblob import TextBlob
import datetime
import matplotlib.pyplot as plt
import altair as alt
from wordcloud import WordCloud
from collections import Counter
import plotly.express as px
import numpy as np

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="Amazon Insights Hub",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS for Theming and Wider Content Block ---
st.markdown("""
    <style>
    /* 1. Overall Page Background & Default Text Color */
    body {
        background-color: #222831 !important; /* Darker gray page background */
        color: #EAE4D5 !important; /* Beige/off-white default text */
    }
    .stApp {
        background-color: #222831 !important; /* Darker gray page background */
    }

    /* 2. Main Content Area ("Div") Background & Styling */
    .main .block-container {
        max-width: 95%; 
        padding-left: 2.5rem; 
        padding-right: 2.5rem;
        padding-top: 1rem; 
        padding-bottom: 3rem; 
        background-color: #F2F2F2 !important; /* Light gray background for the main content area */
        color: #000000 !important; /* Text inside this block should be black for contrast */
        border-radius: 10px; 
        box-shadow: 0 6px 15px rgba(0,0,0,0.25); /* Adjusted shadow for darker page background */
    }

    /* 3. Global Text Color Application & Header Adjustments */
    h1, h2, h3, h4, h5, h6 {
        color: #EAE4D5 !important; /* Beige/off-white for headers on dark page background */
    }
    /* Specifically reduce top margin for the main title (h1) */
    h1 {
        margin-top: 0rem !important; 
        padding-top: 0rem !important; 
    }
    
    p, li, .stMarkdown {
        color: #EAE4D5 !important; /* Beige/off-white for general text on dark page background */
    }
    
    /* Specific Streamlit Widget Text Styling */
    label[data-testid="stWidgetLabel"] p, div[data-testid="stWidgetLabel"] p {
        color: #EAE4D5 !important; /* Beige/off-white for labels on dark page background */
        font-weight: 500 !important; 
    }

    /* --- MODIFIED BUTTON STYLING --- */
    .stButton>button, .stDownloadButton>button {
        background-color: #667788 !important; /* Slightly lighter dark gray for buttons for contrast */
        border: 1px solid #667788 !important; 
        border-radius: 6px !important;
        color: #EAE4D5 !important; /* Beige/off-white text for buttons */
    }
    .stButton>button p, .stDownloadButton>button p { 
        color: #EAE4D5 !important; /* Beige/off-white text for buttons */
        font-weight: 500 !important;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        background-color: #7A8B9E !important; /* Lighter on hover */
        border-color: #7A8B9E !important;
        color: #EAE4D5 !important; 
    }
     .stButton>button:active, .stDownloadButton>button:active { 
        background-color: #556677 !important;
        border-color: #556677 !important;
        color: #EAE4D5 !important;
    }
    .stButton>button:focus, .stDownloadButton>button:focus {
        color: #EAE4D5 !important;
    }
    /* --- END OF MODIFIED BUTTON STYLING --- */

    div[data-testid="stRadio"] label span, div[data-testid="stSelectbox"] li span {
        color: #EAE4D5 !important; /* Beige/off-white for radio/selectbox options */
    }
    
    div[data-testid="stTextInput"] input, 
    div[data-testid="stNumberInput"] input,
    div[data-testid="stDateInput"] input {
        color: #000000 !important; /* Text in input fields should be black for contrast with #F2F2F2 background */
        background-color: #FFFFFF !important; /* White background for input fields for clarity */
        border: 1px solid #CCCCCC !important; /* Light gray border for inputs */
    }

    div[data-testid="stDataFrame"] table th {
        color: #000000 !important; /* Black text for table headers */
        background-color: #D0C8B0 !important; /* Keeping the darker beige for table headers */
    }
    div[data-testid="stDataFrame"] table td {
        color: #000000 !important; /* Black text for table cells (on #F2F2F2 main block background) */
    }

    div[data-testid="stExpander"] details summary div[role="button"] p {
         color: #EAE4D5 !important; /* Beige/off-white for expander summaries */
    }

    a {
        color: #EAE4D5 !important; /* Beige/off-white for links */
        text-decoration: underline !important; 
    }
    a:hover {
        color: #FFFFFF !important; /* White on hover for links */
    }

    [data-testid="stAppViewBlockContainer"] h1,
    [data-testid="stAppViewBlockContainer"] h2,
    [data-testid="stAppViewBlockContainer"] h3 {
        color: #EAE4D5 !important; /* Beige/off-white for main titles */
    }
    
    /* Keep titles within the main block also black for consistency with other text in that block */
    .main .block-container h1,
    .main .block-container h2,
    .main .block-container h3,
    .main .block-container h4,
    .main .block-container h5,
    .main .block-container h6 {
        color: #000000 !important; /* Black for headers inside the light gray content block */
    }
    /* Ensure the main page title outside the block container remains beige */
     div[data-testid="stAppViewContainer"] > section > div > div:not([data-testid="stVerticalBlockBorderWrapper"]) > div > div > h1 {
        color: #EAE4D5 !important;
    }

    /* Style for the main content block text (black) */
    .main .block-container p, 
    .main .block-container li, 
    .main .block-container .stMarkdown p, /* Target p tags within markdown in main block */
    .main .block-container div[data-testid="stText"], /* Target st.text, st.write if they render as simple text divs */
    .main .block-container label[data-testid="stWidgetLabel"] p, /* Labels of widgets within main block */
    .main .block-container div[data-testid="stRadio"] label span, /* Radio button text within main block */
    .main .block-container div[data-testid="stSelectbox"] li span, /* Selectbox text within main block */
    .main .block-container div[data-testid="stExpander"] details summary div[role="button"] p /* Expander titles within main block */
     {
        color: #000000 !important;
    }

    /* Custom styles for the square blocks */
    .summary-block {
        background-color: #D0C8B0; /* Light beige/tan */
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px; /* Space between blocks */
        text-align: center;
        flex: 1; /* Allows blocks to take equal width in a flex container */
        min-width: 150px; /* Ensures a minimum square-like appearance */
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        height: 120px; /* Fixed height to enforce square-like shape */
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .summary-block p {
        color: #333333 !important; /* Darker text for readability on light background */
        font-weight: bold;
        margin: 0;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Initialize Session State ---
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'InitialPage'
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'uploaded_df' not in st.session_state:
    st.session_state.uploaded_df = None

# --- Load Models (Cached) ---
@st.cache_resource
def load_models():
    try:
        with open('random_forest_regressor.pkl', 'rb') as f:
            reg_model = pickle.load(f)
        with open('review_classifier.pkl', 'rb') as f:
            clf_model = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        return reg_model, clf_model, label_encoder
    except FileNotFoundError as e:
        st.error(f"FATAL: Model file not found: {e}. Application cannot proceed without models.")
        st.stop()
    except Exception as e:
        st.error(f"FATAL: Error loading models: {e}. Application cannot proceed.")
        st.stop()

reg_model, clf_model, label_encoder = load_models()


# --- Helper function for "Back to Main Menu" button ---
def back_to_main_menu_button():
    if st.button("⬅️ Back to Main Menu / Upload New Data"):
        st.session_state.current_page = 'InitialPage'
        st.rerun()

# --- INITIAL PAGE Definition ---
def show_initial_page():
    st.title("📊 Amazon Insights Hub") 
    st.markdown("---")

    st.subheader("🚀 Choose a Dashboard to Explore")
    st.write("") 

    def handle_button_click(config): # Keep this helper function as is
        page_to_load = config["page"]
        if config.get("check_models"):
            if not reg_model or not clf_model or not label_encoder:
                st.error(f"Models required for {page_to_load.replace('_', ' ')} are not loaded. Cannot navigate.")
                return
        if config.get("check_data"):
            if st.session_state.uploaded_df is None:
                st.warning(f"⚠️ Please upload a dataset first to access {page_to_load.replace('_', ' ')}.", icon="📄")
                return
        st.session_state.current_page = page_to_load
        st.rerun()

    # UPDATED button_configs (Seller Quality removed)
    button_configs = [
        {"label": "🔍 Product Viability", "page": "Product Viability", "check_models": True},
        {"label": "💹 Market Intelligence", "page": "Market Intelligence", "check_data": True},
        {"label": "🚨 Risk Management", "page": "Risk Management", "check_data": True},
        {"label": "📈 EDA Dashboard", "page": "EDA Dashboard", "check_data": True},
        {"label": "🗣️ Customer Voice", "page": "Customer Voice Dashboard", "check_data": True},
        {"label": "💡 Product Intelligence", "page": "Product Intelligence", "check_data": True},
        # {"label": "🏷️ Seller Quality", "page": "Seller Quality Dashboard", "check_data": True}, # REMOVED/COMMENTED OUT
    ]

    # Row 1 of Buttons (3 buttons)
    cols_r1 = st.columns(3)
    for i in range(3):
        with cols_r1[i]:
            config = button_configs[i]
            button_key_r1 = f"btn_nav_{config['page'].replace(' ', '_').lower()}_r1"
            if st.button(config["label"], use_container_width=True, key=button_key_r1):
                handle_button_click(config)
    
    st.write("") 

    # Row 2 of Buttons (remaining 3 buttons)
    cols_r2 = st.columns(3) 
    # Iterate for buttons from index 3 up to the new length of button_configs (which is 6)
    for i in range(3, len(button_configs)): 
        with cols_r2[i-3]: # Adjust index for columns (0, 1, 2)
            config = button_configs[i]
            button_key_r2 = f"btn_nav_{config['page'].replace(' ', '_').lower()}_r2_{i-3}"
            if st.button(config["label"], use_container_width=True, key=button_key_r2):
                handle_button_click(config)

    # The cols_r3 block for the 7th button is now REMOVED as there are only 6 buttons
    
    st.markdown("<hr style='margin-top: 2.5em; margin-bottom: 1.5em; border-color: #667788;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: #EAE4D5 !important;'>📤 Upload Your Dataset</h3>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
                "Upload Amazon Review Dataset (CSV)", 
                type="csv", 
                key="main_uploader_final_no_sq", # New key
                label_visibility="collapsed"
            )

    # ... (rest of the file uploader logic remains the same) ...
    if uploaded_file:
        try:
            if st.session_state.uploaded_df is not None:
                current_filename = getattr(st.session_state.uploaded_df, '_uploaded_file_name', None)
                if current_filename and uploaded_file.name != current_filename: 
                    st.session_state.predictions = []
                    st.toast("New dataset uploaded. Predictions cleared.", icon="🧹")
            
            df_upload = pd.read_csv(uploaded_file)
            st.session_state.uploaded_df = df_upload
            setattr(st.session_state.uploaded_df, '_uploaded_file_name', uploaded_file.name) 
            
            st.success("✅ Dataset uploaded successfully!") 
            
            with st.expander("Quick Summary of Uploaded Data", expanded=False):
                # Using columns for the two square blocks
                col_rows, col_cols = st.columns(2)

                with col_rows:
                    st.markdown(f"""
                        <div class="summary-block">
                            <p>Total Rows</p>
                            <p style="font-size: 2em; color: #000000 !important;">{df_upload.shape[0]}</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col_cols:
                    st.markdown(f"""
                        <div class="summary-block">
                            <p>Total Columns</p>
                            <p style="font-size: 2em; color: #000000 !important;">{df_upload.shape[1]}</p>
                        </div>
                    """, unsafe_allow_html=True)

                st.markdown("<p style='color: #000000;'>First 5 rows:</p>", unsafe_allow_html=True) # Changed color to black for consistency
                st.dataframe(df_upload.head())
        except Exception as e:
            st.error(f"Error reading or processing CSV: {e}") 
            st.session_state.uploaded_df = None
            
    elif st.session_state.uploaded_df is not None:
        st.info("ℹ️ Using previously uploaded dataset. To use a new one, upload it above.") 
    else:
        st.markdown("<p style='text-align: center; margin-top: 1em;'>Please upload a dataset to enable most dashboards.</p>", unsafe_allow_html=True)
# --- PAGE ROUTING ---
# [Previous code remains the same]

# --- PAGE ROUTING ---
if st.session_state.current_page == 'InitialPage':
    show_initial_page()

elif st.session_state.current_page == 'Product Viability':
    st.title("🔍 Product Viability")
    back_to_main_menu_button()

    if not reg_model or not clf_model or not label_encoder:
        st.error("Predictive models are not available. Please check model files.")
        st.stop()

    sub_page = st.radio("Choose Product Viability Option", ['Prediction', 'Analysis'])

    if sub_page == 'Prediction':
        st.subheader("🔮 Predict Review Score & Sentiment")
        st.write("Enter product review information:")

        helpfulness = st.slider("Helpfulness Ratio (0 to 1)", 0.0, 1.0, 0.3)
        length = st.number_input("Review Length (word count)", value=100, min_value=1)
        sentiment = st.slider("Sentiment Score (-1 to 1)", -1.0, 1.0, 0.4)
        review_year = st.number_input("Review Year", value=datetime.datetime.now().year, min_value=2000, max_value=2050)
        review_month = st.number_input("Review Month", min_value=1, max_value=12, value=datetime.datetime.now().month)

        if st.button("Predict"):
            input_df_reg = pd.DataFrame([{
                'HelpfulnessRatio': helpfulness,
                'ReviewLength': length,
                'SentimentScore': sentiment,
                'ReviewYear': review_year,
                'ReviewMonth': review_month
            }])

            predicted_score = reg_model.predict(input_df_reg)[0]
            st.success(f"⭐ Predicted Rating: **{predicted_score:.2f}**")

            st.session_state.predictions.append({
                "Rating": predicted_score,
                "Helpfulness": helpfulness,
                "Length": length,
                "InputSentiment": sentiment,
                "Year": review_year,
                "Month": review_month,
                "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

        if st.session_state.predictions:
            preds_df = pd.DataFrame(st.session_state.predictions)
            st.write("### 📋 All Predictions So Far")
            st.dataframe(preds_df)

            successful_reviews = preds_df[preds_df['Rating'] >= 4.0]
            total_reviews = len(preds_df)
            if total_reviews > 0:
                success_rate = (len(successful_reviews) / total_reviews) * 100
                st.markdown(f"✅ **Product Success Rate (based on predicted rating >= 4.0):** `{success_rate:.2f}%` from {total_reviews} prediction(s).")

    elif sub_page == 'Analysis':
        st.subheader("📊 Product Review Analysis (Based on Predictions Made)")
        preds_df = pd.DataFrame(st.session_state.predictions)

        if preds_df.empty:
            st.warning("No predictions yet. Use the 'Prediction' tab to add entries.")
        else:
            st.write("Recent Predictions:")
            st.dataframe(preds_df.tail())

            st.write("### Predicted Rating Distribution")
            fig_rating_hist, ax_rating_hist = plt.subplots()
            ax_rating_hist.hist(preds_df['Rating'], bins=5, range=(1,5), color='skyblue', edgecolor='black')
            ax_rating_hist.set_xlabel("Predicted Rating")
            ax_rating_hist.set_ylabel("Count")
            fig_rating_hist.patch.set_alpha(0)
            ax_rating_hist.set_facecolor('none')
            ax_rating_hist.tick_params(colors='black', which='both')
            ax_rating_hist.xaxis.label.set_color('black')
            ax_rating_hist.yaxis.label.set_color('black')
            ax_rating_hist.title.set_color('black')
            st.pyplot(fig_rating_hist)


elif st.session_state.current_page == 'EDA Dashboard':
    st.title("📈 EDA Dashboard - Amazon Reviews")
    back_to_main_menu_button()

    if st.session_state.uploaded_df is None:
        st.warning("⚠️ Please upload a dataset from the 'Main Menu' page first.")
        st.stop()
    else:
        df = st.session_state.uploaded_df.copy()

        required_eda_cols = ['Text', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Time', 'Score']
        # Optional columns for more EDA metrics if available
        optional_eda_cols = ['ProductId', 'UserId']
        
        if not all(col in df.columns for col in required_eda_cols):
            missing_cols = [col for col in required_eda_cols if col not in df.columns]
            st.error(f"Dataset is missing one or more required columns for EDA: {', '.join(missing_cols)}")
            st.stop()

        df['ReviewLength'] = df['Text'].apply(lambda x: len(str(x).split()))
        df['HelpfulnessRatio'] = df['HelpfulnessNumerator'] / (df['HelpfulnessDenominator'] + 1)
        try:
            df['ReviewTime'] = pd.to_datetime(df['Time'], unit='s', errors='coerce')
            df.dropna(subset=['ReviewTime'], inplace=True)
        except Exception as e:
            st.error(f"Error converting 'Time' column to datetime: {e}. Please ensure it's a Unix timestamp.")
            st.stop()

        df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
        df.dropna(subset=['Score'], inplace=True)

        if df.empty:
            st.warning("No data available for EDA after initial processing and cleaning.")
            st.stop()

        # --- 0. Key EDA Insights Summary ---
        st.subheader("📊 Key EDA Insights Summary")

        total_reviews_eda = len(df)
        review_period_start = df['ReviewTime'].min().strftime('%b %Y')
        review_period_end = df['ReviewTime'].max().strftime('%b %Y')
        avg_score_eda = df['Score'].mean()
        
        score_counts_eda = df['Score'].value_counts()
        most_common_score_eda = score_counts_eda.idxmax() if not score_counts_eda.empty else "N/A"
        
        avg_review_length_eda = df['ReviewLength'].mean()
        avg_helpfulness_eda = df['HelpfulnessRatio'].mean()

        high_scores_count = df[df['Score'] >= 4].shape[0]
        low_scores_count = df[df['Score'] <= 2].shape[0]
        percent_positive_reviews = (high_scores_count / total_reviews_eda * 100) if total_reviews_eda > 0 else 0
        percent_negative_reviews = (low_scores_count / total_reviews_eda * 100) if total_reviews_eda > 0 else 0
        
        num_unique_products_eda = df['ProductId'].nunique() if 'ProductId' in df.columns else "N/A"
        num_unique_users_eda = df['UserId'].nunique() if 'UserId' in df.columns else "N/A"


        col_eda1, col_eda2, col_eda3 = st.columns(3)
        with col_eda1:
            st.metric(label="Total Reviews Analyzed", value=f"{total_reviews_eda:,}")
            if num_unique_products_eda != "N/A":
                st.metric(label="Unique Products", value=f"{num_unique_products_eda:,}")
        with col_eda2:
            st.metric(label="Avg. Score", value=f"{avg_score_eda:.2f} ★")
            if num_unique_users_eda != "N/A":
                st.metric(label="Unique Reviewers", value=f"{num_unique_users_eda:,}")
        with col_eda3:
            st.metric(label="Avg. Review Length", value=f"{avg_review_length_eda:.0f} words")
            st.metric(label="Review Period", value=f"{review_period_start} - {review_period_end}")
        
        st.markdown("---")
        st.markdown("#### Rating Overview:")
        col_eda_ratings1, col_eda_ratings2, col_eda_ratings3 = st.columns(3)
        with col_eda_ratings1:
             st.markdown(f"⭐ **Most Common Rating:** {most_common_score_eda} stars")
        with col_eda_ratings2:
            st.markdown(f"👍 **Positive Reviews (4-5★):** {percent_positive_reviews:.1f}%")
        with col_eda_ratings3:
            st.markdown(f"👎 **Negative Reviews (1-2★):** {percent_negative_reviews:.1f}%")
        
        st.markdown(f"📊 **Overall Avg. Helpfulness:** {avg_helpfulness_eda:.2f}")
        st.markdown("---")


        # --- 1. Ratings Distribution (Chart 1) ---
        st.subheader("1. ⭐ Ratings Distribution")
        # score_counts is already calculated as score_counts_eda for the summary, we can reuse it or the df directly
        if not df.empty and 'Score' in df.columns and not df['Score'].dropna().empty: # ensure Score has non-NA values
            score_counts_for_chart = df['Score'].value_counts().sort_index()
            chart_data = pd.DataFrame({'Score': score_counts_for_chart.index, 'Count': score_counts_for_chart.values})
            rating_dist_chart = alt.Chart(chart_data).mark_bar().encode(
                x=alt.X('Score:O', title='Rating Score'),
                y=alt.Y('Count:Q', title='Number of Reviews'),
                tooltip=['Score', 'Count']
            ).properties(
                title='Ratings Distribution'
            ).configure_axis(
                labelColor='white',
                titleColor='white'
            ).configure_title(
                color='white'
            ).configure_legend(
                labelColor='white',
                titleColor='white'
            )
            st.altair_chart(rating_dist_chart, use_container_width=True)

            with st.expander("📘 What this chart shows"):
                # These insights are now largely covered by the summary metrics above
                # We can keep more detailed textual interpretation if needed or simplify
                total_reviews_val_chart = score_counts_for_chart.sum() # Use chart specific counts
                most_common_score_chart = score_counts_for_chart.idxmax()
                most_common_count_chart = score_counts_for_chart.max()
                
                low_scores_chart = score_counts_for_chart[score_counts_for_chart.index <= 2].sum()
                high_scores_chart = score_counts_for_chart[score_counts_for_chart.index >= 4].sum()

                if total_reviews_val_chart > 0:
                    low_pct_chart = (low_scores_chart / total_reviews_val_chart) * 100
                    high_pct_chart = (high_scores_chart / total_reviews_val_chart) * 100
                    st.markdown(f"""
                    - The most common rating is **{most_common_score_chart} stars**, appearing in **{most_common_count_chart:,} reviews**.
                    - **{high_pct_chart:.1f}%** of reviews are positive (4 or 5 stars).
                    - **{low_pct_chart:.1f}%** of reviews are negative (1 or 2 stars).
                    - This indicates that the general sentiment is **{'positive' if high_pct_chart > 60 else 'mixed' if low_pct_chart > 30 else 'neutral'}**.
                    """)
                else:
                    st.write("Not enough data for rating insights in this chart's scope.")
        else:
            st.write("No score data to display or 'Score' column is empty after processing.")

        # --- 2. Review Length Distribution (Chart 2) ---
        st.subheader("2. 📝 Review Length Distribution")
        if not df.empty and 'ReviewLength' in df.columns and not df['ReviewLength'].dropna().empty:
            review_length_chart = alt.Chart(df).mark_bar().encode(
                alt.X('ReviewLength', bin=alt.Bin(maxbins=50), title='Review Length (words)'),
                y='count()',
                tooltip=[alt.Tooltip('ReviewLength', type='quantitative', format=',d', title='Length (words)'), alt.Tooltip('count()', title='Number of Reviews')]
            ).properties(
                title="Review Length Histogram"
            ).configure_axis(
                labelColor='white',
                titleColor='white'
            ).configure_title(
                color='white'
            )
            st.altair_chart(review_length_chart, use_container_width=True)
            with st.expander("📘 What this chart shows"):
                # avg_length, median_length already calculated in summary
                short_reviews = df[df['ReviewLength'] < 20].shape[0]
                long_reviews = df[df['ReviewLength'] > 100].shape[0]

                st.markdown(f"""
                - The **average review length** is about **{avg_review_length_eda:.0f} words** (Median: {df['ReviewLength'].median():.0f} words).
                - There are **{short_reviews:,} short reviews (< 20 words)**, and **{long_reviews:,} long reviews (> 100 words)**.
                - Longer reviews may indicate thoughtful feedback, while short ones could be spammy or casual.
                """)
        else:
            st.write("No review length data to display.")

        # --- 3. Helpfulness Ratio Distribution (Chart 3) ---
        st.subheader("3. 👍 Helpfulness Ratio Distribution")
        if not df.empty and 'HelpfulnessRatio' in df.columns and not df['HelpfulnessRatio'].dropna().empty :
            helpfulness_chart = alt.Chart(df).mark_bar().encode(
                alt.X('HelpfulnessRatio', bin=alt.Bin(maxbins=40), title='Helpfulness Ratio', axis=alt.Axis(format=".2f")), # Added format to axis
                y='count()',
                tooltip=[alt.Tooltip('HelpfulnessRatio', type='quantitative', format='.2f', title='Helpfulness Ratio'), alt.Tooltip('count()', title='Number of Reviews')]
            ).properties(
                title="Helpfulness Ratio Histogram"
            ).configure_axis(
                labelColor='white',
                titleColor='white'
            ).configure_title(
                color='white'
            )
            st.altair_chart(helpfulness_chart, use_container_width=True)
            with st.expander("📘 What this chart shows"):
                # avg_help already calculated in summary
                high_helpful = df[df['HelpfulnessRatio'] > 0.75].shape[0]
                low_helpful = df[df['HelpfulnessRatio'] < 0.25].shape[0]

                st.markdown(f"""
                - The **average helpfulness ratio** is **{avg_helpfulness_eda:.2f}**.
                - **{high_helpful:,} reviews** have a helpfulness ratio > 0.75 (highly useful).
                - **{low_helpful:,} reviews** have a ratio < 0.25 (likely unhelpful).
                - Amazon can surface highly helpful reviews more prominently.
                """)
        else:
            st.write("No helpfulness ratio data to display.")

        # --- 4. Reviews Over Time (Chart 4) ---
        st.subheader("4. 🕒 Reviews Over Time")
        if not df.empty and 'ReviewTime' in df.columns and pd.api.types.is_datetime64_any_dtype(df['ReviewTime']) and not df['ReviewTime'].dropna().empty:
            reviews_by_month_series = df.set_index('ReviewTime').resample('M').size()
            reviews_by_month_df = reviews_by_month_series.reset_index()
            reviews_by_month_df.columns = ['Month', 'ReviewCount']
            reviews_by_month_df['Month'] = pd.to_datetime(reviews_by_month_df['Month']) # Ensure it's datetime for Altair

            time_chart = alt.Chart(reviews_by_month_df).mark_line(point=True).encode(
                x=alt.X('Month:T', title='Month', axis=alt.Axis(format='%Y-%m')),
                y=alt.Y('ReviewCount:Q', title='Number of Reviews'),
                tooltip=[alt.Tooltip('Month:T', title='Month', format='%B %Y'), alt.Tooltip('ReviewCount:Q', title='Reviews')]
            ).properties(
                title="Number of Reviews per Month"
            ).configure_axis(
                labelColor='white',
                titleColor='white'
            ).configure_title(
                color='white'
            )
            st.altair_chart(time_chart, use_container_width=True)

            with st.expander("📘 What this chart shows"):
                if not reviews_by_month_series.empty:
                    # review_period_start, review_period_end already in summary
                    peak_month_ts = reviews_by_month_series.idxmax()
                    peak_month_str = peak_month_ts.strftime('%B %Y')
                    peak_count = reviews_by_month_series.max()

                    st.markdown(f"""
                    - Review data spans from **{review_period_start}** to **{review_period_end}**.
                    - The highest number of reviews was in **{peak_month_str}**, with **{peak_count:,} reviews**.
                    - Trends over time help identify **product launches, seasonality**, or **spikes due to issues or promotions**.
                    """)
                else:
                    st.write("Not enough monthly data to derive insights.")
        else:
            st.write("No review time data to display or 'ReviewTime' is not in datetime format.")

        # --- 5. Review Length by Rating (Box Plot - Chart 5) ---
        st.subheader("5. 📦 Review Length by Rating (Box Plot)")
        if not df.empty and 'Score' in df.columns and 'ReviewLength' in df.columns and not df['Score'].dropna().empty and not df['ReviewLength'].dropna().empty:
            box_data = df[['Score', 'ReviewLength']].copy()
            box_data['Score'] = box_data['Score'].astype('category')

            box_plot = alt.Chart(box_data).mark_boxplot(extent='min-max').encode(
                x=alt.X('Score:N', title="Rating Score"), # Use :N for nominal (categorical)
                y=alt.Y('ReviewLength:Q', title='Review Length (words)', scale=alt.Scale(zero=False)),
                color='Score:N',
                tooltip=[alt.Tooltip('Score:N', title='Rating'), alt.Tooltip('ReviewLength:Q', title='Length (median)')] # Boxplot tooltip shows median by default
            ).properties(
                title="Review Length by Rating"
            ).configure_axis(
                labelColor='white',
                titleColor='white'
            ).configure_title(
                color='white'
            ).configure_legend(
                labelColor='white',
                titleColor='white'
            )
            st.altair_chart(box_plot, use_container_width=True)
            with st.expander("📘 What this chart shows"):
                avg_lengths_df = df.copy()
                avg_lengths_df['Score'] = pd.to_numeric(avg_lengths_df['Score'], errors='coerce')
                avg_lengths_df.dropna(subset=['Score'], inplace=True)
                if not avg_lengths_df.empty:
                    avg_lengths = avg_lengths_df.groupby('Score')['ReviewLength'].mean().round(1)
                    median_lengths_by_score = avg_lengths_df.groupby('Score')['ReviewLength'].median().round(1)
                    if not avg_lengths.empty:
                        longest_score_avg = avg_lengths.idxmax()
                        longest_avg_val = avg_lengths.max()
                        shortest_score_avg = avg_lengths.idxmin()
                        shortest_avg_val = avg_lengths.min()

                        st.markdown(f"""
                        - On average, **{longest_score_avg}-star reviews** are the longest at **{longest_avg_val:.1f} words**.
                        - **{shortest_score_avg}-star reviews** are the shortest, with just **{shortest_avg_val:.1f} words** on average.
                        - The box plot also shows the median, quartiles, and potential outliers for review length at each rating score.
                        - This can suggest if customers write more detailed feedback when leaving specific types of scores.
                        """)
                        st.write("Median lengths by score:", median_lengths_by_score)
                    else:
                        st.write("Not enough data for review length by rating insights after grouping.")
                else:
                    st.write("Not enough data for review length by rating insights.")
        else:
            st.write("No score or review length data for box plot.")

        # --- 6. Average Helpfulness Ratio by Rating (Pie Chart - Chart 6) ---
        st.subheader("6. 🥧 Average Helpfulness Ratio by Rating")
        if not df.empty and 'Score' in df.columns and 'HelpfulnessRatio' in df.columns and not df['Score'].dropna().empty and not df['HelpfulnessRatio'].dropna().empty:
            helpful_by_score_df = df.copy()
            helpful_by_score_df['Score'] = pd.to_numeric(helpful_by_score_df['Score'], errors='coerce').dropna()

            if not helpful_by_score_df.empty:
                helpful_by_score = helpful_by_score_df.groupby('Score')['HelpfulnessRatio'].mean().reset_index()
                helpful_by_score['Score'] = helpful_by_score['Score'].astype(str) # For discrete categories in pie

                if not helpful_by_score.empty and helpful_by_score['HelpfulnessRatio'].sum() > 0 :
                    fig_pie_help = px.pie(helpful_by_score,
                                        values='HelpfulnessRatio',
                                        names='Score',
                                        title='Average Helpfulness Ratio by Rating Score',
                                        hole=0.3, # Donut chart
                                        color_discrete_sequence=px.colors.qualitative.Pastel)
                    fig_pie_help.update_traces(textposition='inside', textinfo='percent+label+value')
                    fig_pie_help.update_layout(
                        title_font_color='white',
                        legend_title_font_color='white',
                        legend_font_color='white',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_pie_help, use_container_width=True)
                    with st.expander("📘 What this chart shows"):
                        max_row = helpful_by_score.loc[helpful_by_score['HelpfulnessRatio'].idxmax()]
                        min_row = helpful_by_score.loc[helpful_by_score['HelpfulnessRatio'].idxmin()]

                        st.markdown(f"""
                        - This chart shows the average helpfulness ratio for each star rating.
                        - The **{max_row['Score']}-star reviews** have the highest average helpfulness ratio of **{max_row['HelpfulnessRatio']:.2f}**.
                        - The **{min_row['Score']}-star reviews** have the lowest average helpfulness ratio of **{min_row['HelpfulnessRatio']:.2f}**.
                        - This helps visualize which ratings' average helpfulness are most prominent or if certain rated reviews tend to be more helpful than others.
                        """)
                else:
                    st.write("Not enough data or zero total helpfulness ratio for pie chart.")
            else:
                st.write("No valid score data after processing for helpfulness by rating.")
        else:
            st.write("No score or helpfulness ratio data for pie chart.")

elif st.session_state.current_page == 'Customer Voice Dashboard':
    st.title("🗣️ Customer Voice Dashboard")
    back_to_main_menu_button()

    if st.session_state.uploaded_df is None:
        st.warning("⚠️ Please upload a dataset from the 'Main Menu' page first.")
        st.stop()
    else:
        df = st.session_state.uploaded_df.copy()
        required_cv_cols = ['Text', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Score']
        if not all(col in df.columns for col in required_cv_cols):
            missing_cols = [col for col in required_cv_cols if col not in df.columns]
            st.error(f"Dataset is missing one or more required columns for Customer Voice: {', '.join(missing_cols)}")
            st.stop()

        df['Text'] = df['Text'].astype(str)
        df['SentimentScore'] = df['Text'].apply(lambda x: TextBlob(x).sentiment.polarity if pd.notnull(x) else 0)
        df['ReviewLength'] = df['Text'].apply(lambda x: len(x.split()))
        df['HelpfulnessRatio'] = df['HelpfulnessNumerator'] / (df['HelpfulnessDenominator'] + 1)
        df['Score'] = pd.to_numeric(df['Score'], errors='coerce') # Ensure Score is numeric for context

        if df.empty:
            st.warning("No data available for Customer Voice analysis after initial processing.")
            st.stop()

        # --- 0. Key Customer Voice Insights Summary ---
        st.subheader("🗣️ Key Customer Voice Insights Summary")

        total_reviews_cv = len(df)
        avg_sentiment_cv = df['SentimentScore'].mean()
        
        positive_reviews_cv = df[df['SentimentScore'] > 0.05]
        negative_reviews_cv = df[df['SentimentScore'] < -0.05]
        neutral_reviews_cv = df[(df['SentimentScore'] >= -0.05) & (df['SentimentScore'] <= 0.05)]

        count_positive_cv = len(positive_reviews_cv)
        count_negative_cv = len(negative_reviews_cv)
        count_neutral_cv = len(neutral_reviews_cv)

        percent_positive_cv = (count_positive_cv / total_reviews_cv * 100) if total_reviews_cv > 0 else 0
        percent_negative_cv = (count_negative_cv / total_reviews_cv * 100) if total_reviews_cv > 0 else 0
        percent_neutral_cv = (count_neutral_cv / total_reviews_cv * 100) if total_reviews_cv > 0 else 0
        
        avg_review_length_cv = df['ReviewLength'].mean()
        avg_helpfulness_cv = df['HelpfulnessRatio'].mean()

        strong_positive_threshold = 0.6
        strong_negative_threshold = -0.6
        count_strong_positive = df[df['SentimentScore'] > strong_positive_threshold].shape[0]
        count_strong_negative = df[df['SentimentScore'] < strong_negative_threshold].shape[0]

        # Correlations (handle potential all-NaN columns or constant values which result in NaN correlation)
        corr_sentiment_length = np.nan
        if df['SentimentScore'].nunique() > 1 and df['ReviewLength'].nunique() > 1: # Check for variance
             corr_sentiment_length = df['SentimentScore'].corr(df['ReviewLength'])
        
        corr_sentiment_helpfulness = np.nan
        if df['SentimentScore'].nunique() > 1 and df['HelpfulnessRatio'].nunique() > 1:
            corr_sentiment_helpfulness = df['SentimentScore'].corr(df['HelpfulnessRatio'])


        col_cv1, col_cv2, col_cv3 = st.columns(3)
        with col_cv1:
            st.metric(label="Total Reviews Analyzed", value=f"{total_reviews_cv:,}")
            st.metric(label="Avg. Sentiment Score", value=f"{avg_sentiment_cv:.2f}")
        with col_cv2:
            st.metric(label="Positive Reviews (>0.05)", value=f"{percent_positive_cv:.1f}% ({count_positive_cv:,})")
            st.metric(label="Negative Reviews (<-0.05)", value=f"{percent_negative_cv:.1f}% ({count_negative_cv:,})")
        with col_cv3:
            st.metric(label="Neutral Reviews", value=f"{percent_neutral_cv:.1f}% ({count_neutral_cv:,})")
            st.metric(label="Avg. Review Length", value=f"{avg_review_length_cv:.0f} words")

        st.markdown("---")
        st.markdown("#### Deeper Sentiment Insights:")
        col_ds1, col_ds2, col_ds3 = st.columns(3)
        with col_ds1:
            st.markdown(f"😃 **Strongly Positive (>{strong_positive_threshold}):** {count_strong_positive:,} reviews")
        with col_ds2:
            st.markdown(f"😠 **Strongly Negative (<{strong_negative_threshold}):** {count_strong_negative:,} reviews")
        with col_ds3:
             st.markdown(f"👍 **Avg. Helpfulness Ratio:** {avg_helpfulness_cv:.2f}")


        st.markdown("#### Correlations with Sentiment:")
        col_corr1, col_corr2 = st.columns(2)
        with col_corr1:
            st.markdown(f"🆚 **Sentiment vs. Length:** {corr_sentiment_length:.2f}" if not pd.isna(corr_sentiment_length) else "🆚 **Sentiment vs. Length:** N/A (insufficient variance)")
        with col_corr2:
            st.markdown(f"🆚 **Sentiment vs. Helpfulness:** {corr_sentiment_helpfulness:.2f}" if not pd.isna(corr_sentiment_helpfulness) else "🆚 **Sentiment vs. Helpfulness:** N/A (insufficient variance)")
        st.markdown("*(Correlation ranges from -1 to +1. Values near 0 suggest weak linear relationship.)*")
        st.markdown("---")

        # --- 1. Sentiment Score Distribution (Chart 1) ---
        st.subheader("1. 💬 Sentiment Score Distribution")
        # The expander for this chart already contains avg_sent, pos_pct, neg_pct, neutral_pct
        # So, the summary above provides a quick glance, and this chart provides the visual.
        if not df.empty and 'SentimentScore' in df.columns and not df['SentimentScore'].dropna().empty:
            sentiment_hist = alt.Chart(df).mark_bar().encode(
                alt.X('SentimentScore', bin=alt.Bin(maxbins=30), title='Sentiment Score', axis=alt.Axis(format=".2f")),
                y='count()',
                tooltip=[alt.Tooltip('SentimentScore', type='quantitative', format='.2f'), 'count()']
            ).properties(
                title="Sentiment Score Distribution"
            ).configure_axis(
                labelColor='black',
                titleColor='black'
            ).configure_title(
                color='black'
            )
            st.altair_chart(sentiment_hist, use_container_width=True)

            with st.expander("📘 Insights on Sentiment Scores"):
                # avg_sent, total_reviews_cv, pos_pct, neg_pct, neutral_pct already calculated for summary
                st.markdown(f"""
                - The **average sentiment score** is **{avg_sentiment_cv:.2f}**. (Ranges from -1 (very negative) to +1 (very positive)).
                - **{percent_positive_cv:.1f}%** of reviews are classified as positive (score > 0.05).
                - **{percent_negative_cv:.1f}%** are classified as negative (score < -0.05).
                - Neutral reviews account for approximately **{percent_neutral_cv:.1f}%**.
                """)
        else:
            st.write("No sentiment score data to display.")

        # --- 2. Frequently Used Words (Word Cloud - Chart 2) ---
        st.subheader("2. ☁️ Frequently Used Words (Overall)")
        if not df.empty and 'Text' in df.columns and not df['Text'].dropna().empty:
            all_text = ' '.join(df['Text'].astype(str).dropna())
            if all_text.strip():
                try:
                    custom_stopwords = set(WordCloud().stopwords)
                    custom_stopwords.update(['product', 'item', 'review', 'amazon', 'order', 'get', 'got', 'would', 'make', 'use', 'also', 'one', 'like', 'thi', 'veri', 'good', 'great', 'time', 'I', 'it', 's', 't', 'm', 've', 're', 'price', 'day', 'week', 'month', 'year', 'really', 'even', 'bought', 'recommend', 'will', 'well', 'much', 'buy', 'need', 'work', 'just', 'ca', 'na', 'wa', 'ha', 'doe'])

                    wordcloud = WordCloud(width=800, height=300, background_color='white', stopwords=custom_stopwords, collocations=True, max_words=100, colormap='cividis').generate(all_text)
                    fig_wc, ax_wc = plt.subplots(figsize=(10, 4))
                    ax_wc.imshow(wordcloud, interpolation='bilinear')
                    ax_wc.axis('off')
                    fig_wc.patch.set_alpha(0)
                    st.pyplot(fig_wc)
                    with st.expander("📘 Insights on Word Cloud"):
                        st.markdown("""
                        - The largest words (and common phrases if `collocations=True`) are the most frequently used in reviews overall.
                        - This can help spot recurring **product features, general discussion topics, or common adjectives** used by customers.
                        - Useful for understanding the core vocabulary customers use when talking about the products.
                        """)
                except Exception as e:
                    st.error(f"Could not generate word cloud: {e}")
            else:
                st.write("No text data available for word cloud after processing.")
        else:
            st.write("No text data to display for word cloud.")

        # --- 3. Review Length vs. Sentiment (Scatter Plot - Chart 3) ---
        st.subheader("3. 📝 Review Length vs. Sentiment")
        if not df.empty and 'ReviewLength' in df.columns and 'SentimentScore' in df.columns and not df['ReviewLength'].dropna().empty and not df['SentimentScore'].dropna().empty:
            df_display = df.copy()
            df_display['TooltipText'] = df_display['Text'].astype(str).str.slice(0, 100) + '...'

            # Determine a dynamic size for circles based on helpfulness, if desired, or keep fixed.
            # For simplicity, keeping fixed size for now.
            scatter_chart = alt.Chart(df_display.sample(n=min(1000, len(df_display)), random_state=1) if len(df_display) > 1000 else df_display).mark_circle(size=60, opacity=0.6).encode(
                x=alt.X('ReviewLength:Q', scale=alt.Scale(type="log", zero=False), title='Review Length (words, Log Scale)'), # Log scale for length often better
                y=alt.Y('SentimentScore:Q', title='Sentiment Score', axis=alt.Axis(format=".2f")),
                color=alt.Color('Score:N', title='Actual Rating', legend=alt.Legend(orient="top")),
                tooltip=[alt.Tooltip('ReviewLength:Q', title='Length'), 
                         alt.Tooltip('SentimentScore:Q', title='Sentiment', format=".2f"), 
                         'Score:N', 
                         alt.Tooltip('TooltipText:N', title='Review Start')]
            ).properties(
                title="Review Length vs Sentiment (Sampled if >1000 points)", width=700
            ).interactive().configure_axis(
                labelColor='black',
                titleColor='black'
            ).configure_title(
                color='black'
            ).configure_legend(
                labelColor='black',
                titleColor='black'
            )
            st.altair_chart(scatter_chart, use_container_width=True)
            with st.expander("📘 Insights on Review Length vs Sentiment"):
                if not df.empty and 'ReviewLength' in df.columns and not df['ReviewLength'].dropna().empty:
                    longest_review_row = df.loc[df['ReviewLength'].idxmax()]
                    shortest_review_row = df.loc[df['ReviewLength'].idxmin()]
                    st.markdown(f"""
                    - **Correlation (Sentiment vs. Length):** {corr_sentiment_length:.2f} (if shown in summary above).
                    - Longest review is **{longest_review_row['ReviewLength']} words** (Sentiment: {longest_review_row['SentimentScore']:.2f}, Rating: {longest_review_row['Score']}★).
                    - Shortest review is **{shortest_review_row['ReviewLength']} words** (Sentiment: {shortest_review_row['SentimentScore']:.2f}, Rating: {shortest_review_row['Score']}★).
                    - Observe if longer reviews tend to have more **polarized** sentiment (either very positive or very negative), or if they cluster around certain rating scores. A log scale for review length helps visualize a wide range of lengths.
                    """)
                else:
                    st.write("No data for review length vs sentiment insights.")
        else:
            st.write("No review length or sentiment score data for scatter plot.")

        # --- 4. Helpfulness Ratio vs. Sentiment (Scatter Plot - Chart 4) ---
        st.subheader("4. 👍 Helpfulness Ratio vs. Sentiment")
        if not df.empty and 'HelpfulnessRatio' in df.columns and 'SentimentScore' in df.columns and not df['HelpfulnessRatio'].dropna().empty and not df['SentimentScore'].dropna().empty:
            df_display_helpful = df.copy()
            if 'TooltipText' not in df_display_helpful.columns: # Should exist from previous chart if df is same
                df_display_helpful['TooltipText'] = df_display_helpful['Text'].astype(str).str.slice(0, 100) + '...'

            helpful_chart = alt.Chart(df_display_helpful.sample(n=min(1000, len(df_display_helpful)), random_state=1) if len(df_display_helpful) > 1000 else df_display_helpful).mark_circle(size=60, opacity=0.6).encode(
                x=alt.X('HelpfulnessRatio:Q', scale=alt.Scale(zero=False), title='Helpfulness Ratio', axis=alt.Axis(format=".2f")),
                y=alt.Y('SentimentScore:Q', title='Sentiment Score', axis=alt.Axis(format=".2f")),
                color=alt.Color('Score:N', title='Actual Rating', legend=alt.Legend(orient="top")),
                tooltip=[alt.Tooltip('HelpfulnessRatio:Q', title='Helpfulness', format=".2f"), 
                         alt.Tooltip('SentimentScore:Q', title='Sentiment', format=".2f"), 
                         'Score:N', 
                         alt.Tooltip('TooltipText:N', title='Review Start')]
            ).properties(
                title="Helpfulness vs Sentiment (Sampled if >1000 points)", width=700
            ).interactive().configure_axis(
                labelColor='black',
                titleColor='black'
            ).configure_title(
                color='black'
            ).configure_legend(
                labelColor='black',
                titleColor='black'
            )
            st.altair_chart(helpful_chart, use_container_width=True)
            with st.expander("📘 Insights on Helpfulness vs Sentiment"):
                if not df.empty and 'HelpfulnessRatio' in df.columns and not df['HelpfulnessRatio'].dropna().empty:
                    high_help_df = df[df['HelpfulnessRatio'] > 0.75]
                    # low_help_df = df[df['HelpfulnessRatio'] < 0.25] # Not used in this text now
                    high_help_count_cv = high_help_df.shape[0]
                    # low_help_count_cv = low_help_df.shape[0]
                    avg_sent_high_help_cv = high_help_df['SentimentScore'].mean() if high_help_count_cv > 0 else float('nan')
                    # avg_sent_low_help_cv = low_help_df['SentimentScore'].mean() if low_help_count_cv > 0 else float('nan')
                    st.markdown(f"""
                    - **Correlation (Sentiment vs. Helpfulness):** {corr_sentiment_helpfulness:.2f} (if shown in summary above).
                    - **{high_help_count_cv} reviews** are highly helpful (ratio > 0.75), with an average sentiment score of **{avg_sent_high_help_cv:.2f}**.
                    - This can indicate if helpful reviews tend to be more positive, negative, or neutral. For example, very critical (negative sentiment) but detailed reviews might be rated as highly helpful if they provide specific, actionable feedback.
                    """)
                else:
                    st.write("No data for helpfulness vs sentiment insights.")
        else:
            st.write("No helpfulness or sentiment score data for scatter plot.")

# Corrected Product Intelligence Dashboard
elif st.session_state.current_page == 'Product Intelligence':
    st.title("💡 Product Intelligence Dashboard")
    back_to_main_menu_button()

    if st.session_state.uploaded_df is None:
        st.warning("⚠️ Please upload a dataset from the 'Main Menu' page first.")
        st.stop()
    else:
        df = st.session_state.uploaded_df.copy()
        required_pi_cols = ['ProductId', 'Score', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Text']
        if not all(col in df.columns for col in required_pi_cols):
            missing_cols = [col for col in required_pi_cols if col not in df.columns]
            st.error(f"Dataset is missing one or more required columns for Product Intelligence: {', '.join(missing_cols)}")
            st.stop()

        df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
        df['ProductId'] = df['ProductId'].astype(str)
        df.dropna(subset=['Score', 'ProductId'], inplace=True) # Critical for grouping

        if df.empty: # Check after initial cleaning specific to this page
            st.warning("No valid data for Product Intelligence after filtering for Score and ProductId.")
            st.stop()

        df['HelpfulnessRatio'] = df['HelpfulnessNumerator'] / (df['HelpfulnessDenominator'] + 1)
        df['Text'] = df['Text'].astype(str)
        df['SentimentScore'] = df['Text'].apply(lambda x: TextBlob(x).sentiment.polarity if pd.notnull(x) else 0)

        # Calculate product_stats early for the summary and charts
        if not df.empty:
            product_stats = df.groupby('ProductId').agg(
                AvgRating=('Score', 'mean'),
                ReviewCount=('Score', 'count'),
                AvgHelpfulness=('HelpfulnessRatio', 'mean'),
                AvgSentiment=('SentimentScore', 'mean')
            ).reset_index()
            product_stats = product_stats.dropna(subset=['AvgRating']) # Ensure products have an average rating
        else:
            product_stats = pd.DataFrame() # Empty dataframe if df was empty before groupby

        if product_stats.empty:
            st.warning("No product statistics could be generated. Ensure your dataset has valid ProductIDs and Scores.")
            st.stop()

        # --- 0. Key Product Intelligence Summary ---
        st.subheader("💡 Key Product Intelligence Summary")

        num_products_analyzed = len(product_stats)
        overall_avg_rating_all_prods = product_stats['AvgRating'].mean()
        overall_avg_sentiment_all_prods = product_stats['AvgSentiment'].mean()
        overall_avg_helpfulness_all_prods = product_stats['AvgHelpfulness'].mean()
        avg_reviews_per_product_pi = product_stats['ReviewCount'].mean()
        
        top_rated_product_pi = product_stats.loc[product_stats['AvgRating'].idxmax()] if not product_stats.empty else None
        most_reviewed_product_pi = product_stats.loc[product_stats['ReviewCount'].idxmax()] if not product_stats.empty else None
        highest_sentiment_product_pi = product_stats.loc[product_stats['AvgSentiment'].idxmax()] if not product_stats.empty else None
        
        products_above_4_star_pct = (product_stats[product_stats['AvgRating'] > 4.0].shape[0] / num_products_analyzed * 100) if num_products_analyzed > 0 else 0

        col_pi1, col_pi2, col_pi3 = st.columns(3)
        with col_pi1:
            st.metric(label="Total Unique Products", value=f"{num_products_analyzed:,}")
            st.metric(label="Avg. Reviews per Product", value=f"{avg_reviews_per_product_pi:.1f}")
        with col_pi2:
            st.metric(label="Overall Avg. Product Rating", value=f"{overall_avg_rating_all_prods:.2f} ★")
            st.metric(label="Overall Avg. Product Sentiment", value=f"{overall_avg_sentiment_all_prods:.2f}")
        with col_pi3:
            st.metric(label="Products Rated > 4★ (Avg.)", value=f"{products_above_4_star_pct:.1f}%")
            st.metric(label="Overall Avg. Product Helpfulness", value=f"{overall_avg_helpfulness_all_prods:.2f}")
        
        st.markdown("---")
        st.markdown("#### Top Performers by Category:")
        col_tp1, col_tp2, col_tp3 = st.columns(3)
        with col_tp1:
            if top_rated_product_pi is not None:
                st.markdown(f"⭐ **Highest Rated:** {top_rated_product_pi['ProductId']} ({top_rated_product_pi['AvgRating']:.2f}★)")
            else:
                st.markdown("⭐ **Highest Rated:** N/A")
        with col_tp2:
            if most_reviewed_product_pi is not None:
                st.markdown(f"💬 **Most Reviewed:** {most_reviewed_product_pi['ProductId']} ({most_reviewed_product_pi['ReviewCount']:,} reviews)")
            else:
                st.markdown("💬 **Most Reviewed:** N/A")
        with col_tp3:
            if highest_sentiment_product_pi is not None:
                st.markdown(f"😃 **Highest Sentiment:** {highest_sentiment_product_pi['ProductId']} (Sent: {highest_sentiment_product_pi['AvgSentiment']:.2f})")
            else:
                st.markdown("😃 **Highest Sentiment:** N/A")
        st.markdown("---")


        # Insight functions (used for chart expanders)
        def rating_insights(stats_df): # stats_df here is product_stats
            if stats_df.empty: return "No product rating data for detailed insights."
            # overall_avg_rating_all_prods is already available from summary
            highest = stats_df.loc[stats_df['AvgRating'].idxmax()] if not stats_df['AvgRating'].empty else None
            lowest = stats_df.loc[stats_df['AvgRating'].idxmin()] if not stats_df['AvgRating'].empty else None
            
            insights_str = f"The overall average rating across **{len(stats_df)} products** is **{overall_avg_rating_all_prods:.2f} stars**.\n\n"
            if highest is not None and not pd.isna(highest['AvgRating']):
                insights_str += f"Product **{highest['ProductId']}** has the highest average rating of **{highest['AvgRating']:.2f} ★** ({int(highest['ReviewCount']):,} reviews).\n\n"
            if lowest is not None and not pd.isna(lowest['AvgRating']):
                insights_str += f"Product **{lowest['ProductId']}** has the lowest average rating of **{lowest['AvgRating']:.2f} ★** ({int(lowest['ReviewCount']):,} reviews)."
            return insights_str

        def review_count_insights(stats_df):
            if stats_df.empty or 'ReviewCount' not in stats_df.columns: return "No review count data for detailed insights."
            # avg_reviews_per_product_pi is from summary
            product_with_max_reviews = stats_df.loc[stats_df['ReviewCount'].idxmax()] if not stats_df['ReviewCount'].empty else None
            total_reviews_sum_all_prods = stats_df['ReviewCount'].sum()
            insight_str = f"Across {len(stats_df)} products, there are a total of **{int(total_reviews_sum_all_prods):,} reviews**, averaging **{avg_reviews_per_product_pi:.1f} reviews per product**.\n\n"
            if product_with_max_reviews is not None:
                insight_str += f"Product **{product_with_max_reviews['ProductId']}** has the most reviews with **{int(product_with_max_reviews['ReviewCount']):,}**."
            return insight_str

        def helpfulness_insights(stats_df):
            if stats_df.empty or 'AvgHelpfulness' not in stats_df.columns: return "No helpfulness data for detailed insights."
            # overall_avg_helpfulness_all_prods from summary
            most_helpful = stats_df.loc[stats_df['AvgHelpfulness'].idxmax()] if not stats_df['AvgHelpfulness'].empty else None
            least_helpful = stats_df.loc[stats_df['AvgHelpfulness'].idxmin()] if not stats_df['AvgHelpfulness'].empty else None
            
            insights_str = f"The overall average helpfulness ratio across products is **{overall_avg_helpfulness_all_prods:.2f}**.\n\n"
            if most_helpful is not None and not pd.isna(most_helpful['AvgHelpfulness']):
                insights_str += f"Product **{most_helpful['ProductId']}** has the most helpful reviews on average (ratio: **{most_helpful['AvgHelpfulness']:.2f}**).\n\n"
            if least_helpful is not None and not pd.isna(least_helpful['AvgHelpfulness']):
                insights_str += f"Product **{least_helpful['ProductId']}** has the least helpful reviews on average (ratio: **{least_helpful['AvgHelpfulness']:.2f}**)."
            return insights_str

        def sentiment_insights(stats_df):
            if stats_df.empty or 'AvgSentiment' not in stats_df.columns: return "No sentiment data for detailed insights."
            # overall_avg_sentiment_all_prods from summary
            most_positive = stats_df.loc[stats_df['AvgSentiment'].idxmax()] if not stats_df['AvgSentiment'].empty else None
            most_negative = stats_df.loc[stats_df['AvgSentiment'].idxmin()] if not stats_df['AvgSentiment'].empty else None

            insights_str = f"The overall average sentiment score across products is **{overall_avg_sentiment_all_prods:.2f}**.\n\n"
            if most_positive is not None and not pd.isna(most_positive['AvgSentiment']):
                insights_str += f"Product **{most_positive['ProductId']}** has the most positive average sentiment (score: **{most_positive['AvgSentiment']:.2f}**).\n\n"
            if most_negative is not None and not pd.isna(most_negative['AvgSentiment']):
                insights_str += f"Product **{most_negative['ProductId']}** has the most negative average sentiment (score: **{most_negative['AvgSentiment']:.2f}**)."
            return insights_str

        # --- Charts start here ---
        # Note: product_stats is already defined and filtered
        st.subheader("1. Average Product Ratings")
        # Show top N products for readability, e.g., top 30 by review count, then sort by rating
        # This ensures we see relevant products that also have a decent number of reviews.
        top_products_for_rating_chart = product_stats.nlargest(30, 'ReviewCount').sort_values('AvgRating', ascending=False)
        
        if not top_products_for_rating_chart.empty:
            # Determine dynamic domain for y-axis if needed, or keep fixed [1,5]
            min_rating_domain = top_products_for_rating_chart['AvgRating'].min() - 0.1 if not top_products_for_rating_chart.empty else 1
            max_rating_domain = top_products_for_rating_chart['AvgRating'].max() + 0.1 if not top_products_for_rating_chart.empty else 5
            min_rating_domain = max(1, min_rating_domain) # Ensure it doesn't go below 1
            max_rating_domain = min(5, max_rating_domain) # Ensure it doesn't go above 5


            rating_chart = alt.Chart(top_products_for_rating_chart).mark_bar().encode(
                x=alt.X('ProductId:N', sort='-y', title='Product ID (Top 30 by Review Count, then by Rating)'),
                y=alt.Y('AvgRating:Q', title='Average Rating', scale=alt.Scale(domain=[min_rating_domain, max_rating_domain])),
                tooltip=['ProductId', alt.Tooltip('AvgRating', format=".2f"), 'ReviewCount']
            ).properties(
                width=700, height=350, title="Average Rating (Top Products)"
            ).configure_axis(labelColor='white', titleColor='white').configure_title(color='white')
            st.altair_chart(rating_chart, use_container_width=True)
        else:
            st.write("Not enough data to display the average product ratings chart for top products.")
            
        with st.expander("📘 What this chart shows"):
            st.markdown(rating_insights(product_stats)) # Pass the full product_stats for overall insights

        st.subheader("2. Number of Reviews per Product")
        top_prods_by_reviews_pie = product_stats.nlargest(10, 'ReviewCount') # Use nlargest for clarity
        if not top_prods_by_reviews_pie.empty:
            pie_chart = px.pie(
                top_prods_by_reviews_pie,
                names='ProductId',
                values='ReviewCount',
                title='Top 10 Products by Review Count',
                hole=0.3,
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            pie_chart.update_traces(textposition='inside', textinfo='percent+label')
            pie_chart.update_layout(title_font_color='white', legend_title_font_color='white', legend_font_color='black', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(pie_chart, use_container_width=True)
        else:
            st.write("Not enough product data to display review count pie chart.")
        with st.expander("📘 What this chart shows"):
            st.markdown(review_count_insights(product_stats))

        st.subheader("3. Average Helpfulness Ratio by Product")
        top_prods_for_help_chart = product_stats.nlargest(30, 'ReviewCount').sort_values('AvgHelpfulness', ascending=False)
        if not top_prods_for_help_chart.empty:
            helpfulness_chart_pi = alt.Chart(top_prods_for_help_chart).mark_bar(color='green').encode(
                x=alt.X('AvgHelpfulness:Q', title='Avg Helpfulness Ratio', axis=alt.Axis(format=".2f")),
                y=alt.Y('ProductId:N', sort='-x', title='Product ID (Top 30 by Review Count)'),
                tooltip=['ProductId', alt.Tooltip('AvgHelpfulness', format=".2f"), 'ReviewCount']
            ).properties(
                width=700, height=450, title="Average Helpfulness (Top Products)"
            ).configure_axis(labelColor='white', titleColor='white').configure_title(color='black')
            st.altair_chart(helpfulness_chart_pi, use_container_width=True)
        else:
            st.write("Not enough data to display the average helpfulness chart for top products.")
        with st.expander("📘 What this chart shows"):
            st.markdown(helpfulness_insights(product_stats))

        st.subheader("4. Average Sentiment Score by Product")
        top_prods_for_sent_chart = product_stats.nlargest(30, 'ReviewCount').sort_values('AvgSentiment', ascending=False)
        if not top_prods_for_sent_chart.empty:
            sentiment_chart_pi = alt.Chart(top_prods_for_sent_chart).mark_bar(color='purple').encode(
                x=alt.X('AvgSentiment:Q', title='Avg Sentiment Score', axis=alt.Axis(format=".2f")),
                y=alt.Y('ProductId:N', sort='-x', title='Product ID (Top 30 by Review Count)'),
                tooltip=['ProductId', alt.Tooltip('AvgSentiment', format=".2f"), 'ReviewCount']
            ).properties(
                width=700, height=450, title="Average Sentiment Score (Top Products)"
            ).configure_axis(labelColor='white', titleColor='white').configure_title(color='black')
            st.altair_chart(sentiment_chart_pi, use_container_width=True)
        else:
            st.write("Not enough data to display the average sentiment chart for top products.")
        with st.expander("📘 What this chart shows"):
            st.markdown(sentiment_insights(product_stats))

        # Download button for product_stats (full, not just charted)
        csv = product_stats.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Product Summary CSV",
            data=csv,
            file_name='product_intelligence_summary.csv',
            mime='text/csv'
        )

# Corrected Seller Quality Dashboard
elif st.session_state.current_page == 'Seller Quality Dashboard':
    st.title("🏷️ Seller Quality Dashboard")
    back_to_main_menu_button()

    if st.session_state.uploaded_df is None:
        st.warning("⚠️ Please upload the dataset from the 'Main Menu' page first.")
        st.stop()
    else:
        df = st.session_state.uploaded_df.copy()
        required_sq_cols = ['UserId', 'Score', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Text', 'Time']
        if not all(col in df.columns for col in required_sq_cols):
            missing_cols = [col for col in required_sq_cols if col not in df.columns]
            st.error(f"Dataset is missing one or more required columns for Seller/User Quality: {', '.join(missing_cols)}")
            st.stop()

        df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
        df['UserId'] = df['UserId'].astype(str)
        df.dropna(subset=['Score', 'UserId'], inplace=True)

        df['HelpfulnessRatio'] = df['HelpfulnessNumerator'] / (df['HelpfulnessDenominator'] + 1)
        df['Text'] = df['Text'].astype(str)
        df['SentimentScore'] = df['Text'].apply(lambda x: TextBlob(x).sentiment.polarity if pd.notnull(x) else 0)
        try:
            df['ReviewTime'] = pd.to_datetime(df['Time'], unit='s', errors='coerce')
        except Exception as e:
            st.warning(f"Could not convert 'Time' to datetime for seller quality: {e}")
            df['ReviewTime'] = pd.NaT


        if not df.empty:
            seller_df = df.groupby('UserId').agg(
                AvgRating=('Score', 'mean'),
                ReviewCount=('Score', 'count'),
                AvgHelpfulness=('HelpfulnessRatio', 'mean'),
                AvgSentiment=('SentimentScore', 'mean'),
                FirstReview=('ReviewTime', 'min'),
                LastReview=('ReviewTime', 'max')
            ).reset_index()
            seller_df = seller_df.dropna(subset=['AvgRating'])
        else:
            seller_df = pd.DataFrame()

        if not seller_df.empty:
            st.subheader("1. 📊 Average Rating by User")
            top_n_sellers = seller_df.sort_values('ReviewCount', ascending=False).head(20)

            rating_chart_sq = alt.Chart(top_n_sellers.sort_values('AvgRating', ascending=False)).mark_bar().encode(
                x=alt.X('UserId:N', sort='-y', title='User ID (Top 20 by Review Count)'),
                y=alt.Y('AvgRating:Q', title='Avg Rating', scale=alt.Scale(domain=[1,5])),
                tooltip=['UserId', alt.Tooltip('AvgRating', format=".2f"), 'ReviewCount']
            ).properties(
                height=400, title="Average Rating for Top 20 Users (by Review Count)"
            ).configure_axis(labelColor='black', titleColor='black').configure_title(color='black')
            st.altair_chart(rating_chart_sq, use_container_width=True)
            with st.expander("📘 Insights"):
                if not seller_df.empty:
                    top_seller = seller_df.loc[seller_df['AvgRating'].idxmax()] if not seller_df.empty else None
                    low_seller = seller_df.loc[seller_df['AvgRating'].idxmin()] if not seller_df.empty else None
                    avg_rating_overall_users = seller_df['AvgRating'].mean()
                    st.markdown(f"""
                    - Overall, user **{top_seller['UserId'] if top_seller is not None else 'N/A'}** tends to give the highest average rating of **{top_seller['AvgRating']:.2f}** stars ({int(top_seller['ReviewCount'] if top_seller is not None else 0)} reviews).
                    - User **{low_seller['UserId'] if low_seller is not None else 'N/A'}** tends to give the lowest average rating of **{low_seller['AvgRating']:.2f}** stars ({int(low_seller['ReviewCount'] if low_seller is not None else 0)} reviews).
                    - The average rating given across all users is **{avg_rating_overall_users:.2f}**.
                    """)

            st.subheader("2. 📦 Number of Reviews per User")
            review_count_chart_sq = alt.Chart(top_n_sellers.sort_values('ReviewCount', ascending=False)).mark_bar(color='orange').encode(
                x=alt.X('UserId:N', sort='-y', title='User ID (Top 20 by Review Count)'),
                y=alt.Y('ReviewCount:Q', title='Review Count'),
                tooltip=['UserId', 'ReviewCount', alt.Tooltip('AvgRating', format=".2f")]
            ).properties(
                height=400, title="Review Count for Top 20 Users"
            ).configure_axis(labelColor='black', titleColor='black').configure_title(color='black')
            st.altair_chart(review_count_chart_sq, use_container_width=True)
            with st.expander("📘 Insights"):
                if not seller_df.empty and 'ReviewCount' in seller_df.columns:
                    busiest_seller = seller_df.loc[seller_df['ReviewCount'].idxmax()] if not seller_df.empty else None
                    avg_reviews_per_seller = seller_df['ReviewCount'].mean()
                    st.markdown(f"""
                    - 🥇 Most active user: **{busiest_seller['UserId'] if busiest_seller is not None else 'N/A'}** with **{int(busiest_seller['ReviewCount'] if busiest_seller is not None else 0)}** reviews.
                    - Average number of reviews per user: **{avg_reviews_per_seller:.1f}**.
                    """)

            st.subheader("3. 👍 Average Helpfulness of Reviews by User")
            helpfulness_chart_sq = alt.Chart(top_n_sellers.sort_values('AvgHelpfulness', ascending=False)).mark_bar(color='green').encode(
                x=alt.X('AvgHelpfulness:Q', title='Avg Helpfulness Ratio of Reviews Written', axis=alt.Axis(format=".2f")), # Corrected
                y=alt.Y('UserId:N', sort='-x', title='User ID (Top 20 by Review Count)'),
                tooltip=['UserId', alt.Tooltip('AvgHelpfulness', format=".2f"), 'ReviewCount']
            ).properties(
                height=500, title="Average Helpfulness of Reviews by Top 20 Users"
            ).configure_axis(labelColor='black', titleColor='black').configure_title(color='black')
            st.altair_chart(helpfulness_chart_sq, use_container_width=True)
            with st.expander("📘 Insights"):
                if not seller_df.empty and 'AvgHelpfulness' in seller_df.columns:
                    top_help_seller = seller_df.loc[seller_df['AvgHelpfulness'].idxmax()] if not seller_df.empty else None
                    low_help_seller = seller_df.loc[seller_df['AvgHelpfulness'].idxmin()] if not seller_df.empty else None
                    avg_help_overall_users = seller_df['AvgHelpfulness'].mean()
                    st.markdown(f"""
                    - ✅ User whose reviews are on average most helpful: **{top_help_seller['UserId'] if top_help_seller is not None else 'N/A'}** (avg. ratio: **{top_help_seller['AvgHelpfulness']:.2f}**).
                    - ❌ User whose reviews are on average least helpful: **{low_help_seller['UserId'] if low_help_seller is not None else 'N/A'}** (avg. ratio: **{low_help_seller['AvgHelpfulness']:.2f}**).
                    - The average helpfulness ratio of reviews across all users is **{avg_help_overall_users:.2f}**.
                    """)

            st.subheader("4. 💬 Average Sentiment of Reviews by User")
            sentiment_chart_sq = alt.Chart(top_n_sellers.sort_values('AvgSentiment', ascending=False)).mark_bar(color='purple').encode(
                x=alt.X('AvgSentiment:Q', title='Avg Sentiment of Reviews Written', axis=alt.Axis(format=".2f")), # Corrected
                y=alt.Y('UserId:N', sort='-x', title='User ID (Top 20 by Review Count)'),
                tooltip=['UserId', alt.Tooltip('AvgSentiment', format=".2f"), 'ReviewCount']
            ).properties(
                height=400, title="Average Sentiment of Reviews by Top 20 Users"
            ).configure_axis(labelColor='black', titleColor='black').configure_title(color='black')
            st.altair_chart(sentiment_chart_sq, use_container_width=True)
            with st.expander("📘 Insights"):
                if not seller_df.empty and 'AvgSentiment' in seller_df.columns:
                    best_sent_seller = seller_df.loc[seller_df['AvgSentiment'].idxmax()] if not seller_df.empty else None
                    worst_sent_seller = seller_df.loc[seller_df['AvgSentiment'].idxmin()] if not seller_df.empty else None
                    avg_sent_overall_users = seller_df['AvgSentiment'].mean()
                    st.markdown(f"""
                    - 😃 User who writes the most positive reviews on average: **{best_sent_seller['UserId'] if best_sent_seller is not None else 'N/A'}** (avg. score: **{best_sent_seller['AvgSentiment']:.2f}**).
                    - 😠 User who writes the most negative reviews on average: **{worst_sent_seller['UserId'] if worst_sent_seller is not None else 'N/A'}** (avg. score: **{worst_sent_seller['AvgSentiment']:.2f}**).
                    - The average sentiment of reviews across all users is **{avg_sent_overall_users:.2f}**.
                    """)
        else:
            st.info("No seller/user statistics to display. Ensure 'UserId' column exists and has valid data, and the dataframe is not empty after processing.")

# --------------- 💹 Market Intelligence Dashboard ---------------
elif st.session_state.current_page == 'Market Intelligence':
    st.title("💹 Market Intelligence Dashboard")
    back_to_main_menu_button()

    if st.session_state.uploaded_df is None:
        st.warning("⚠️ Please upload a dataset from the 'Main Menu' page first.")
        st.stop()
    else:
        df_mi = st.session_state.uploaded_df.copy()

        required_mi_cols = ['Time', 'Score', 'Text', 'ProductId', 'HelpfulnessNumerator', 'HelpfulnessDenominator']
        if not all(col in df_mi.columns for col in required_mi_cols):
            missing_cols = [col for col in required_mi_cols if col not in df_mi.columns]
            st.error(f"Dataset is missing one or more required columns for Market Intelligence: {', '.join(missing_cols)}")
            st.stop()

        try:
            df_mi['ReviewTime'] = pd.to_datetime(df_mi['Time'], unit='s', errors='coerce')
            df_mi.dropna(subset=['ReviewTime'], inplace=True)
        except Exception as e:
            st.error(f"Error converting 'Time' column: {e}. This column is essential for Market Intelligence.")
            st.stop()

        df_mi['Score'] = pd.to_numeric(df_mi['Score'], errors='coerce')
        df_mi['Text'] = df_mi['Text'].astype(str)
        df_mi['ProductId'] = df_mi['ProductId'].astype(str)
        df_mi.dropna(subset=['Score', 'Text', 'ProductId'], inplace=True)

        if df_mi.empty:
            st.warning("No data available for Market Intelligence after initial processing and cleaning.")
            st.stop()

        df_mi['SentimentScore'] = df_mi['Text'].apply(lambda x: TextBlob(x).sentiment.polarity if pd.notnull(x) else 0)
        df_mi['HelpfulnessRatio'] = df_mi['HelpfulnessNumerator'] / (df_mi['HelpfulnessDenominator'] + 1)
        # Ensure ReviewYearMonth is string for consistent grouping if used later, though not explicitly used in charts here
        # df_mi['ReviewYearMonth'] = df_mi['ReviewTime'].dt.to_period('M').astype(str) 

        # --- 0. Key Market Insights Summary (Dynamic) ---
        st.subheader("📊 Key Market Insights Summary")

        start_date = df_mi['ReviewTime'].min().strftime('%B %Y')
        end_date = df_mi['ReviewTime'].max().strftime('%B %Y')
        total_reviews_analyzed = len(df_mi)
        overall_avg_score = df_mi['Score'].mean()
        overall_avg_sentiment = df_mi['SentimentScore'].mean()
        num_unique_products = df_mi['ProductId'].nunique()

        monthly_trends_calc = df_mi.set_index('ReviewTime').resample('M').agg(
            TotalReviews=('Score', 'count'),
            AverageScore=('Score', 'mean'),
            AverageSentiment=('SentimentScore', 'mean') # Added for potential summary use
        ).reset_index()

        peak_volume_month_data = None
        peak_score_month_data = None
        peak_sentiment_month_data = None # For summary
        
        if not monthly_trends_calc.empty:
            if 'TotalReviews' in monthly_trends_calc.columns and not monthly_trends_calc['TotalReviews'].empty:
                peak_volume_month_data = monthly_trends_calc.loc[monthly_trends_calc['TotalReviews'].idxmax()]
            if 'AverageScore' in monthly_trends_calc.columns and not monthly_trends_calc['AverageScore'].empty:
                peak_score_month_data = monthly_trends_calc.loc[monthly_trends_calc['AverageScore'].idxmax()]
            if 'AverageSentiment' in monthly_trends_calc.columns and not monthly_trends_calc['AverageSentiment'].empty: # For summary
                peak_sentiment_month_data = monthly_trends_calc.loc[monthly_trends_calc['AverageSentiment'].idxmax()]


        product_summary_calc = df_mi.groupby('ProductId').agg(
            AvgRating=('Score', 'mean'),
            ReviewCount=('Score', 'count'),
            AvgSentiment=('SentimentScore','mean') # Added for summary
        ).reset_index()

        product_highest_rating_data = None
        product_most_reviews_data = None
        product_best_sentiment_data = None # For summary
        highly_rated_products_percentage = 0

        if not product_summary_calc.empty:
            if 'AvgRating' in product_summary_calc.columns and not product_summary_calc['AvgRating'].empty:
                 product_highest_rating_data = product_summary_calc.loc[product_summary_calc['AvgRating'].idxmax()]
            if 'ReviewCount' in product_summary_calc.columns and not product_summary_calc['ReviewCount'].empty:
                product_most_reviews_data = product_summary_calc.loc[product_summary_calc['ReviewCount'].idxmax()]
            if 'AvgSentiment' in product_summary_calc.columns and not product_summary_calc['AvgSentiment'].empty: # For summary
                product_best_sentiment_data = product_summary_calc.loc[product_summary_calc['AvgSentiment'].idxmax()]

            if len(product_summary_calc) > 0 and 'AvgRating' in product_summary_calc.columns:
                 highly_rated_products_percentage = (product_summary_calc[product_summary_calc['AvgRating'] > 4.0].shape[0] / len(product_summary_calc)) * 100


        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric(label="Review Period", value=f"{start_date} - {end_date}")
            st.metric(label="Total Reviews Analyzed", value=f"{total_reviews_analyzed:,}")
        with col_s2:
            st.metric(label="Overall Average Score", value=f"{overall_avg_score:.2f} ★")
            st.metric(label="Overall Average Sentiment", value=f"{overall_avg_sentiment:.2f}")
        with col_s3:
            st.metric(label="Unique Products in Dataset", value=f"{num_unique_products:,}")
            st.metric(label="Products Rated > 4 ★ (Avg)", value=f"{highly_rated_products_percentage:.1f}%")

        st.markdown("---")
        st.markdown("#### Peak Performance & Top Mentions:")

        row1_col1, row1_col2, row1_col3 = st.columns(3)
        with row1_col1:
            if peak_volume_month_data is not None:
                st.markdown(f"📅 **Most Active Month (Volume):**<br>{peak_volume_month_data['ReviewTime'].strftime('%B %Y')} ({peak_volume_month_data['TotalReviews']:,} reviews)", unsafe_allow_html=True)
            else:
                st.markdown("📅 **Most Active Month (Volume):** N/A")
        with row1_col2:
            if peak_score_month_data is not None:
                st.markdown(f"⭐ **Best Month (Avg. Score):**<br>{peak_score_month_data['ReviewTime'].strftime('%B %Y')} ({peak_score_month_data['AverageScore']:.2f} ★)", unsafe_allow_html=True)
            else:
                st.markdown("⭐ **Best Month (Avg. Score):** N/A")
        with row1_col3:
             if peak_sentiment_month_data is not None:
                st.markdown(f"😃 **Best Month (Avg. Sentiment):**<br>{peak_sentiment_month_data['ReviewTime'].strftime('%B %Y')} (Score: {peak_sentiment_month_data['AverageSentiment']:.2f})", unsafe_allow_html=True)
             else:
                st.markdown("😃 **Best Month (Avg. Sentiment):** N/A")


        row2_col1, row2_col2, row2_col3 = st.columns(3)
        with row2_col1:
            if product_highest_rating_data is not None:
                st.markdown(f"🏆 **Highest Rated Product (Avg.):**<br>{product_highest_rating_data['ProductId']} ({product_highest_rating_data['AvgRating']:.2f} ★)", unsafe_allow_html=True)
            else:
                st.markdown("🏆 **Highest Rated Product (Avg.):** N/A")
        with row2_col2:
            if product_most_reviews_data is not None:
                st.markdown(f"💬 **Most Reviewed Product:**<br>{product_most_reviews_data['ProductId']} ({product_most_reviews_data['ReviewCount']:,} reviews)", unsafe_allow_html=True)
            else:
                st.markdown("💬 **Most Reviewed Product:** N/A")
        with row2_col3:
            if product_best_sentiment_data is not None:
                st.markdown(f"💖 **Top Sentiment Product (Avg.):**<br>{product_best_sentiment_data['ProductId']} (Score: {product_best_sentiment_data['AvgSentiment']:.2f})", unsafe_allow_html=True)
            else:
                st.markdown("💖 **Top Sentiment Product (Avg.):** N/A")
        st.markdown("---")


        st.subheader("1. 📈 Detailed Market Trends Over Time")

        # monthly_trends for charts (already calculated as monthly_trends_calc)
        monthly_trends_for_charts = monthly_trends_calc.copy()
        if not monthly_trends_for_charts.empty:
             monthly_trends_for_charts['ReviewTime'] = pd.to_datetime(monthly_trends_for_charts['ReviewTime']).dt.strftime('%Y-%m')


        if not monthly_trends_for_charts.empty:
            col1_mt, col2_mt = st.columns(2)
            with col1_mt:
                st.write("#### Review Volume Trend")
                volume_chart = alt.Chart(monthly_trends_for_charts).mark_line(point=True).encode(
                    x=alt.X('ReviewTime:T', title='Month', axis=alt.Axis(format='%Y-%m')),
                    y=alt.Y('TotalReviews:Q', title='Number of Reviews'),
                    tooltip=[alt.Tooltip('ReviewTime:T', title='Month'), alt.Tooltip('TotalReviews:Q', title='Reviews')]
                ).properties(height=300).configure_axis(labelColor='white', titleColor='white').configure_title(color='black')
                st.altair_chart(volume_chart, use_container_width=True)
                # Dynamic insight for Volume Trend
                min_vol_month = monthly_trends_calc.loc[monthly_trends_calc['TotalReviews'].idxmin()]
                avg_monthly_vol = monthly_trends_calc['TotalReviews'].mean()
                st.markdown(f"""
                * Review volume peaked in **{peak_volume_month_data['ReviewTime'].strftime('%B %Y')}** ({peak_volume_month_data['TotalReviews']:,} reviews).
                * Lowest volume was in **{min_vol_month['ReviewTime'].strftime('%B %Y')}** ({min_vol_month['TotalReviews']:,} reviews).
                * Average monthly review volume: **{avg_monthly_vol:,.0f}**.
                """)

            with col2_mt:
                st.write("#### Average Score Trend")
                score_trend_chart = alt.Chart(monthly_trends_for_charts).mark_line(point=True, color='green').encode(
                    x=alt.X('ReviewTime:T', title='Month', axis=alt.Axis(format='%Y-%m')),
                    y=alt.Y('AverageScore:Q', title='Average Score', scale=alt.Scale(domain=[1,5])),
                    tooltip=[alt.Tooltip('ReviewTime:T', title='Month'), alt.Tooltip('AverageScore:Q', title='Avg Score', format=".2f")]
                ).properties(height=300).configure_axis(labelColor='white', titleColor='white').configure_title(color='black')
                st.altair_chart(score_trend_chart, use_container_width=True)
                # Dynamic insight for Score Trend
                min_score_month = monthly_trends_calc.loc[monthly_trends_calc['AverageScore'].idxmin()]
                st.markdown(f"""
                * Highest average score of **{peak_score_month_data['AverageScore']:.2f} ★** was observed in **{peak_score_month_data['ReviewTime'].strftime('%B %Y')}**.
                * Lowest average score was **{min_score_month['AverageScore']:.2f} ★** in **{min_score_month['ReviewTime'].strftime('%B %Y')}**.
                * The overall average score across months is **{monthly_trends_calc['AverageScore'].mean():.2f} ★**.
                """)

            st.write("#### Average Sentiment Trend")
            sentiment_trend_chart = alt.Chart(monthly_trends_for_charts).mark_line(point=True, color='purple').encode(
                x=alt.X('ReviewTime:T', title='Month', axis=alt.Axis(format='%Y-%m')),
                y=alt.Y('AverageSentiment:Q', title='Average Sentiment Score', axis=alt.Axis(format=".2f")),
                tooltip=[alt.Tooltip('ReviewTime:T', title='Month'), alt.Tooltip('AverageSentiment:Q', title='Avg Sentiment', format=".2f")]
            ).properties(height=300).configure_axis(labelColor='white', titleColor='white').configure_title(color='black')
            st.altair_chart(sentiment_trend_chart, use_container_width=True)
            # Dynamic insight for Sentiment Trend
            min_sentiment_month = monthly_trends_calc.loc[monthly_trends_calc['AverageSentiment'].idxmin()]
            st.markdown(f"""
            * Market sentiment peaked at **{peak_sentiment_month_data['AverageSentiment']:.2f}** in **{peak_sentiment_month_data['ReviewTime'].strftime('%B %Y')}**.
            * The most negative average sentiment was **{min_sentiment_month['AverageSentiment']:.2f}** in **{min_sentiment_month['ReviewTime'].strftime('%B %Y')}**.
            * The overall average sentiment across months is **{monthly_trends_calc['AverageSentiment'].mean():.2f}**.
            """)

            with st.expander("📘 Interpreting Market Trends (General Guide)"):
                st.markdown("""
                - **Review Volume:** Shows the overall market activity. Spikes might indicate popular product launches, seasonal demand, or marketing campaigns. Dips could signal declining interest or market saturation for the products in the dataset.
                - **Average Score:** Reflects overall customer satisfaction. A declining trend might warrant investigation into product quality or unmet expectations across the board.
                - **Average Sentiment:** Provides a nuanced view of customer opinion. Even if scores are stable, sentiment might reveal subtle shifts in how positively or negatively customers are expressing themselves.
                """)
        else:
            st.write("Not enough monthly data to display market trends.")

        st.subheader("2. 🌟 Product Performance Snapshot")
        # product_summary_calc is already available
        product_summary_mi_filtered = product_summary_calc[product_summary_calc['ReviewCount'] > 5].copy()

        max_review_count_for_domain = 10
        if not product_summary_mi_filtered.empty:
            max_review_count_for_domain = product_summary_mi_filtered['ReviewCount'].max()
        if max_review_count_for_domain <= 1: max_review_count_for_domain = 10


        if not product_summary_mi_filtered.empty:
            st.write("#### Product Performance Quadrant (Avg Rating vs. Review Count)")
            performance_scatter = alt.Chart(product_summary_mi_filtered).mark_circle(size=100, opacity=0.7).encode(
                x=alt.X('ReviewCount:Q', title='Number of Reviews (Log Scale)',
                        scale=alt.Scale(type="log", domain=[1, max_review_count_for_domain])),
                y=alt.Y('AvgRating:Q', title='Average Rating', scale=alt.Scale(domain=[1,5])),
                color=alt.Color('AvgSentiment:Q', scale=alt.Scale(scheme='redblue'), title='Avg. Sentiment'),
                tooltip=['ProductId',
                         alt.Tooltip('ReviewCount:Q', title='Reviews'),
                         alt.Tooltip('AvgRating:Q', title='Avg Rating', format=".2f"),
                         alt.Tooltip('AvgSentiment:Q', title='Avg Sentiment', format=".2f")]
            ).properties(
                height=400, title="Product Performance: Rating vs. Volume (Log Scale for Volume)"
            ).interactive().configure_axis(labelColor='white', titleColor='white').configure_title(color='black').configure_legend(labelColor='black', titleColor='black')
            st.altair_chart(performance_scatter, use_container_width=True)
            
            # Dynamic insight for Product Performance Quadrant
            num_prods_in_chart = len(product_summary_mi_filtered)
            prods_above_4_star_in_chart = product_summary_mi_filtered[product_summary_mi_filtered['AvgRating'] > 4.0].shape[0]
            percent_prods_above_4_star_in_chart = (prods_above_4_star_in_chart / num_prods_in_chart * 100) if num_prods_in_chart > 0 else 0
            avg_sentiment_in_chart = product_summary_mi_filtered['AvgSentiment'].mean()
            
            st.markdown(f"""
            * This snapshot analyzes **{num_prods_in_chart} products** (those with more than 5 reviews).
            * Among these, **{prods_above_4_star_in_chart} products ({percent_prods_above_4_star_in_chart:.1f}%)** have an average rating above 4.0 stars.
            * The average sentiment score for products in this chart is **{avg_sentiment_in_chart:.2f}**.
            * *(Refer to the 'Key Market Insights Summary' at the top for overall best performing products across the entire dataset.)*
            """)

            with st.expander("📘 Interpreting Product Performance Quadrant (General Guide)"):
                st.markdown("""
                This chart helps identify different product categories (among those with >5 reviews):
                - **High Rating, High Reviews (Top Right):** Market leaders, strong performers.
                - **High Rating, Low Reviews (Top Left):** Potential stars, niche products, or new successful entries. Could benefit from increased visibility.
                - **Low Rating, High Reviews (Bottom Right):** Problematic products with high visibility. Require urgent attention.
                - **Low Rating, Low Reviews (Bottom Left):** Underperformers or new products struggling to gain traction.
                - **Color:** Indicates the average sentiment, adding another layer to performance (e.g., a high-rated product might still have lukewarm sentiment).
                *(Note: Review count is on a log scale to better visualize products with vastly different review volumes.)*
                """)
        else:
            st.write("Not enough product data (or products with >5 reviews) to display performance snapshot.")

        st.subheader("3. 🗣️ Emerging Themes from Recent Reviews")

        if 'ReviewTime' in df_mi.columns and pd.api.types.is_datetime64_any_dtype(df_mi['ReviewTime']) and not df_mi['ReviewTime'].empty:
            latest_date = df_mi['ReviewTime'].max()
            start_recent_period = latest_date - pd.DateOffset(years=1)
            recent_reviews_df = df_mi[df_mi['ReviewTime'] >= start_recent_period].copy()

            if recent_reviews_df.empty:
                st.write(f"No reviews found from the last year (since {start_recent_period.strftime('%B %Y')}). Showing themes from the latest 1000 reviews instead if available.")
                recent_reviews_df = df_mi.nlargest(1000, 'ReviewTime').copy()
        else:
            recent_reviews_df = df_mi.nlargest(1000, 'Time').copy()


        if not recent_reviews_df.empty and 'Text' in recent_reviews_df.columns and not recent_reviews_df['Text'].dropna().empty:
            num_recent_reviews_analyzed = len(recent_reviews_df)
            st.write(f"Analyzing keywords from {num_recent_reviews_analyzed} recent reviews.")

            col1_wc, col2_wc = st.columns(2)
            common_stopwords_ext = ['product', 'item', 'review', 'amazon', 'order', 'get', 'got', 'would', 'make', 'use', 'also', 'one', 'like', 'thi', 'veri', 'good', 'great', 'time', 'I', 'it', 's', 't', 'm', 've', 're', 'price', 'quality', 'love', 'day', 'week', 'month', 'year', 'really', 'even', 'bought', 'recommend', 'will', 'well', 'much', 'buy', 'need', 'work', 'just', 'ca', 'na', 'wa', 'ha', 'doe']

            positive_text_recent = ' '.join(recent_reviews_df[recent_reviews_df['SentimentScore'] > 0.1]['Text'].astype(str).dropna())
            num_pos_recent = recent_reviews_df[recent_reviews_df['SentimentScore'] > 0.1].shape[0]
            if positive_text_recent.strip():
                with col1_wc:
                    st.write("#### Positive Keywords (Recent)")
                    try:
                        current_stopwords = set(WordCloud().stopwords)
                        current_stopwords.update(common_stopwords_ext)
                        wc_pos = WordCloud(width=400, height=200, background_color='white', stopwords=current_stopwords, collocations=True, max_words=50, colormap='viridis').generate(positive_text_recent)
                        fig_wc_p, ax_wc_p = plt.subplots()
                        ax_wc_p.imshow(wc_pos, interpolation='bilinear')
                        ax_wc_p.axis('off')
                        fig_wc_p.patch.set_alpha(0)
                        st.pyplot(fig_wc_p)
                        st.markdown(f"*Generated from **{num_pos_recent}** recent positive reviews.*")
                    except Exception as e:
                        st.error(f"Could not generate positive word cloud: {e}")
            else:
                with col1_wc:
                    st.write("No distinct positive text found in recent reviews for word cloud.")

            negative_text_recent = ' '.join(recent_reviews_df[recent_reviews_df['SentimentScore'] < -0.1]['Text'].astype(str).dropna())
            num_neg_recent = recent_reviews_df[recent_reviews_df['SentimentScore'] < -0.1].shape[0]
            if negative_text_recent.strip():
                with col2_wc:
                    st.write("#### Negative Keywords (Recent)")
                    try:
                        current_stopwords_neg = set(WordCloud().stopwords)
                        current_stopwords_neg.update(common_stopwords_ext)
                        current_stopwords_neg.update(['problem', 'issue', 'bad', 'disappointed', 'return', 'not', 'no', 'don', 'didn'])
                        wc_neg = WordCloud(width=400, height=200, background_color='white', stopwords=current_stopwords_neg, collocations=True, max_words=50, colormap='plasma').generate(negative_text_recent)
                        fig_wc_n, ax_wc_n = plt.subplots()
                        ax_wc_n.imshow(wc_neg, interpolation='bilinear')
                        ax_wc_n.axis('off')
                        fig_wc_n.patch.set_alpha(0)
                        st.pyplot(fig_wc_n)
                        st.markdown(f"*Generated from **{num_neg_recent}** recent negative reviews.*")
                    except Exception as e:
                        st.error(f"Could not generate negative word cloud: {e}")
            else:
                with col2_wc:
                    st.write("No distinct negative text found in recent reviews for word cloud.")

            with st.expander("📘 Interpreting Emerging Themes (General Guide)"):
                st.markdown("""
                - These word clouds highlight the most frequent terms (and common two-word phrases if `collocations=True`) in recent positive and negative reviews.
                - **Positive Keywords:** Indicate what customers currently appreciate (e.g., specific features, aspects of service, emerging positive trends).
                - **Negative Keywords:** Point to current pain points, defects, or areas of dissatisfaction that might be growing in relevance.
                - Comparing these to older review themes (if the dataset spanned a longer period and was filtered accordingly) could show how market perceptions are evolving.
                """)
        else:
            st.write("Not enough recent review text to generate emerging themes.")

# Corrected Risk Management Dashboard
elif st.session_state.current_page == 'Risk Management':
    st.title("🚨 Risk Management Dashboard")
    back_to_main_menu_button()

    if st.session_state.uploaded_df is None:
        st.warning("⚠️ Please upload a dataset from the 'Main Menu' page first.")
        st.stop()
    else:
        df_rm = st.session_state.uploaded_df.copy()

        required_rm_cols = ['Time', 'Score', 'Text', 'ProductId', 'HelpfulnessNumerator', 'HelpfulnessDenominator']
        if not all(col in df_rm.columns for col in required_rm_cols):
            missing_cols = [col for col in required_rm_cols if col not in df_rm.columns]
            st.error(f"Dataset is missing one or more required columns for Risk Management: {', '.join(missing_cols)}")
            st.stop()

        try:
            df_rm['ReviewTime'] = pd.to_datetime(df_rm['Time'], unit='s', errors='coerce')
            df_rm.dropna(subset=['ReviewTime'], inplace=True)
        except Exception as e:
            st.error(f"Error converting 'Time' column: {e}. This column is essential for Risk Management analysis.")
            st.stop()

        df_rm['Score'] = pd.to_numeric(df_rm['Score'], errors='coerce')
        df_rm['Text'] = df_rm['Text'].astype(str)
        df_rm['ProductId'] = df_rm['ProductId'].astype(str)
        df_rm.dropna(subset=['Score', 'Text', 'ProductId'], inplace=True)

        if df_rm.empty:
            st.warning("No data available for Risk Management after initial processing and cleaning.")
            st.stop()

        df_rm['SentimentScore'] = df_rm['Text'].apply(lambda x: TextBlob(x).sentiment.polarity if pd.notnull(x) else 0)
        df_rm['HelpfulnessRatio'] = df_rm['HelpfulnessNumerator'] / (df_rm['HelpfulnessDenominator'] + 1)
        df_rm['ReviewYearMonth'] = df_rm['ReviewTime'].dt.to_period('M').astype(str)

        # --- 0. Key Risk Metrics Summary ---
        st.subheader("🚨 Key Risk Metrics Summary")

        total_reviews_rm = len(df_rm)
        low_rated_reviews = df_rm[df_rm['Score'] <= 2].copy() # 1-star or 2-star reviews
        count_low_rated = len(low_rated_reviews)
        percent_low_rated = (count_low_rated / total_reviews_rm * 100) if total_reviews_rm > 0 else 0

        avg_sentiment_low_rated = low_rated_reviews['SentimentScore'].mean() if not low_rated_reviews.empty else 0.0

        product_low_rating_summary = df_rm.groupby('ProductId').agg(
            TotalProductReviews=('Score', 'count'),
            LowRatedProductReviews=('Score', lambda x: (x <= 2).sum())
        ).reset_index()
        
        product_low_rating_summary['LowRatingPercentage'] = 0.0
        if not product_low_rating_summary.empty and 'TotalProductReviews' in product_low_rating_summary.columns and 'LowRatedProductReviews' in product_low_rating_summary.columns:
            product_low_rating_summary['LowRatingPercentage'] = product_low_rating_summary.apply(
                lambda row: (row['LowRatedProductReviews'] / row['TotalProductReviews'] * 100) if row['TotalProductReviews'] > 0 else 0, axis=1
            )
        
        product_low_rating_summary = product_low_rating_summary.sort_values('LowRatingPercentage', ascending=False)
        
        top_risk_products = pd.DataFrame() # Default to empty
        unique_products_with_low_ratings = 0
        if not product_low_rating_summary.empty:
            top_risk_products = product_low_rating_summary.head(5)
            unique_products_with_low_ratings = product_low_rating_summary[product_low_rating_summary['LowRatingPercentage'] > 0].shape[0]


        col_rm1, col_rm2, col_rm3 = st.columns(3)
        with col_rm1:
            st.metric(label="Total Reviews", value=f"{total_reviews_rm:,}")
            st.metric(label="Reviews with Score <= 2 ★", value=f"{count_low_rated:,}")
        with col_rm2:
            st.metric(label="% of Low-Rated Reviews", value=f"{percent_low_rated:.1f}%")
            st.metric(label="Avg. Sentiment of Low-Rated Reviews", value=f"{avg_sentiment_low_rated:.2f}")
        with col_rm3:
            st.metric(label="Unique Products with Low Ratings", value=f"{unique_products_with_low_ratings:,}")
            if not top_risk_products.empty and 'ProductId' in top_risk_products.columns and 'LowRatingPercentage' in top_risk_products.columns:
                st.markdown("Most concerning product:")
                st.markdown(f"**{top_risk_products.iloc[0]['ProductId']}** ({top_risk_products.iloc[0]['LowRatingPercentage']:.1f}% low ratings)")
            else:
                st.markdown("Most concerning product: N/A")

        st.markdown("---")

        # --- 1. Low Rating Trends Over Time ---
        st.subheader("1. 📉 Low Rating Trends Over Time")
        low_rating_monthly_trends = df_rm[df_rm['Score'] <= 2].set_index('ReviewTime').resample('M').size().reset_index(name='LowRatingCount')
        low_rating_monthly_trends['ReviewTime'] = pd.to_datetime(low_rating_monthly_trends['ReviewTime']).dt.strftime('%Y-%m') # ensure correct format for axis

        if not low_rating_monthly_trends.empty:
            low_rating_trend_chart = alt.Chart(low_rating_monthly_trends).mark_line(point=True, color='red').encode(
                x=alt.X('ReviewTime:T', title='Month', axis=alt.Axis(format='%Y-%m')),
                y=alt.Y('LowRatingCount:Q', title='Number of Low-Rated Reviews'),
                tooltip=[alt.Tooltip('ReviewTime:T', title='Month'), alt.Tooltip('LowRatingCount:Q', title='Low Reviews')]
            ).properties(
                height=350, title="Trend of 1-Star and 2-Star Reviews Over Time"
            ).configure_axis(labelColor='white', titleColor='white').configure_title(color='black')
            st.altair_chart(low_rating_trend_chart, use_container_width=True)
            with st.expander("📘 Interpreting Low Rating Trends"):
                st.markdown("""
                - This chart shows the number of 1-star and 2-star reviews over time.
                - **Spikes or sustained increases** in this trend indicate growing customer dissatisfaction or emerging product issues.
                - It's crucial to investigate the causes behind these trends to mitigate risks.
                """)
        else:
            st.write("No low-rated review data over time to display.")

        # --- 2. Top Products by Low Rating Percentage ---
        st.subheader("2. ⚠️ Top Products by Low Rating Percentage")
        if not product_low_rating_summary.empty and 'LowRatingPercentage' in product_low_rating_summary.columns: # Check again for the processed df
            top_risk_products_chart_data = product_low_rating_summary[product_low_rating_summary['LowRatingPercentage'] > 0].head(20)

            if not top_risk_products_chart_data.empty:
                low_rating_product_chart = alt.Chart(top_risk_products_chart_data).mark_bar(color='darkred').encode(
                    x=alt.X('LowRatingPercentage:Q', title='Percentage of 1-2 Star Reviews', axis=alt.Axis(format=".1f")), # Corrected
                    y=alt.Y('ProductId:N', sort='-x', title='Product ID'),
                    tooltip=['ProductId', alt.Tooltip('LowRatingPercentage', format=".1f")]
                ).properties(
                    height=400, title="Top 20 Products with Highest % of Low-Rated Reviews"
                ).configure_axis(labelColor='white', titleColor='white').configure_title(color='black')
                st.altair_chart(low_rating_product_chart, use_container_width=True)
                with st.expander("📘 Interpreting Top Risk Products"):
                    st.markdown("""
                    - This chart identifies products that have a disproportionately high percentage of 1-star or 2-star reviews.
                    - Products at the top of this list are high-risk and warrant immediate attention for quality control, feature improvements, or customer support.
                    """)
            else:
                st.write("No products with low ratings found to display in chart.")
        else:
            st.write("No product low rating percentage data to display.")

        # --- 3. Negative Review Keywords (Word Cloud) ---
        st.subheader("3. ☁️ Negative Review Keywords")
        negative_reviews_text = ' '.join(df_rm[df_rm['SentimentScore'] < -0.1]['Text'].astype(str).dropna())

        if negative_reviews_text.strip():
            try:
                custom_stopwords_neg_rm = set(WordCloud().stopwords)
                custom_stopwords_neg_rm.update(['product', 'item', 'review', 'amazon', 'order', 'get', 'got', 'would', 'make', 'use', 'also', 'one', 'like', 'thi', 'veri', 'good', 'great', 'time', 'I', 'it', 's', 't', 'm', 've', 're', 'problem', 'issue', 'bad', 'disappointed', 'return', 'not', 'no', 'don', 'didn', 'waste', 'money', 'poor', 'worst', 'broken', 'small', 'big'])

                wc_neg_rm = WordCloud(width=800, height=400, background_color='white', stopwords=custom_stopwords_neg_rm, collocations=True, max_words=100, colormap='Reds').generate(negative_reviews_text)
                fig_wc_neg_rm, ax_wc_neg_rm = plt.subplots(figsize=(10, 5))
                ax_wc_neg_rm.imshow(wc_neg_rm, interpolation='bilinear')
                ax_wc_neg_rm.axis('off')
                fig_wc_neg_rm.patch.set_alpha(0)
                st.pyplot(fig_wc_neg_rm)
                with st.expander("📘 Interpreting Negative Review Keywords"):
                    st.markdown("""
                    - This word cloud highlights the most frequent terms used in reviews with a negative sentiment score.
                    - Larger words represent more common complaints, directly pointing to specific product flaws, service failures, or unmet expectations.
                    - This is a powerful tool for quickly identifying **root causes of dissatisfaction**.
                    """)
            except Exception as e:
                st.error(f"Could not generate negative word cloud: {e}")
        else:
            st.write("No significant negative review text to generate word cloud.")

        # --- 4. Sentiment Distribution of Low-Rated Reviews ---
        st.subheader("4. 📊 Sentiment Distribution of Low-Rated Reviews (1 & 2 Stars)")
        if not low_rated_reviews.empty:
            def categorize_sentiment(score):
                if score > 0.05:
                    return 'Positive'
                elif score < -0.05:
                    return 'Negative'
                else:
                    return 'Neutral'

            low_rated_reviews.loc[:, 'SentimentCategory'] = low_rated_reviews['SentimentScore'].apply(categorize_sentiment) # Use .loc for assignment
            sentiment_counts_low_rated = low_rated_reviews['SentimentCategory'].value_counts().reset_index()
            sentiment_counts_low_rated.columns = ['SentimentCategory', 'Count']

            sentiment_bar_chart_low_rated = alt.Chart(sentiment_counts_low_rated).mark_bar().encode(
                x=alt.X('SentimentCategory:N', title='Sentiment Category', sort=['Negative', 'Neutral', 'Positive']),
                y=alt.Y('Count:Q', title='Number of Reviews'),
                color=alt.Color('SentimentCategory:N', scale=alt.Scale(domain=['Negative', 'Neutral', 'Positive'], range=['red', 'gray', 'green']), legend=None),
                tooltip=['SentimentCategory', 'Count']
            ).properties(
                height=350, title="Sentiment Breakdown for 1-Star and 2-Star Reviews"
            ).configure_axis(labelColor='white', titleColor='white').configure_title(color='black')
            st.altair_chart(sentiment_bar_chart_low_rated, use_container_width=True)
            with st.expander("📘 Interpreting Sentiment of Low-Rated Reviews"):
                st.markdown("""
                - This chart reveals the underlying sentiment of reviews that received very low star ratings (1 or 2 stars).
                - While most low-rated reviews are expected to be "Negative", a significant number of "Neutral" or even "Positive" reviews with low star ratings could indicate:
                    - **Misunderstanding of the rating scale.**
                    - **Issues not directly related to product quality** (e.g., shipping, packaging) but still leading to low scores.
                    - **Sarcasm or nuanced feedback** that sentiment analysis might misinterpret.
                - This helps prioritize and understand the *type* of low-score feedback.
                """)
        else:
            st.write("No low-rated reviews to analyze sentiment distribution.")