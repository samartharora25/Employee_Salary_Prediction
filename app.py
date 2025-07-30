import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Configure Streamlit page
st.set_page_config(
    page_title="💰 Salary Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-title {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-title {
        font-size: 1.8rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.5rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .prediction-amount {
        font-size: 3rem;
        font-weight: bold;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .info-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #007bff;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        border-radius: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_encoders():
    """
    Load the trained Random Forest model and label encoders
    """
    try:
        # Try to load the complete package first
        if os.path.exists('salary_predictor_model.pkl'):
            model_package = joblib.load('salary_predictor_model.pkl')
            return model_package['model'], model_package['encoders'], model_package['feature_names']

        # Fallback to individual files
        elif os.path.exists('random_forest_model.pkl') and os.path.exists('label_encoders.pkl'):
            model = joblib.load('random_forest_model.pkl')
            encoders = joblib.load('label_encoders.pkl')
            feature_names = ['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']
            return model, encoders, feature_names

        else:
            return None, None, None

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None


@st.cache_data
def load_sample_data():
    """
    Load sample data for analysis and insights
    """
    try:
        df = pd.read_csv('Salary Data.csv')
        df = df.dropna()
        df = df[df['Salary'] > 1000]
        return df
    except FileNotFoundError:
        return None


def make_prediction(model, encoders, age, gender, education, job_title, experience):
    """
    Make salary prediction using the trained model
    """
    try:
        # Encode categorical variables
        gender_encoded = encoders['Gender'].transform([gender])[0]
        education_encoded = encoders['Education Level'].transform([education])[0]
        job_title_encoded = encoders['Job Title'].transform([job_title])[0]

        # Create feature array
        features = np.array([[age, gender_encoded, education_encoded, job_title_encoded, experience]])

        # Make prediction
        prediction = model.predict(features)[0]

        # Get confidence interval (using prediction intervals from Random Forest)
        # Get predictions from all trees
        tree_predictions = np.array([tree.predict(features)[0] for tree in model.estimators_])
        confidence_lower = np.percentile(tree_predictions, 10)
        confidence_upper = np.percentile(tree_predictions, 90)

        return max(0, prediction), confidence_lower, confidence_upper

    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None, None


def create_feature_importance_chart(model):
    """
    Create feature importance visualization
    """
    feature_names = ['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']
    importance_values = model.feature_importances_

    # Create the chart
    fig = px.bar(
        x=importance_values,
        y=feature_names,
        orientation='h',
        title='Feature Importance in Salary Prediction',
        labels={'x': 'Importance Score', 'y': 'Features'},
        color=importance_values,
        color_continuous_scale='viridis'
    )

    fig.update_layout(
        height=400,
        showlegend=False,
        title_font_size=18,
        title_x=0.5
    )

    return fig


def create_salary_analysis_charts(df):
    """
    Create salary analysis visualizations
    """
    if df is None:
        return None, None, None

    # 1. Salary distribution
    fig1 = px.histogram(
        df, x='Salary', nbins=30,
        title='Salary Distribution',
        labels={'Salary': 'Salary ($)', 'count': 'Number of Employees'},
        color_discrete_sequence=['#667eea']
    )
    fig1.update_layout(showlegend=False, title_x=0.5)

    # 2. Average salary by education and gender
    avg_salary = df.groupby(['Education Level', 'Gender'])['Salary'].mean().reset_index()
    fig2 = px.bar(
        avg_salary,
        x='Education Level',
        y='Salary',
        color='Gender',
        title='Average Salary by Education Level and Gender',
        barmode='group'
    )
    fig2.update_layout(title_x=0.5)

    # 3. Experience vs Salary scatter
    fig3 = px.scatter(
        df,
        x='Years of Experience',
        y='Salary',
        color='Education Level',
        size='Age',
        title='Experience vs Salary (bubble size = age)',
        hover_data=['Job Title', 'Gender']
    )
    fig3.update_layout(title_x=0.5)

    return fig1, fig2, fig3


def main():
    # Main title
    st.markdown('<h1 class="main-title">💰 Employee Salary Predictor</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center; font-size: 1.2rem; color: #666;">Powered by Random Forest Machine Learning</p>',
        unsafe_allow_html=True)

    # Load model and data
    model, encoders, feature_names = load_model_and_encoders()
    sample_data = load_sample_data()

    # Check if model is loaded
    if model is None or encoders is None:
        st.error("🚨 **Model not found!** Please run the training script first.")
        st.markdown("""
        ### Steps to fix this:
        1. Make sure you have `Salary Data.csv` in the same folder
        2. Run the training script: `python salary_predictor.py`
        3. Refresh this page
        """)
        st.stop()

    # Sidebar for navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.radio(
        "Choose a section:",
        ["🏠 Home", "🔮 Predict Salary", "📊 Data Analysis", "🤖 Model Info"]
    )

    # HOME PAGE
    if page == "🏠 Home":
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div class="info-card">
                <h2>🎯 Welcome to the Salary Predictor!</h2>
                <p>This application uses <strong>Random Forest machine learning</strong> to predict employee salaries based on:</p>
                <ul>
                    <li>📅 Age</li>
                    <li>👤 Gender</li>
                    <li>🎓 Education Level</li>
                    <li>💼 Job Title</li>
                    <li>⏰ Years of Experience</li>
                </ul>
                <p>Use the sidebar to navigate through different sections!</p>
            </div>
            """, unsafe_allow_html=True)

        # Quick stats if data is available
        if sample_data is not None:
            st.markdown('<h2 class="sub-title">📈 Quick Dataset Stats</h2>', unsafe_allow_html=True)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f'<div class="metric-card"><h3>{len(sample_data)}</h3><p>Total Records</p></div>',
                            unsafe_allow_html=True)
            with col2:
                st.markdown(
                    f'<div class="metric-card"><h3>${sample_data["Salary"].mean():,.0f}</h3><p>Avg Salary</p></div>',
                    unsafe_allow_html=True)
            with col3:
                st.markdown(
                    f'<div class="metric-card"><h3>{sample_data["Job Title"].nunique()}</h3><p>Job Titles</p></div>',
                    unsafe_allow_html=True)
            with col4:
                st.markdown(
                    f'<div class="metric-card"><h3>{sample_data["Years of Experience"].max():.0f}</h3><p>Max Experience</p></div>',
                    unsafe_allow_html=True)

    # PREDICTION PAGE
    elif page == "🔮 Predict Salary":
        st.markdown('<h2 class="sub-title">🔮 Make a Salary Prediction</h2>', unsafe_allow_html=True)

        # Get unique values for dropdowns
        if sample_data is not None:
            unique_genders = sorted(sample_data['Gender'].unique())
            unique_education = sorted(sample_data['Education Level'].unique())
            unique_job_titles = sorted(sample_data['Job Title'].unique())
        else:
            # Fallback values
            unique_genders = ['Male', 'Female']
            unique_education = ["Bachelor's", "Master's", "PhD"]
            unique_job_titles = ['Software Engineer', 'Data Analyst', 'Manager']

        # Create input form
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### 👤 Employee Information")
            age = st.slider("Age", min_value=18, max_value=70, value=30, help="Employee's age in years")
            gender = st.selectbox("Gender", unique_genders, help="Employee's gender")
            education = st.selectbox("Education Level", unique_education, help="Highest education level")

        with col2:
            st.markdown("### 💼 Professional Details")
            job_title = st.selectbox("Job Title", unique_job_titles, help="Current job position")
            experience = st.slider("Years of Experience", min_value=0, max_value=40, value=5,
                                   help="Total years of professional experience")

        # Prediction button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🚀 Predict Salary", key="predict_btn"):
                with st.spinner("🤖 Analyzing employee profile..."):
                    prediction, conf_lower, conf_upper = make_prediction(
                        model, encoders, age, gender, education, job_title, experience
                    )

                    if prediction is not None:
                        # Display prediction result
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h2>💰 Predicted Salary</h2>
                            <div class="prediction-amount">${prediction:,.0f}</div>
                            <p>Confidence Range: ${conf_lower:,.0f} - ${conf_upper:,.0f}</p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Additional insights
                        st.markdown("### 📊 Prediction Insights")

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            # Compare with similar profiles
                            if sample_data is not None:
                                similar_profiles = sample_data[
                                    (sample_data['Education Level'] == education) &
                                    (sample_data['Gender'] == gender) &
                                    (abs(sample_data['Age'] - age) <= 5)
                                    ]
                                if len(similar_profiles) > 0:
                                    avg_similar = similar_profiles['Salary'].mean()
                                    difference = prediction - avg_similar
                                    st.metric(
                                        "vs Similar Profiles",
                                        f"${prediction:,.0f}",
                                        f"${difference:,.0f} ({(difference / avg_similar) * 100:+.1f}%)"
                                    )
                                else:
                                    st.metric("Predicted Salary", f"${prediction:,.0f}")
                            else:
                                st.metric("Predicted Salary", f"${prediction:,.0f}")

                        with col2:
                            if sample_data is not None:
                                percentile = (sample_data['Salary'] < prediction).mean() * 100
                                st.metric("Salary Percentile", f"{percentile:.0f}%",
                                          help="Percentage of employees earning less than this prediction")

                        with col3:
                            confidence_range = conf_upper - conf_lower
                            st.metric("Prediction Range", f"±${confidence_range / 2:,.0f}",
                                      help="80% confidence interval range")

                        # Feature contribution
                        st.markdown("### 🎯 What Influences This Salary?")
                        importance_fig = create_feature_importance_chart(model)
                        st.plotly_chart(importance_fig, use_container_width=True)

        # Bulk prediction section
        st.markdown("---")
        st.markdown("### 📋 Bulk Predictions")
        st.info("Upload a CSV file with employee data to predict multiple salaries at once")

        uploaded_file = st.file_uploader("Choose CSV file", type="csv",
                                         help="CSV should have columns: Age, Gender, Education Level, Job Title, Years of Experience")

        if uploaded_file is not None:
            try:
                bulk_df = pd.read_csv(uploaded_file)
                st.write("📄 Preview of uploaded data:")
                st.dataframe(bulk_df.head(), use_container_width=True)

                if st.button("🚀 Generate Bulk Predictions"):
                    with st.spinner("Generating predictions..."):
                        predictions = []
                        for _, row in bulk_df.iterrows():
                            pred, _, _ = make_prediction(
                                model, encoders, row['Age'], row['Gender'],
                                row['Education Level'], row['Job Title'],
                                row['Years of Experience']
                            )
                            predictions.append(pred if pred is not None else 0)

                        bulk_df['Predicted_Salary'] = predictions

                        st.success("✅ Predictions generated successfully!")
                        st.dataframe(bulk_df, use_container_width=True)

                        # Download button
                        csv = bulk_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name="salary_predictions.csv",
                            mime="text/csv"
                        )

            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    # DATA ANALYSIS PAGE
    elif page == "📊 Data Analysis":
        st.markdown('<h2 class="sub-title">📊 Dataset Analysis</h2>', unsafe_allow_html=True)

        if sample_data is None:
            st.warning("📄 Sample data not available for analysis")
            return

        # Dataset overview
        st.markdown("### 📋 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Records", len(sample_data))
        with col2:
            st.metric("Average Salary", f"${sample_data['Salary'].mean():,.0f}")
        with col3:
            st.metric("Median Salary", f"${sample_data['Salary'].median():,.0f}")
        with col4:
            st.metric("Salary Range", f"${sample_data['Salary'].max() - sample_data['Salary'].min():,.0f}")

        # Visualizations
        fig1, fig2, fig3 = create_salary_analysis_charts(sample_data)

        if fig1 and fig2 and fig3:
            st.plotly_chart(fig1, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig2, use_container_width=True)
            with col2:
                st.plotly_chart(fig3, use_container_width=True)

        # Detailed statistics
        st.markdown("### 📈 Detailed Statistics")

        tab1, tab2, tab3 = st.tabs(["By Education", "By Gender", "By Job Title"])

        with tab1:
            edu_stats = sample_data.groupby('Education Level')['Salary'].agg(['count', 'mean', 'median', 'std']).round(
                2)
            edu_stats.columns = ['Count', 'Average', 'Median', 'Std Dev']
            st.dataframe(edu_stats, use_container_width=True)

        with tab2:
            gender_stats = sample_data.groupby('Gender')['Salary'].agg(['count', 'mean', 'median', 'std']).round(2)
            gender_stats.columns = ['Count', 'Average', 'Median', 'Std Dev']
            st.dataframe(gender_stats, use_container_width=True)

        with tab3:
            job_stats = sample_data.groupby('Job Title')['Salary'].agg(['count', 'mean', 'median']).round(2)
            job_stats = job_stats.sort_values('mean', ascending=False).head(10)
            job_stats.columns = ['Count', 'Average', 'Median']
            st.dataframe(job_stats, use_container_width=True)

    # MODEL INFO PAGE
    elif page == "🤖 Model Info":
        st.markdown('<h2 class="sub-title">🤖 Model Information</h2>', unsafe_allow_html=True)

        # Model details
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("""
            <div class="info-card">
                <h3>🌳 Random Forest Regressor</h3>
                <p><strong>Algorithm Type:</strong> Ensemble Learning</p>
                <p><strong>Number of Trees:</strong> 100</p>
                <p><strong>Max Depth:</strong> 10</p>
                <p><strong>Min Samples Split:</strong> 5</p>
                <p><strong>Min Samples Leaf:</strong> 2</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="info-card">
                <h3>📊 How it Works</h3>
                <p>Random Forest combines multiple decision trees to make predictions:</p>
                <ol>
                    <li>Creates 100 different decision trees</li>
                    <li>Each tree trains on random data samples</li>
                    <li>Each tree uses random feature subsets</li>
                    <li>Final prediction = average of all trees</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)

        # Feature importance
        st.markdown("### 🎯 Feature Importance")
        importance_fig = create_feature_importance_chart(model)
        st.plotly_chart(importance_fig, use_container_width=True)

        # Model advantages
        st.markdown("### ✅ Why Random Forest?")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **🎯 Advantages:**
            - High accuracy and robust predictions
            - Handles both numerical and categorical data
            - Provides feature importance rankings
            - Resistant to overfitting
            - Works well with missing values
            - No need for feature scaling
            """)

        with col2:
            st.markdown("""
            **⚙️ Technical Details:**
            - Ensemble of 100 decision trees
            - Uses bootstrap aggregating (bagging)
            - Each tree sees random subset of features
            - Reduces variance through averaging
            - Provides confidence intervals
            - Parallel processing capable
            """)

        # Performance metrics (if available)
        st.markdown("### 📈 Model Performance")
        st.info("""
        🔍 **Model Evaluation:**
        The model was trained on 80% of the data and tested on 20%. 
        Performance metrics were calculated during training and saved with the model.
        Run the training script to see detailed performance metrics.
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>🤖 <strong>Employee Salary Predictor</strong> | Built with Streamlit & Random Forest</p>
        <p>📊 Accurate • ⚡ Fast • 🎯 Reliable</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
