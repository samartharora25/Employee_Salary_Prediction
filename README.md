# Employee_Salary_Prediction
BY SAMARTH ARORA
https://1711b6b91433.ngrok-free.app/


📁 Project Structure
salary_predictor/
├── salary_predictor.py      # Training script
├── app.py                   # Streamlit frontend
├── requirements.txt         # Dependencies
├── Salary Data.csv         # Your dataset
├── random_forest_model.pkl # Generated model file
├── label_encoders.pkl      # Generated encoders
├── salary_predictor_model.pkl # Complete model package
└── *.png                   # Generated plots

The app has 4 main sections:
🏠 Home Page

Welcome message and overview
Quick dataset statistics
Navigation guide

🔮 Predict Salary

Individual Predictions: Enter employee details
Bulk Predictions: Upload CSV for multiple predictions
Insights: Compare with similar profiles
Feature Importance: See what drives salary predictions

📊 Data Analysis

Dataset overview and statistics
Interactive charts and visualizations
Salary breakdowns by demographics

🤖 Model Info

Random Forest algorithm details
Feature importance rankings
Technical specifications

📊 Understanding the Results
Model Performance Metrics:

R² Score: Percentage of variance explained (higher = better)

0.9+: Excellent
0.8-0.9: Good
0.7-0.8: Decent
<0.7: Needs improvement


RMSE: Average prediction error in dollars (lower = better)
MAE: Mean absolute error in dollars (lower = better)

Feature Importance:
Shows which factors most influence salary predictions:

Job Title: Usually most important (40-50%)
Experience: Second most important (20-30%)
Education: Moderate importance (10-20%)
Age: Lower importance (5-15%)
Gender: Least important (0-10%)
