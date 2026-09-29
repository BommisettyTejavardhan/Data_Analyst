import os
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Hydration Level Prediction",
    page_icon="💧",
    layout="centered"
)

# ----------------------------
# Smart Dataset Generator
# ----------------------------
def create_smart_dataset(file_path):
    """Generates a mathematically accurate dataset based on biological water needs."""
    np.random.seed(42)
    n_samples = 1500
    
    ages = np.random.randint(18, 70, n_samples)
    genders = np.random.choice(['Male', 'Female'], n_samples)
    weights = np.random.randint(40, 120, n_samples)
    activities = np.random.choice(['Low', 'Moderate', 'High'], n_samples)
    weathers = np.random.choice(['Cold', 'Warm', 'Hot'], n_samples)
    water_intakes = np.round(np.random.uniform(0.5, 6.0, n_samples), 1)
    
    labels = []
    for weight, activity, weather, intake in zip(weights, activities, weathers, water_intakes):
        # Base biological need: ~35ml per kg of body weight
        target_water = weight * 0.035 
        
        # Activity modifiers
        if activity == 'High': target_water += 1.0
        elif activity == 'Moderate': target_water += 0.5
            
        # Weather modifiers
        if weather == 'Hot': target_water += 1.0
        elif weather == 'Warm': target_water += 0.5
            
        # Determine label (giving a 10% leeway margin)
        if intake >= (target_water * 0.90):
            labels.append('Good')
        else:
            labels.append('Poor')
            
    df = pd.DataFrame({
        "Age": ages,
        "Gender": genders,
        "Weight (kg)": weights,
        "Daily Water Intake (liters)": water_intakes,
        "Physical Activity Level": activities,
        "Weather": weathers,
        "Hydration Level": labels
    })
    
    df.to_csv(file_path, index=False)
    return df

# ----------------------------
# Caching Data & Model
# ----------------------------
@st.cache_data
def load_data():
    # USING A NEW FILENAME TO BYPASS THE OLD BAD DATA
    file_path = "Smart_Water_Intake.csv"
    if not os.path.exists(file_path):
        return create_smart_dataset(file_path)
    return pd.read_csv(file_path)

@st.cache_resource
def get_trained_model(df):
    df_encoded = df.copy()
    encoders = {}
    
    categorical_cols = ["Gender", "Physical Activity Level", "Weather", "Hydration Level"]
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        encoders[col] = le
        
    X = df_encoded.drop("Hydration Level", axis=1)
    y = df_encoded["Hydration Level"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = accuracy_score(y_test, model.predict(X_test))
    
    return model, encoders, accuracy, X

# Load data and train model
df = load_data()
model, encoders, accuracy, X_features = get_trained_model(df)

# ----------------------------
# Sidebar Navigation
# ----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["About", "Dataset", "Model Performance", "Prediction"])

# ----------------------------
# 1. About Page
# ----------------------------
if page == "About":
    st.title("💧 About This Project")
    st.markdown("""
    Welcome to the **Hydration Level Prediction** app!
    
    Maintaining proper hydration is critical for overall health. This machine learning tool evaluates daily habits and environmental factors to predict whether your current hydration level is **Good** or **Poor**.
    
    ### How it works:
    1. **Data:** We analyze metrics like Age, Weight, Daily Water Intake, Physical Activity, and Weather conditions.
    2. **Machine Learning:** A Random Forest Classifier identifies patterns in the data to estimate hydration sufficiency.
    3. **Recommendations:** Based on the prediction, the app provides actionable tips to maintain or improve your water intake.
    """)

# ----------------------------
# 2. Dataset Page
# ----------------------------
elif page == "Dataset":
    st.title("📊 Dataset Overview")
    st.write("Explore the data used to train the machine learning model.")
    
    st.subheader("Raw Data Preview")
    st.dataframe(df.head(10))
    
    st.subheader("Dataset Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Rows", df.shape[0])
    with col2:
        st.metric("Total Features", df.shape[1] - 1)
        
    st.write("### Numerical Summary")
    st.dataframe(df.describe())
    
    st.write("### Outcome Distribution")
    st.bar_chart(df['Hydration Level'].value_counts())

# ----------------------------
# 3. Model Performance Page
# ----------------------------
elif page == "Model Performance":
    st.title("⚙️ Model Performance")
    
    st.subheader("Accuracy")
    st.metric(label="Testing Accuracy", value=f"{accuracy * 100:.2f}%")
    st.write("The Random Forest model was trained on 80% of the data and tested on the remaining 20%.")
    
    st.markdown("---")
    
    st.subheader("Feature Importance")
    st.write("This chart shows which factors have the biggest impact on predicting a person's hydration level:")
    
    importances = model.feature_importances_
    feature_names = X_features.columns
    
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
    
    st.bar_chart(data=importance_df.set_index("Feature"))

# ----------------------------
# 4. Prediction Page
# ----------------------------
elif page == "Prediction":
    st.title("🔮 Hydration Prediction")
    st.write("Enter your daily details below to check your hydration status.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 18, 70, 25)
        gender = st.selectbox("Gender", encoders["Gender"].classes_)
        weight = st.slider("Weight (kg)", 40, 120, 70)
        
    with col2:
        water = st.slider("Daily Water Intake (liters)", 0.5, 6.0, 2.5, 0.1)
        activity = st.selectbox("Physical Activity Level", encoders["Physical Activity Level"].classes_)
        weather = st.selectbox("Weather", encoders["Weather"].classes_)

    if st.button("Predict Hydration", type="primary"):
        
        input_df = pd.DataFrame({
            "Age": [age],
            "Gender": [encoders["Gender"].transform([gender])[0]],
            "Weight (kg)": [weight],
            "Daily Water Intake (liters)": [water],
            "Physical Activity Level": [encoders["Physical Activity Level"].transform([activity])[0]],
            "Weather": [encoders["Weather"].transform([weather])[0]]
        })

        # Force input_df to match the training column order
        input_df = input_df[X_features.columns]

        prediction = model.predict(input_df)[0]
        result = encoders["Hydration Level"].inverse_transform([prediction])[0]

        st.markdown("---")
        st.subheader("Prediction Result")

        if result == "Good":
            st.success("💧 Hydration Level : GOOD")
            st.balloons()
            st.info("""
            ### Recommendations
            ✅ Continue drinking enough water.  
            ✅ Maintain your current lifestyle.  
            ✅ Eat water-rich fruits and vegetables.  
            ✅ Stay hydrated during exercise.
            """)
        else:
            st.error("⚠️ Hydration Level : POOR")
            st.warning("""
            ### Recommendations
            🚰 Drink more water! You are likely dehydrated.  
            🥤 Consume electrolyte drinks if doing heavy activity.  
            🍉 Eat fruits with high water content (watermelon, cucumber).  
            🏃 Avoid prolonged activity without hydration.
            """)

# ----------------------------
# Global Footer
# ----------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("Developed using Python | Streamlit | Machine Learning")