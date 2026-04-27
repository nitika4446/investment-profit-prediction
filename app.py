import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os


# -----------------------------------
# Streamlit Page Config
# -----------------------------------
st.set_page_config(
    page_title="Investment Profit Predictor",
    layout="wide"
)

st.title("📈 Investment Profit Prediction Dashboard")
st.write("Predict company profit based on investment spending.")

# -----------------------------------
# File Paths
# -----------------------------------
DATA_FILE = "Investment.csv"
MODEL_FILE = "investment_model.pkl"

# -----------------------------------
# Train Model Function
# -----------------------------------
def train_model():
    df = pd.read_csv(DATA_FILE)

    # Features and target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Convert categorical state column
    X = pd.get_dummies(X, drop_first=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Accuracy
    score = r2_score(y_test, y_pred)

    # Save model + columns
    with open(MODEL_FILE, "wb") as f:
        pickle.dump((model, X.columns), f)

    return score


# -----------------------------------
# Load or Train Model
# -----------------------------------
if not os.path.exists(MODEL_FILE):
    score = train_model()
else:
    score = None

with open(MODEL_FILE, "rb") as f:
    model, columns = pickle.load(f)

# -----------------------------------
# Sidebar Inputs
# -----------------------------------
st.sidebar.header("Enter Investment Details")

marketing = st.sidebar.number_input(
    "Digital Marketing Spend",
    min_value=0.0
)

promotion = st.sidebar.number_input(
    "Promotion Spend",
    min_value=0.0
)

research = st.sidebar.number_input(
    "Research Spend",
    min_value=0.0
)

state = st.sidebar.selectbox(
    "Select State",
    ["New York", "California", "Florida"]
)

# -----------------------------------
# Create Input Data
# -----------------------------------
input_data = pd.DataFrame({
    "Digital Marketing Spend": [marketing],
    "Promotion Spend": [promotion],
    "Research Spend": [research],
    "State": [state]
})

# Apply same encoding
input_data = pd.get_dummies(input_data)

# Match training columns
for col in columns:
    if col not in input_data.columns:
        input_data[col] = 0

# Maintain same order
input_data = input_data[columns]

# -----------------------------------
# Prediction Button
# -----------------------------------
if st.button("Predict Profit"):
    prediction = model.predict(input_data)[0]

    st.success(f"Estimated Profit: ₹ {prediction:,.2f}")

    # Store history
    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append({
        "Marketing Spend": marketing,
        "Promotion Spend": promotion,
        "Research Spend": research,
        "State": state,
        "Predicted Profit": prediction
    })

# -----------------------------------
# Model Performance
# -----------------------------------
st.subheader("📊 Model Performance")

if score:
    st.write(f"Model R² Score: {score:.2f}")
else:
    st.write("Model loaded successfully.")

# -----------------------------------
# Prediction History
# -----------------------------------
st.subheader("🕒 Prediction History")

if "history" in st.session_state and len(st.session_state.history) > 0:
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df)
else:
    st.write("No predictions made yet.")

# -----------------------------------
# Footer
# -----------------------------------
st.markdown("---")
st.write("Built with ❤️ using Python, Scikit-learn, Streamlit & GitHub")






