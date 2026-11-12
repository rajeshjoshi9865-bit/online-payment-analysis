
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import json
from pathlib import Path

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Payment Insights",
    page_icon="💳",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "payments_features.csv"
MODEL_PATH = BASE_DIR / "models" / "fraud_model.joblib"
CONFIG_PATH = BASE_DIR / "models" / "model_config.json"

# ---------------- LOAD DATA AND MODEL ----------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

df = load_data()
model = load_model()

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

FEATURES = config["features"]
THRESHOLD = config.get("threshold", 0.5)

# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>
.stApp {
    background-color: #f4f7fb;
}
[data-testid="stSidebar"] {
    background-color: #10243A;
}
[data-testid="stSidebar"] * {
    color: white;
}
.metric-card {
    background: white;
    padding: 20px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- NAVIGATION ----------------

st.sidebar.title("💳 PAYMENT INSIGHTS")
st.sidebar.caption("Online Payment Analysis")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Home",
        "Data Analysis",
        "Behaviour Insights",
        "Fraud Detection",
        "Fraud Prediction",
        "About Project"
    ]
)

# ---------------- HOME ----------------

if page == "Home":

    st.title("Online Payment Behaviour Analysis")
    st.subheader("Understand. Analyse. Predict.")

    st.write(
        "An interactive platform to analyse online payment behaviour "
        "and identify potentially fraudulent transactions using data science."
    )

    total = len(df)
    fraud = int(df["isFraud"].sum())
    genuine = total - fraud
    fraud_rate = fraud / total * 100

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Transactions", f"{total:,}")
    c2.metric("Genuine Transactions", f"{genuine:,}")
    c3.metric("Fraud Cases", f"{fraud:,}")
    c4.metric("Fraud Rate", f"{fraud_rate:.3f}%")

    st.divider()

    st.subheader("Project Overview")

    st.write("""
    This project analyses online payment patterns, explores suspicious
    transaction behaviour, and applies machine learning to predict
    potentially fraudulent transactions.
    """)

    st.info("Use the navigation menu on the left to explore the website.")

# ---------------- DATA ANALYSIS ----------------

elif page == "Data Analysis":

    st.title("📊 Data Analysis")

    payment_types = ["All"] + sorted(df["type"].dropna().unique().tolist())

    selected_type = st.selectbox("Select Payment Type", payment_types)

    filtered = df.copy()

    if selected_type != "All":
        filtered = filtered[filtered["type"] == selected_type]

    st.metric("Transactions in Selection", f"{len(filtered):,}")

    type_data = filtered.groupby("type").size().reset_index(name="Count")

    fig1 = px.bar(
        type_data,
        x="type",
        y="Count",
        color="type",
        title="Transactions by Payment Type"
    )
    st.plotly_chart(fig1, use_container_width=True)

    hour_data = filtered.groupby("hour").size().reset_index(name="Count")

    fig2 = px.line(
        hour_data,
        x="hour",
        y="Count",
        markers=True,
        title="Transaction Activity by Hour"
    )
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.histogram(
        filtered,
        x="amount",
        color="isFraud",
        nbins=40,
        title="Transaction Amount Distribution",
        labels={"isFraud": "Fraud (0 = No, 1 = Yes)"}
    )
    st.plotly_chart(fig3, use_container_width=True)

# ---------------- BEHAVIOUR INSIGHTS ----------------

elif page == "Behaviour Insights":

    st.title("🧠 Payment Behaviour Insights")

    time_data = (
        df.groupby("time_band")
        .agg(
            Transactions=("isFraud", "size"),
            Fraud_Cases=("isFraud", "sum")
        )
        .reset_index()
    )

    time_data["Fraud Rate (%)"] = (
        time_data["Fraud_Cases"] / time_data["Transactions"] * 100
    )

    fig = px.bar(
        time_data,
        x="time_band",
        y="Fraud Rate (%)",
        color="time_band",
        title="Fraud Rate by Time of Day"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Account Balance Behaviour")

    balance_data = (
        df.groupby("isFraud")["drains_account"]
        .mean()
        .reset_index()
    )

    balance_data["drains_account"] *= 100
    balance_data["Category"] = balance_data["isFraud"].map(
        {0: "Genuine", 1: "Fraud"}
    )

    fig2 = px.bar(
        balance_data,
        x="Category",
        y="drains_account",
        color="Category",
        title="Percentage of Transactions Draining Sender Balance",
        labels={"drains_account": "Percentage (%)"}
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Payment Type Behaviour")

    fraud_type = (
        df.groupby("type")["isFraud"]
        .mean()
        .mul(100)
        .reset_index(name="Fraud Rate (%)")
    )

    fig3 = px.bar(
        fraud_type,
        x="type",
        y="Fraud Rate (%)",
        color="type",
        title="Fraud Rate by Payment Type"
    )
    st.plotly_chart(fig3, use_container_width=True)

# ---------------- FRAUD DETECTION ----------------

elif page == "Fraud Detection":

    st.title("🛡️ Fraud Detection")

    fraud_data = pd.DataFrame({
        "Transaction Category": ["Genuine", "Fraud"],
        "Count": [
            int((df["isFraud"] == 0).sum()),
            int((df["isFraud"] == 1).sum())
        ]
    })

    fig1 = px.pie(
        fraud_data,
        names="Transaction Category",
        values="Count",
        hole=0.5,
        title="Genuine vs Fraudulent Transactions"
    )
    st.plotly_chart(fig1, use_container_width=True)

    fraud_type = (
        df.groupby("type")["isFraud"]
        .sum()
        .reset_index(name="Fraud Cases")
    )

    fig2 = px.bar(
        fraud_type,
        x="type",
        y="Fraud Cases",
        color="Fraud Cases",
        title="Fraud Cases by Payment Type"
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Fraud by Time of Day")

    time_data = (
        df.groupby("time_band")
        .agg(
            Transactions=("isFraud", "size"),
            Fraud_Cases=("isFraud", "sum")
        )
        .reset_index()
    )

    time_data["Fraud Rate (%)"] = (
        time_data["Fraud_Cases"] / time_data["Transactions"] * 100
    )

    fig3 = px.bar(
        time_data,
        x="time_band",
        y="Fraud Rate (%)",
        title="Fraud Rate by Time of Day"
    )
    st.plotly_chart(fig3, use_container_width=True)

# ---------------- FRAUD PREDICTION ----------------

elif page == "Fraud Prediction":

    st.title("🔍 Fraud Prediction")
    st.write(
        "Enter transaction details below. The trained Random Forest model "
        "will estimate the probability of fraud."
    )

    st.info(
        "This is an academic demonstration using a historical sample dataset. "
        "It is not connected to a real bank or live payment system."
    )

    with st.form("prediction_form"):

        payment_type = st.selectbox(
            "Payment Type",
            ["TRANSFER", "CASH_OUT"]
        )

        amount = st.number_input(
            "Transaction Amount",
            min_value=1.0,
            value=10000.0
        )

        old_sender = st.number_input(
            "Sender Balance Before Transaction",
            min_value=0.0,
            value=15000.0
        )

        new_sender = st.number_input(
            "Sender Balance After Transaction",
            min_value=0.0,
            value=5000.0
        )

        old_receiver = st.number_input(
            "Receiver Balance Before Transaction",
            min_value=0.0,
            value=0.0
        )

        new_receiver = st.number_input(
            "Receiver Balance After Transaction",
            min_value=0.0,
            value=10000.0
        )

        hour = st.slider(
            "Transaction Hour",
            min_value=0,
            max_value=23,
            value=14
        )

        receiver_count = st.number_input(
            "Estimated Previous Receiver Transactions",
            min_value=1,
            value=5
        )

        receiver_merchant = st.checkbox(
            "Receiver is a merchant",
            value=False
        )

        submitted = st.form_submit_button(
            "Predict Transaction"
        )

    if submitted:

        if new_sender > old_sender + 0.01:
            st.error("Sender's closing balance cannot exceed the opening balance in this demo.")
        elif amount > old_sender + 0.01:
            st.error("Transaction amount cannot exceed the sender's opening balance in this demo.")
        else:

            usage_ratio = (
                min(amount / old_sender, 10)
                if old_sender > 0 else 0
            )

            drains_account = int(
                old_sender > 0
                and new_sender == 0
                and abs(amount - old_sender) <= 1
            )

            input_values = {
                "type_TRANSFER": int(payment_type == "TRANSFER"),
                "log_amount": np.log10(max(amount, 1)),
                "log_old_bal_orig": np.log10(old_sender + 1),
                "log_old_bal_dest": np.log10(old_receiver + 1),
                "usage_ratio": usage_ratio,
                "drains_account": drains_account,
                "orig_zero_before": int(old_sender == 0),
                "dest_zero_both": int(
                    old_receiver == 0
                    and new_receiver == 0
                    and not receiver_merchant
                ),
                "is_night": int(hour <= 5),
                "hour": hour,
                "log_dest_count": np.log10(max(receiver_count, 1))
            }

            input_df = pd.DataFrame(
                [[input_values[f] for f in FEATURES]],
                columns=FEATURES
            )

            probability = model.predict_proba(input_df)[0][1]
            prediction = int(probability >= THRESHOLD)

            st.divider()

            st.subheader("Prediction Result")

            col1, col2 = st.columns(2)

            col1.metric(
                "Estimated Fraud Probability",
                f"{probability * 100:.2f}%"
            )

            col2.metric(
                "Model Decision",
                "Potential Fraud" if prediction else "Likely Genuine"
            )

            if prediction:
                st.error(
                    "⚠️ The model has classified this transaction as potentially fraudulent."
                )
            else:
                st.success(
                    "✅ The model has classified this transaction as likely genuine."
                )

            st.caption(
                f"Decision threshold: {THRESHOLD:.2f}. "
                "This result is an estimate, not a guarantee."
            )

# ---------------- ABOUT PROJECT ----------------

elif page == "About Project":

    st.title("About the Project")

    st.subheader("Objective")

    st.write(
        "To analyse online payment behaviour, discover suspicious patterns, "
        "and build a machine learning model to predict fraudulent transactions."
    )

    st.subheader("Technologies Used")

    st.write("""
    - Python
    - Pandas and NumPy
    - Plotly
    - Scikit-learn
    - Random Forest
    - Streamlit
    """)

    st.subheader("Dataset Information")

    st.write(f"Total records: {len(df):,}")
    st.write(f"Total columns: {len(df.columns)}")

    st.subheader("Dataset Preview")

    st.dataframe(df.head(10), use_container_width=True)

    st.warning(
        "The project uses a historical sample dataset. "
        "The model is a demonstration and should not be used to approve "
        "or reject real financial transactions."
    )
