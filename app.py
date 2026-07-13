
import streamlit as st
import pandas as pd
import joblib
import os
import requests
import plotly.graph_objects as go
import plotly.express as px

from streamlit_lottie import st_lottie
# ----------------------------
# Load Model Files
# ----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(
    os.path.join(BASE_DIR, "churn_model.pkl")
)

scaler = joblib.load(
    os.path.join(BASE_DIR, "scaler.pkl")
)

features = joblib.load(
    os.path.join(BASE_DIR, "features.pkl")
)


# ----------------------------
# Title
# ----------------------------
# ----------------------------
# PAGE CONFIG
# ----------------------------

st.set_page_config(
    page_title="Telecom Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

# ----------------------------
# CUSTOM CSS
# ----------------------------

st.markdown("""
<style>

/* Main background */
.stApp{
    background:#F4F8FC;
}

/* Gradient Header */
.header{
background: linear-gradient(90deg,#2563EB,#7C3AED);
padding:30px;
border-radius:18px;
text-align:center;
color:white;
box-shadow:0px 6px 18px rgba(0,0,0,0.25);
margin-bottom:25px;
}

.header h1{
font-size:42px;
margin-bottom:8px;
}

.header p{
font-size:18px;
opacity:0.9;
}

/* Sidebar */

section[data-testid="stSidebar"]{
background:#1E293B;
}

section[data-testid="stSidebar"] *{
color:white;
}

/* Cards */

.card{
background:white;
padding:18px;
border-radius:15px;
box-shadow:0px 5px 15px rgba(0,0,0,0.12);
margin-bottom:15px;
}

/* Button */

.stButton>button{
background:#2563EB;
color:white;
font-size:18px;
font-weight:bold;
border-radius:10px;
padding:10px 25px;
width:100%;
border:none;
}

.stButton>button:hover{
background:#1D4ED8;
}

</style>
""", unsafe_allow_html=True)


# ----------------------------
# HEADER
# ----------------------------

# ----------------------------
# HEADER
# ----------------------------

st.markdown("""

<div class="header">

<h1>📞 Telecom Customer Churn Prediction</h1>

<p>
Predict customer churn using Machine Learning (XGBoost)
</p>

</div>

""", unsafe_allow_html=True)


# ----------------------------
# LOTTIE ANIMATION
# ----------------------------

from streamlit_lottie import st_lottie
import requests


def load_lottie(url):
    r = requests.get(url)

    if r.status_code != 200:
        return None

    return r.json()


animation = load_lottie(
    "https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json"
)

st_lottie(
    animation,
    height=250,
    key="telecom_animation"
)

# ----------------------------
# SIDEBAR INPUTS
# ----------------------------

st.sidebar.title("📝 Customer Information")

st.sidebar.markdown("---")

gender = st.sidebar.selectbox(
    "Gender",
    ["Male","Female"]
)

senior = st.sidebar.selectbox(
    "Senior Citizen",
    [0,1]
)

partner = st.sidebar.selectbox(
    "Partner",
    ["Yes","No"]
)

dependents = st.sidebar.selectbox(
    "Dependents",
    ["Yes","No"]
)

tenure = st.sidebar.slider(
    "Tenure (Months)",
    0,
    72,
    12
)

phone = st.sidebar.selectbox(
    "Phone Service",
    ["Yes","No"]
)

multiple = st.sidebar.selectbox(
    "Multiple Lines",
    ["Yes","No","No phone service"]
)

internet = st.sidebar.selectbox(
    "Internet Service",
    ["DSL","Fiber optic","No"]
)

security = st.sidebar.selectbox(
    "Online Security",
    ["Yes","No","No internet service"]
)

backup = st.sidebar.selectbox(
    "Online Backup",
    ["Yes","No","No internet service"]
)

device = st.sidebar.selectbox(
    "Device Protection",
    ["Yes","No","No internet service"]
)

support = st.sidebar.selectbox(
    "Tech Support",
    ["Yes","No","No internet service"]
)

stream_tv = st.sidebar.selectbox(
    "Streaming TV",
    ["Yes","No","No internet service"]
)

stream_movie = st.sidebar.selectbox(
    "Streaming Movies",
    ["Yes","No","No internet service"]
)

contract = st.sidebar.selectbox(
    "Contract",
    ["Month-to-month","One year","Two year"]
)

paperless = st.sidebar.selectbox(
    "Paperless Billing",
    ["Yes","No"]
)

payment = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly = st.sidebar.slider(
    "Monthly Charges",
    0.0,
    150.0,
    70.0
)

total = st.sidebar.number_input(
    "Total Charges",
    min_value=0.0,
    value=1000.0
)

st.sidebar.markdown("---")
st.sidebar.success("Click **Predict Churn** to analyze the customer.")


# ----------------------------
# Preprocessing Function
# ----------------------------

def preprocess_input(data):
    """
    Convert user input into the same format used during model training.
    """

    # Convert dictionary to DataFrame
    df = pd.DataFrame([data])

    # One-Hot Encode categorical variables
    df = pd.get_dummies(df)

    # Add missing columns expected by the model
    for col in features:
        if col not in df.columns:
            df[col] = 0

    # Arrange columns in the same order as training
    df = df[features]

    # Scale only numerical features
    num_cols = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    df[num_cols] = scaler.transform(df[num_cols])

    return df

# ---------------------------------
# Prediction Dashboard
# ---------------------------------

if st.sidebar.button("🚀 Predict Churn", use_container_width=True):

    # Customer Input
    input_data = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": device,
        "TechSupport": support,
        "StreamingTV": stream_tv,
        "StreamingMovies": stream_movie,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }

    # ---------------------------------
    # Prediction
    # ---------------------------------

    final_input = preprocess_input(input_data)

    prediction = model.predict(final_input)

    probability = model.predict_proba(final_input)[0][1]

    prediction_text = "Churn" if prediction[0] == 1 else "Stay"

    if probability >= 0.7:
        risk = "High"
        risk_color = "#EF4444"

    elif probability >= 0.4:
        risk = "Medium"
        risk_color = "#F59E0B"

    else:
        risk = "Low"
        risk_color = "#22C55E"

        # ---------------------------------
    # Dashboard Cards
    # ---------------------------------

    st.markdown("""
    <h2 style="
    color:#000000;
    font-weight:700;
    margin-bottom:20px;
    ">
    📊 Prediction Dashboard
    </h2>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("""
        <div style="
        background:linear-gradient(135deg,#2563EB,#1D4ED8);
        padding:22px;
        border-radius:15px;
        color:white;
        text-align:center;">
        <h4>🤖 Model</h4>
        <h2>XGBoost</h2>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div style="
        background:linear-gradient(135deg,#7C3AED,#9333EA);
        padding:22px;
        border-radius:15px;
        color:white;
        text-align:center;">
        <h4>📈 Probability</h4>
        <h2>{probability:.2%}</h2>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div style="
        background:linear-gradient(135deg,#06B6D4,#0284C7);
        padding:22px;
        border-radius:15px;
        color:white;
        text-align:center;">
        <h4>🎯 Prediction</h4>
        <h2>{prediction_text}</h2>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div style="
        background:{risk_color};
        padding:22px;
        border-radius:15px;
        color:white;
        text-align:center;">
        <h4>⚠️ Risk</h4>
        <h2>{risk}</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
      # ---------------------------------
    # Prediction Result
    # ---------------------------------

    left, right = st.columns([2, 1])

    # LEFT PANEL
    with left:

        st.markdown("""
        <h2 style="
        color:#1E293B;
        font-weight:700;
        ">
        📊 Prediction Result
        </h2>
        """, unsafe_allow_html=True)

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={"suffix": "%"},
                title={"text": "Churn Probability"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#2563EB"},
                    "steps": [
                        {"range": [0, 40], "color": "#22C55E"},
                        {"range": [40, 70], "color": "#FACC15"},
                        {"range": [70, 100], "color": "#EF4444"}
                    ]
                }
            )
        )

        fig.update_layout(
            height=330,
            margin=dict(l=20, r=20, t=40, b=20)
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        if prediction[0] == 1:
            st.error("⚠️ Customer is likely to churn.")
        else:
            st.success("✅ Customer is likely to stay.")

        if risk == "High":
            st.error("🔴 High Risk Customer")
        elif risk == "Medium":
            st.warning("🟡 Medium Risk Customer")
        else:
            st.success("🟢 Low Risk Customer")

       # RIGHT PANEL
    with right:

        st.markdown("""
        <h2 style="
        color:#1E293B;
        font-weight:700;
        ">
        🤖 Model Performance
        </h2>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="
            background:white;
            padding:18px;
            border-radius:15px;
            box-shadow:0px 4px 12px rgba(0,0,0,0.15);
            margin-bottom:15px;
        ">

        <h4 style="color:#2563EB;">Algorithm</h4>
        <h2 style="color:#111827;">XGBoost</h2>

        <hr>

        <h4 style="color:#2563EB;">Accuracy</h4>
        <h2 style="color:#111827;">76.3%</h2>

        <hr>

        <h4 style="color:#2563EB;">ROC-AUC</h4>
        <h2 style="color:#111827;">0.81</h2>

        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="
            background:#EFF6FF;
            border-left:6px solid #2563EB;
            padding:15px;
            border-radius:12px;
            color:#111827;
        ">
        <h4>📘 Risk Guide</h4>

        🟢 <b>Low Risk</b> : 0–40%<br><br>

        🟡 <b>Medium Risk</b> : 40–70%<br><br>

        🔴 <b>High Risk</b> : 70–100%

        </div>
        """, unsafe_allow_html=True)

      # ---------------------------------
    # Customer Summary
    # ---------------------------------

    st.markdown("""
    <h2 style="
    color:#1E293B;
    font-weight:700;
    margin-bottom:15px;
    ">
    📋 Customer Summary
    </h2>
    """, unsafe_allow_html=True)

    summary = pd.DataFrame(
        list(input_data.items()),
        columns=["Feature", "Value"]
    )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")


    # ---------------------------------
    # Churn Probability Distribution
    # ---------------------------------

    st.markdown("""
    <h2 style="
    color:#1E293B;
    font-weight:700;
    margin-bottom:15px;
    ">
    🥧 Churn Probability Distribution
    </h2>
    """, unsafe_allow_html=True)

    pie_data = pd.DataFrame({
        "Status": ["Stay", "Churn"],
        "Probability": [
            (1 - probability) * 100,
            probability * 100
        ]
    })

    pie = px.pie(
        pie_data,
        names="Status",
        values="Probability",
        hole=0.60,
        color="Status",
        color_discrete_map={
            "Stay": "#22C55E",
            "Churn": "#EF4444"
        },
        template="plotly_white"
    )

    pie.update_traces(
        textinfo="percent+label",
        textfont=dict(
            size=16,
            color="black"
        ),
        marker=dict(
            line=dict(color="white", width=3)
        ),
        pull=[0,0.08]
    )

    pie.update_layout(
        title=dict(
            text="Prediction Distribution",
            x=0.5,
            font=dict(
                size=22,
                color="black"
            )
        ),
        font=dict(
            color="black",
            size=15
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(
            orientation="h",
            y=-0.15,
            x=0.25,
            font=dict(
                color="black",
                size=14
            )
        ),
        margin=dict(
            t=80,
            b=40,
            l=20,
            r=20
        ),
        height=500
    )

    st.plotly_chart(
        pie,
        use_container_width=True
    )

    st.markdown("---")


# ===========================================
# FOOTER
# ===========================================

st.markdown("""
<style>
.footer{
    margin-top:40px;
    padding:25px;
    border-radius:18px;
    background:linear-gradient(135deg,#2563EB,#7C3AED);
    color:white;
    text-align:center;
    box-shadow:0px 8px 25px rgba(0,0,0,0.25);
}

.footer h3{
    margin-bottom:10px;
    font-size:26px;
}

.footer p{
    font-size:16px;
    margin:5px;
}

.footer span{
    color:#FACC15;
    font-weight:bold;
}
</style>

<div class="footer">

<h3>📞 Telecom Customer Churn Prediction</h3>

<p>
Built with ❤️ using
<span>Python</span> •
<span>Streamlit</span> •
<span>XGBoost</span> •
<span>Plotly</span>
</p>

<p>
Developed by <span>Anjali Jamwal</span>
</p>

<p>
© 2026 All Rights Reserved
</p>

</div>

""", unsafe_allow_html=True)