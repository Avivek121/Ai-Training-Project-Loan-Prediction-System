import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# --- THEME & STYLING ---
st.set_page_config(page_title="Indian Bank Loan Prediction", layout="wide")

def apply_indian_bank_theme():
    st.markdown("""
        <style>
        .stApp {
            background: radial-gradient(circle at top right, #0a192f, #020c1b);
            color: #ccd6f6;
        }
        @keyframes drift {
            0% { transform: translateY(100vh) translateX(0); opacity: 0; }
            50% { opacity: 0.3; }
            100% { transform: translateY(-10vh) translateX(20px); opacity: 0; }
        }
        .money-bg {
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            z-index: -1; pointer-events: none;
        }
        .coin {
            position: absolute; bottom: -10%; font-size: 20px;
            animation: drift 15s linear infinite;
        }
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(100, 255, 218, 0.2);
            padding: 15px; border-radius: 12px;
        }
        .stButton>button {
            background-color: #64ffda; color: #020c1b; font-weight: bold;
            border-radius: 8px; border: none; transition: 0.3s;
        }
        .stButton>button:hover { background-color: #48c9b0; color: white; }
        .result-card {
            padding: 1.5rem; border-radius: 12px;
            border-left: 8px solid; margin-top: 10px;
            background: rgba(255, 255, 255, 0.07);
        }
        </style>
        <div class="money-bg">
            <div class="coin" style="left: 10%; animation-delay: 0s;">💵</div>
            <div class="coin" style="left: 30%; animation-delay: 4s;">💰</div>
            <div class="coin" style="left: 60%; animation-delay: 2s;">💸</div>
            <div class="coin" style="left: 85%; animation-delay: 7s;">🏦</div>
        </div>
    """, unsafe_allow_html=True)

apply_indian_bank_theme()

# --- ML ENGINE ---
@st.cache_resource
def load_and_train_all():
    df = pd.read_csv('loan_sanction_train.csv')
    df.drop("Loan_ID", axis=1, errors='ignore', inplace=True)
    
    # Preprocessing (based on your notebook)
    for col in ['LoanAmount', 'Loan_Amount_Term']:
        df[col] = df[col].fillna(df[col].median())
    for col in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # Feature Engineering
    df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    
    # Encoding
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])
        
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train)
    X_test_sc = sc.transform(X_test)
    
    # Models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(),
        "Gradient Boosting": GradientBoostingClassifier()
    }
    
    accuracies = {}
    trained_models = {}
    for name, m in models.items():
        m.fit(X_train_sc, y_train)
        accuracies[name] = accuracy_score(y_test, m.predict(X_test_sc))
        trained_models[name] = m
        
    return trained_models, sc, X.columns, accuracies, df

models, sc, feature_names, accuracies, raw_processed_df = load_and_train_all()

# --- SIDEBAR ---
st.sidebar.image("https://img.icons8.com/fluency/96/bank.png", width=80)
st.sidebar.title("System Control")
model_choice = st.sidebar.selectbox("Active AI Engine", list(models.keys()))
st.sidebar.divider()
st.sidebar.metric(f"{model_choice} Accuracy", f"{accuracies[model_choice]:.2%}")

# --- DASHBOARD LAYOUT ---
st.title("🏦 Indian Bank Loan Prediction")
st.write(f"Advanced Credit Assessment Module | Engine: **{model_choice}**")

# Section 1: Input Form
with st.container():
    st.subheader("📝 Applicant Credit Profile")
    c1, c2, c3 = st.columns(3)
    with c1:
        name = st.text_input("Full Name", "Arun Kumar")
        income = st.number_input("Monthly Applicant Income (₹)", 5000)
        co_income = st.number_input("Co-Applicant Income (₹)", 0)
    with c2:
        loan_amt = st.number_input("Loan Amount (Thousands ₹)", 150)
        term = st.selectbox("Term in Days", [360, 180, 240, 120])
        credit = st.selectbox("Credit History Score", [1.0, 0.0])
    with c3:
        area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
        edu = st.selectbox("Education", ["Graduate", "Not Graduate"])
        emp = st.selectbox("Self Employed", ["Yes", "No"])

# Section 2: Analysis & Result
if st.button("EXECUTE CREDIT ANALYSIS", use_container_width=True):
    # Prepare input to match model training
    input_data = pd.DataFrame([[
        1, 1, 0, # Dummies for Gender, Married, Dependents
        0 if edu=="Graduate" else 1,
        1 if emp=="Yes" else 0,
        income, co_income, loan_amt, term, credit,
        0 if area=="Rural" else 1 if area=="Semiurban" else 2,
        income + co_income
    ]], columns=feature_names)
    
    scaled_input = sc.transform(input_data)
    prob = models[model_choice].predict_proba(scaled_input)[0][1]
    prediction = models[model_choice].predict(scaled_input)[0]
    
    st.divider()
    res1, res2 = st.columns([1, 1.5])
    
    with res1:
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number", value = prob * 100,
            title = {'text': "Eligibility Score", 'font': {'size': 20, 'color': "white"}},
            gauge = {
                'axis': {'range': [0, 100], 'tickcolor': "white"},
                'bar': {'color': "#64ffda"},
                'steps': [
                    {'range': [0, 40], 'color': "#ff4b4b"},
                    {'range': [40, 75], 'color': "#ffa500"},
                    {'range': [75, 100], 'color': "#00cc96"}]}
        ))
        fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "white"}, height=350)
        st.plotly_chart(fig_gauge, use_container_width=True)

    with res2:
        if prediction == 1:
            st.markdown(f"""
                <div class="result-card" style="border-left-color: #00cc96;">
                    <h2 style="color: #00cc96; margin-top:0;">✅ ELIGIBILITY: APPROVED</h2>
                    <p><b>Analysis:</b> {name} meets the credit-worthiness criteria for the requested amount.</p>
                    <p><b>Model Confidence:</b> {prob:.2%}</p>
                    <p><i>Recommended for standard disbursement protocol.</i></p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="result-card" style="border-left-color: #ff4b4b;">
                    <h2 style="color: #ff4b4b; margin-top:0;">❌ ELIGIBILITY: REJECTED</h2>
                    <p><b>Analysis:</b> Applicant does not meet the minimum safety threshold for risk.</p>
                    <p><b>System Risk Score:</b> {(1-prob):.2%}</p>
                    <p style="color: #8892b0;"><i>Consider a lower loan amount or adding a co-applicant with higher income.</i></p>
                </div>
            """, unsafe_allow_html=True)

# Section 3: Graphs & Accuracy
st.divider()
st.subheader("📊 Model Performance & Insights")
g1, g2 = st.columns(2)

with g1:
    st.write("**Model Accuracy Comparison**")
    acc_df = pd.DataFrame(list(accuracies.items()), columns=['Model', 'Accuracy'])
    fig_acc = px.bar(acc_df, x='Model', y='Accuracy', color='Accuracy', 
                     color_continuous_scale='Tealgrn', template="plotly_dark")
    fig_acc.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_acc, use_container_width=True)

with g2:
    if hasattr(models[model_choice], 'feature_importances_'):
        st.write(f"**{model_choice} Factor Importance**")
        importances = models[model_choice].feature_importances_
        feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=True)
        fig_feat = px.bar(feat_df, y='Feature', x='Importance', orientation='h', template="plotly_dark")
        fig_feat.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_feat, use_container_width=True)
    else:
        st.info("Logistic Regression uses linear coefficients. Switch to Random Forest to see Feature Importance.")

# Section 4: Data Explorer
with st.expander("📂 TRAINING DATA AUDIT (CSV)"):
    st.dataframe(raw_processed_df.head(100), use_container_width=True)