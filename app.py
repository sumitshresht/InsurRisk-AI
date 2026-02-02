import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import shap
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
import streamlit.components.v1 as components
import os

# --- 1. FORCE THEME CONFIGURATION ---
def setup_app_config():
    config_dir = ".streamlit"
    config_path = os.path.join(config_dir, "config.toml")
    if not os.path.exists(config_dir): os.makedirs(config_dir)
    
    config_content = """
[theme]
base="light"
primaryColor="#00b894"
backgroundColor="#f0f2f6"
secondaryBackgroundColor="#ffffff"
textColor="#2d3436"
font="sans serif"
    """
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            f.write(config_content.strip())
        try: st.rerun()
        except: pass

setup_app_config()

# --- 2. APP SETUP ---
st.set_page_config(
    page_title="InsurRisk AI | Glass",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3. SESSION STATE MANAGEMENT (FIX FOR RESET) ---
# We initialize default values in session state if they don't exist
defaults = {
    'age': 32,
    'sex': 'male',
    'children': 1,
    'bmi': 24.5,
    'smoker': 'no',
    'region': 'southwest'
}

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

def reset_callback():
    for key, val in defaults.items():
        st.session_state[key] = val

# --- 4. CUSTOM CSS (FIX FOR SELECTBOX) ---
st.markdown("""
    <style>
    /* MAIN BACKGROUND */
    .stApp {
        background-color: #f0f2f6;
        background-image: 
            radial-gradient(at 10% 10%, rgba(108, 92, 231, 0.05) 0px, transparent 50%),
            radial-gradient(at 90% 10%, rgba(0, 184, 148, 0.05) 0px, transparent 50%),
            radial-gradient(at 50% 90%, rgba(9, 132, 227, 0.05) 0px, transparent 50%);
        background-attachment: fixed;
    }

    /* SIDEBAR GLASS */
    section[data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.75) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255,255,255,0.6);
    }
    
    /* FIX FOR SELECTBOX & DROPDOWN VISIBILITY */
    /* The main box */
    div[data-baseweb="select"] > div {
        background-color: white !important;
        border-color: #dfe6e9 !important;
        color: #2d3436 !important;
    }
    /* The dropdown text color */
    div[data-baseweb="select"] span {
        color: #2d3436 !important;
    }
    /* The dropdown menu options */
    ul[data-baseweb="menu"] {
        background-color: white !important;
        color: #2d3436 !important;
    }
    li[data-baseweb="option"] {
        color: #2d3436 !important;
    }
    
    /* GLASS CARDS */
    .glass-card {
        background: rgba(255, 255, 255, 0.7);
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.8);
        padding: 20px;
        margin-bottom: 15px;
        transition: transform 0.2s;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        background: rgba(255, 255, 255, 0.9);
    }
    
    /* TYPOGRAPHY & BUTTONS */
    h1, h2, h3 { color: #2d3436; font-family: 'Helvetica Neue', sans-serif; }
    .stButton > button {
        background: #2d3436;
        color: white;
        border: none;
        width: 100%;
        padding: 0.5rem;
        border-radius: 8px;
        font-weight: 600;
    }
    .stButton > button:hover {
        background: #00b894;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# HELPER: SHAP
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# --- 5. LOAD DATA & MODEL ---
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
    df = pd.read_csv(url)
    df['bmi_cat'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
    return df

df = load_data()

@st.cache_resource
def train_model(df):
    cat = ['sex', 'smoker', 'region', 'bmi_cat']
    num = ['age', 'bmi', 'children']
    prep = ColumnTransformer([('num', StandardScaler(), num), ('cat', OneHotEncoder(handle_unknown='ignore'), cat)])
    ens = VotingRegressor([('rf', RandomForestRegressor(n_estimators=100, random_state=42)), ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)), ('lr', LinearRegression())])
    pipe = Pipeline([('prep', prep), ('model', ens)])
    X = df.drop(['charges'], axis=1); y = df['charges']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(X_train, y_train)
    return pipe, r2_score(y_test, pipe.predict(X_test)), mean_absolute_error(y_test, pipe.predict(X_test))

model, r2, mae = train_model(df)

# --- 6. SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("### 💎 **InsurRisk** `PRO`")
    st.markdown("---")
    
    with st.container():
        st.caption("APPLICANT PROFILE")
        # NOTE: We bind these widgets to st.session_state using the 'key' parameter
        age = st.slider("Age", 18, 100, key='age')
        sex = st.selectbox("Gender", ["male", "female"], key='sex')
        children = st.slider("Dependents", 0, 5, key='children')
        
    st.markdown("---")
    
    with st.container():
        st.caption("RISK FACTORS")
        bmi = st.slider("BMI Index", 15.0, 55.0, key='bmi')
        smoker = st.radio("Smoker Status", ["yes", "no"], horizontal=True, key='smoker')
        region = st.selectbox("Region", df['region'].unique(), key='region')

    st.markdown("<br>", unsafe_allow_html=True)
    
    # FIXED RESET BUTTON: Uses callback to reset session state variables directly
    st.button("↻ Reset Parameters", on_click=reset_callback)

# --- 7. MAIN DASHBOARD ---
# Logic
if bmi < 18.5: bmi_cat = 'Underweight'
elif bmi < 25: bmi_cat = 'Normal'
elif bmi < 30: bmi_cat = 'Overweight'
else: bmi_cat = 'Obese'

input_data = pd.DataFrame([{
    'age': age, 'sex': sex, 'bmi': bmi, 'children': children, 
    'smoker': smoker, 'region': region, 'bmi_cat': bmi_cat
}])
pred = model.predict(input_data)[0]

# Risk Calculation
q33 = df['charges'].quantile(0.33)
q66 = df['charges'].quantile(0.66)
if pred < q33: tier, color = "Low Exposure", "#00b894"
elif pred < q66: tier, color = "Medium Exposure", "#fdcb6e"
else: tier, color = "High Exposure", "#ff7675"

st.markdown("## **Executive Forecast**")
st.markdown(f"Status: <span style='color:{color}; font-weight:bold; font-size:18px'>{tier.upper()}</span>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# KPI CARDS
c1, c2, c3, c4 = st.columns(4)
c1.markdown(f"""<div class="glass-card"><div style="font-size:12px;color:#b2bec3;font-weight:700">PREDICTED PAYOUT</div><div style="font-size:28px;font-weight:800;color:#2d3436">${pred:,.0f}</div><div style="font-size:12px;color:#00b894">±${mae:,.0f} (MAE)</div></div>""", unsafe_allow_html=True)
c2.markdown(f"""<div class="glass-card"><div style="font-size:12px;color:#b2bec3;font-weight:700">RISK TIER</div><div style="font-size:28px;font-weight:800;color:{color}">{tier.split()[0]}</div><div style="font-size:12px;color:#636e72">Quantile Logic</div></div>""", unsafe_allow_html=True)
c3.markdown(f"""<div class="glass-card"><div style="font-size:12px;color:#b2bec3;font-weight:700">CONFIDENCE</div><div style="font-size:28px;font-weight:800;color:#2d3436">{r2*100:.1f}%</div><div style="font-size:12px;color:#636e72">Ensemble R²</div></div>""", unsafe_allow_html=True)
action = "Auto-Approve" if pred < q66 else "Manual Review"
c4.markdown(f"""<div class="glass-card"><div style="font-size:12px;color:#b2bec3;font-weight:700">DECISION</div><div style="font-size:24px;font-weight:800;color:#2d3436">{action}</div><div style="font-size:12px;color:#0984e3">AI Recommendation</div></div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# VISUALIZATIONS
tab1, tab2 = st.tabs(["📊 Market Benchmarking", "🧠 Feature Impact"])

with tab1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        fig = px.scatter(df, x='bmi', y='charges', color='smoker', color_discrete_map={'yes':'#ff7675', 'no':'#74b9ff'}, opacity=0.5, title="Cost vs BMI Distribution")
        fig.add_trace(go.Scatter(x=[bmi], y=[pred], mode='markers', marker=dict(size=20, color='#2d3436', symbol='x'), name='Applicant'))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color':'#2d3436'}, margin=dict(t=30,l=0,r=0,b=0), height=350)
        st.plotly_chart(fig, width="stretch")
    with col2:
        fig_g = go.Figure(go.Indicator(mode="gauge+number", value=pred, title={'text':"Risk Meter"}, gauge={'axis':{'range':[0,65000]}, 'bar':{'color':'#2d3436'}, 'steps':[{'range':[0,15000],'color':'rgba(0,184,148,0.4)'},{'range':[15000,35000],'color':'rgba(253,203,110,0.4)'},{'range':[35000,65000],'color':'rgba(255,118,117,0.4)'}]}))
        fig_g.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color':'#2d3436'}, margin=dict(t=30,l=20,r=20,b=20), height=350)
        st.plotly_chart(fig_g, width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.caption("How much did each factor add/subtract from the base price?")
    gb = model.named_steps['model'].estimators_[1]
    prep = model.named_steps['prep']
    inp_trans = prep.transform(input_data)
    explainer = shap.TreeExplainer(gb)
    shap_val = explainer.shap_values(inp_trans)
    st_shap(shap.force_plot(explainer.expected_value, shap_val, feature_names=prep.get_feature_names_out()), height=160)
    st.markdown('</div>', unsafe_allow_html=True)