import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Career Predictor",
    page_icon="🚀",
    layout="wide"
)

# -----------------------------
# CYBERPUNK CSS
# -----------------------------
st.markdown("""
<style>

body {
    background-color: #0f0f0f;
}

.main {
    background: linear-gradient(135deg, #0f0f0f, #1a0033);
    color: white;
}

h1 {
    color: #00ffe7;
    text-align: center;
    font-size: 50px;
    text-shadow: 0px 0px 20px #00ffe7;
}

h2, h3 {
    color: #ff00ff;
}

.stButton>button {
    background: linear-gradient(90deg, #ff00ff, #00ffe7);
    color: black;
    border-radius: 10px;
    border: none;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
    box-shadow: 0px 0px 20px #00ffe7;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #00ffe7, #ff00ff);
    color: white;
}

.stTextInput>div>div>input {
    background-color: #111111;
    color: #00ffe7;
    border: 1px solid #ff00ff;
}

.stNumberInput input {
    background-color: #111111;
    color: #00ffe7;
}

.stSelectbox div[data-baseweb="select"] {
    background-color: #111111;
    color: white;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    background: rgba(0,0,0,0.7);
    border: 2px solid #00ffe7;
    box-shadow: 0px 0px 25px #00ffe7;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL FILES
# -----------------------------
pipeline = joblib.load("career_prediction.pkl")

model = pipeline["model"]
cv = pipeline["vectorizer"]

career_encoder = pipeline["career_encoder"]
interest_encoder = pipeline["interest_encoder"]
personality_encoder = pipeline["personality_encoder"]
education_encoder = pipeline["education_encoder"]

# -----------------------------
# TITLE
# -----------------------------
st.title("🚀 AI Career Prediction System")
st.write("### Predict the Best Career Using AI & Machine Learning")

# -----------------------------
# USER INPUTS
# -----------------------------
skills = st.text_input(
    "Enter Skills (Separated by |)",
    "Python|SQL|Machine Learning"
)

# -----------------------------
# INTERESTS
# -----------------------------
interest_options = list(interest_encoder.classes_)

selected_interest = st.selectbox(
    "Select Your Interest",
    interest_options
)

interests = interest_encoder.transform(
    [selected_interest]
)[0]

# -----------------------------
# PERSONALITY
# -----------------------------
personality_options = list(personality_encoder.classes_)

selected_personality = st.selectbox(
    "Select Personality Type",
    personality_options
)

personality = personality_encoder.transform(
    [selected_personality]
)[0]

# -----------------------------
# EDUCATION STREAM
# -----------------------------
education_options = list(education_encoder.classes_)

selected_education = st.selectbox(
    "Select Education Stream",
    education_options
)

education_stream = education_encoder.transform(
    [selected_education]
)[0]

# -----------------------------
# OTHER INPUTS
# -----------------------------
cgpa = st.slider("CGPA", 0.0, 10.0, 7.5)

projects_done = st.slider("Projects Done", 0, 20, 5)

internships = st.slider("Internships", 0, 10, 2)

certifications = st.slider("Certifications", 0, 15, 3)

coding_hours_per_week = st.slider(
    "Coding Hours Per Week",
    0,
    60,
    20
)

communication_skill = st.slider(
    "Communication Skill",
    1,
    10,
    7
)

leadership_score = st.slider(
    "Leadership Score",
    1,
    10,
    6
)

problem_solving_score = st.slider(
    "Problem Solving Score",
    1,
    10,
    8
)

creative_score = st.slider(
    "Creative Score",
    1,
    10,
    6
)

# -----------------------------
# PREDICTION BUTTON
# -----------------------------
if st.button("⚡ Predict Career"):

    # VECTORIZE SKILLS
    new_skills = cv.transform([skills])

    # NUMERIC FEATURES
    new_numeric = pd.DataFrame([[
        interests,
        personality,
        education_stream,
        cgpa,
        projects_done,
        internships,
        certifications,
        coding_hours_per_week,
        communication_skill,
        leadership_score,
        problem_solving_score,
        creative_score
    ]])

    # COMBINE FEATURES
    new_final = hstack([new_skills, new_numeric])

    # PREDICTION
    prediction = model.predict(new_final)

    predicted_career = career_encoder.inverse_transform(prediction)

    # PROBABILITIES
    probabilities = model.predict_proba(new_final)

    top_3 = probabilities[0].argsort()[-3:][::-1]

    # -----------------------------
    # DISPLAY RESULT
    # -----------------------------
    st.markdown(f"""
    <div class="result-box">
        <h2>🎯 Predicted Career</h2>
        <h1>{predicted_career[0]}</h1>
    </div>
    """, unsafe_allow_html=True)

    st.write("## 🔥 Top 3 Career Recommendations")

    for idx in top_3:

        career_name = career_encoder.inverse_transform([idx])[0]

        confidence = round(
            probabilities[0][idx] * 100,
            2
        )

        st.write(f"### {career_name} → {confidence}%")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("""
---
<center>
<h4 style='color:#00ffe7;'>
Made with Streamlit + Machine Learning 🚀
</h4>
</center>
""", unsafe_allow_html=True)