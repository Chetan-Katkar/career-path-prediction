import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

st.set_page_config(
    page_title="Career Path Predictor",
    page_icon="C",
    layout="centered"
)

st.markdown("""
<style>
    .stApp {
        background: #000000;
    }
    [data-testid="stSidebar"] {
        background: #000000;
    }
    .main-title {
        text-align: center;
        padding: 1.5rem 0;
    }
    .main-title h1 {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -1px;
    }
    .main-title p {
        color: #666666;
        font-size: 0.95rem;
    }
    .step-indicator {
        display: flex;
        justify-content: center;
        gap: 0.5rem;
        margin: 1.5rem 0;
    }
    .step-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: #333333;
        display: inline-block;
    }
    .step-dot.active {
        background: #ffffff;
        box-shadow: 0 0 10px rgba(255,255,255,0.5);
    }
    .step-dot.done {
        background: #22c55e;
    }
    .step-label {
        color: #ffffff;
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #222222;
    }
    .prediction-box {
        background: #111111;
        border: 1px solid #333333;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem 0;
    }
    .prediction-box .role-name {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    .prediction-box .role-desc {
        color: #888888;
        font-size: 0.95rem;
    }
    .prob-row {
        display: flex;
        align-items: center;
        margin: 0.4rem 0;
        gap: 0.5rem;
    }
    .prob-label {
        color: #aaaaaa;
        font-size: 0.8rem;
        width: 220px;
        text-align: right;
        flex-shrink: 0;
    }
    .prob-track {
        flex-grow: 1;
        background: #1a1a1a;
        border-radius: 4px;
        height: 20px;
        overflow: hidden;
    }
    .prob-fill {
        height: 100%;
        border-radius: 4px;
        display: flex;
        align-items: center;
        padding-left: 6px;
        font-size: 0.7rem;
        color: white;
        font-weight: 600;
    }
    .prob-fill.top {
        background: #ffffff;
        color: #000000;
    }
    .prob-fill.other {
        background: #333333;
    }
    .nav-buttons {
        display: flex;
        gap: 1rem;
        margin-top: 1.5rem;
    }
    .stButton > button {
        background: #ffffff;
        color: #000000;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 700;
        font-size: 1rem;
        width: 100%;
    }
    .stButton > button:hover {
        background: #e0e0e0;
    }
    .disclaimer-text {
        color: #444444;
        font-size: 0.75rem;
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #1a1a1a;
    }
    div[data-baseweb="select"] > div {
        background: #111111;
        border-color: #333333;
    }
    .stSlider label, .stSelectbox label {
        color: #cccccc !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_path = os.path.join(base_path, 'models')
    model = joblib.load(os.path.join(models_path, 'best_model.pkl'))
    target_encoder = joblib.load(os.path.join(models_path, 'target_encoder.pkl'))
    label_encoders = joblib.load(os.path.join(models_path, 'label_encoders.pkl'))
    feature_names = joblib.load(os.path.join(models_path, 'feature_names.pkl'))
    return model, target_encoder, label_encoders, feature_names

model, target_encoder, label_encoders, feature_names = load_assets()

role_info = {
    'Applications Developer': 'Designs and builds software applications for desktop and mobile platforms.',
    'CRM Technical Developer': 'Develops and customizes Customer Relationship Management systems.',
    'Database Developer': 'Designs, implements, and optimizes database systems and queries.',
    'Mobile Applications Developer': 'Creates applications specifically for iOS and Android devices.',
    'Network Security Engineer': 'Protects computer networks from cyber threats and vulnerabilities.',
    'Software Developer': 'Writes, tests, and maintains software across various platforms.',
    'Software Engineer': 'Applies engineering principles to design large-scale software systems.',
    'Software Quality Assurance (QA) / Testing': 'Ensures software quality through systematic testing and bug tracking.',
    'Systems Security Administrator': 'Manages and secures IT infrastructure and server systems.',
    'Technical Support': 'Provides technical assistance and troubleshooting for end users.',
    'UX Designer': 'Designs intuitive and visually appealing user experiences.',
    'Web Developer': 'Builds and maintains websites and web applications.',
}

TOTAL_STEPS = 5

if 'step' not in st.session_state:
    st.session_state.step = 1
if 'inputs' not in st.session_state:
    st.session_state.inputs = {}

st.markdown("""
<div class="main-title">
    <h1>Career Path Predictor</h1>
    <p>Answer a few questions and discover your ideal tech career</p>
</div>
""", unsafe_allow_html=True)

step = st.session_state.step

dots_html = ""
for i in range(1, TOTAL_STEPS + 1):
    if i < step:
        dots_html += '<span class="step-dot done"></span>'
    elif i == step:
        dots_html += '<span class="step-dot active"></span>'
    else:
        dots_html += '<span class="step-dot"></span>'
st.markdown(f'<div class="step-indicator">{dots_html}</div>', unsafe_allow_html=True)

if step == 1:
    st.markdown('<div class="step-label">Step 1 of 5 — Ratings & Scores</div>', unsafe_allow_html=True)
    
    logical_rating = st.slider("Logical Quotient Rating", 1, 9, 
                               st.session_state.inputs.get('logical_rating', 5))
    hackathons = st.slider("Hackathons Attended", 0, 6,
                           st.session_state.inputs.get('hackathons', 3))
    coding_rating = st.slider("Coding Skills Rating", 1, 9,
                              st.session_state.inputs.get('coding_rating', 5))
    speaking_points = st.slider("Public Speaking Points", 1, 9,
                                st.session_state.inputs.get('speaking_points', 5))
    
    if st.button("Next →"):
        st.session_state.inputs['logical_rating'] = logical_rating
        st.session_state.inputs['hackathons'] = hackathons
        st.session_state.inputs['coding_rating'] = coding_rating
        st.session_state.inputs['speaking_points'] = speaking_points
        st.session_state.step = 2
        st.rerun()

elif step == 2:
    st.markdown('<div class="step-label">Step 2 of 5 — Yes / No Questions</div>', unsafe_allow_html=True)
    
    opts = ["yes", "no"]
    self_learning = st.selectbox("Self-learning capability?", opts,
                                 index=opts.index(st.session_state.inputs.get('self_learning', 'yes')))
    extra_courses = st.selectbox("Have you done extra courses?", opts,
                                 index=opts.index(st.session_state.inputs.get('extra_courses', 'yes')))
    senior_input = st.selectbox("Taken inputs from seniors or elders?", opts,
                                index=opts.index(st.session_state.inputs.get('senior_input', 'yes')))
    team_work = st.selectbox("Have you worked in teams?", opts,
                             index=opts.index(st.session_state.inputs.get('team_work', 'yes')))
    introvert = st.selectbox("Are you an introvert?", opts,
                             index=opts.index(st.session_state.inputs.get('introvert', 'no')))
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back"):
            st.session_state.step = 1
            st.rerun()
    with col2:
        if st.button("Next →"):
            st.session_state.inputs['self_learning'] = self_learning
            st.session_state.inputs['extra_courses'] = extra_courses
            st.session_state.inputs['senior_input'] = senior_input
            st.session_state.inputs['team_work'] = team_work
            st.session_state.inputs['introvert'] = introvert
            st.session_state.step = 3
            st.rerun()

elif step == 3:
    st.markdown('<div class="step-label">Step 3 of 5 — Skill Levels</div>', unsafe_allow_html=True)
    
    skill_opts = ["poor", "medium", "excellent"]
    rw_skills = st.selectbox("Reading and Writing Skills", skill_opts,
                             index=skill_opts.index(st.session_state.inputs.get('rw_skills', 'medium')))
    memory_score = st.selectbox("Memory Capability", skill_opts,
                                index=skill_opts.index(st.session_state.inputs.get('memory_score', 'medium')))
    
    mgmt_opts = ["Management", "Technical"]
    mgmt_tech = st.selectbox("Management or Technical?", mgmt_opts,
                             index=mgmt_opts.index(st.session_state.inputs.get('mgmt_tech', 'Technical')))
    
    worker_opts = ["hard worker", "smart worker"]
    worker_type = st.selectbox("Work Style", worker_opts,
                               index=worker_opts.index(st.session_state.inputs.get('worker_type', 'smart worker')))
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back"):
            st.session_state.step = 2
            st.rerun()
    with col2:
        if st.button("Next →"):
            st.session_state.inputs['rw_skills'] = rw_skills
            st.session_state.inputs['memory_score'] = memory_score
            st.session_state.inputs['mgmt_tech'] = mgmt_tech
            st.session_state.inputs['worker_type'] = worker_type
            st.session_state.step = 4
            st.rerun()

elif step == 4:
    st.markdown('<div class="step-label">Step 4 of 5 — Interests & Preferences</div>', unsafe_allow_html=True)
    
    cert_list = sorted(list(label_encoders['certifications'].classes_))
    certification = st.selectbox("Certification Area", cert_list,
                                 index=cert_list.index(st.session_state.inputs.get('certification', cert_list[0])))
    
    ws_list = sorted(list(label_encoders['workshops'].classes_))
    workshop = st.selectbox("Workshop Attended", ws_list,
                            index=ws_list.index(st.session_state.inputs.get('workshop', ws_list[0])))
    
    subj_list = sorted(list(label_encoders['Interested subjects'].classes_))
    interested_subject = st.selectbox("Interested Subject", subj_list,
                                      index=subj_list.index(st.session_state.inputs.get('interested_subject', subj_list[0])))
    
    career_list = sorted(list(label_encoders['interested career area '].classes_))
    career_area = st.selectbox("Interested Career Area", career_list,
                               index=career_list.index(st.session_state.inputs.get('career_area', career_list[0])))
    
    company_list = sorted(list(label_encoders['Type of company want to settle in?'].classes_))
    company_type = st.selectbox("Preferred Company Type", company_list,
                                index=company_list.index(st.session_state.inputs.get('company_type', company_list[0])))
    
    book_list = sorted(list(label_encoders['Interested Type of Books'].classes_))
    book_type = st.selectbox("Interested Type of Books", book_list,
                             index=book_list.index(st.session_state.inputs.get('book_type', book_list[0])))
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back"):
            st.session_state.step = 3
            st.rerun()
    with col2:
        if st.button("Predict My Career →"):
            st.session_state.inputs['certification'] = certification
            st.session_state.inputs['workshop'] = workshop
            st.session_state.inputs['interested_subject'] = interested_subject
            st.session_state.inputs['career_area'] = career_area
            st.session_state.inputs['company_type'] = company_type
            st.session_state.inputs['book_type'] = book_type
            st.session_state.step = 5
            st.rerun()

elif step == 5:
    binary_map = {'yes': 1, 'no': 0}
    ordinal_map = {'poor': 0, 'medium': 1, 'excellent': 2}
    mgmt_map = {'Management': 0, 'Technical': 1}
    worker_map = {'hard worker': 0, 'smart worker': 1}
    inp = st.session_state.inputs

    input_data = {}
    for feat in feature_names:
        if feat == 'Logical quotient rating':
            input_data[feat] = inp['logical_rating']
        elif feat == 'hackathons':
            input_data[feat] = inp['hackathons']
        elif feat == 'coding skills rating':
            input_data[feat] = inp['coding_rating']
        elif feat == 'public speaking points':
            input_data[feat] = inp['speaking_points']
        elif feat == 'self-learning capability?':
            input_data[feat] = binary_map[inp['self_learning']]
        elif feat == 'Extra-courses did':
            input_data[feat] = binary_map[inp['extra_courses']]
        elif feat == 'Taken inputs from seniors or elders':
            input_data[feat] = binary_map[inp['senior_input']]
        elif feat == 'worked in teams ever?':
            input_data[feat] = binary_map[inp['team_work']]
        elif feat == 'Introvert':
            input_data[feat] = binary_map[inp['introvert']]
        elif feat == 'reading and writing skills':
            input_data[feat] = ordinal_map[inp['rw_skills']]
        elif feat == 'memory capability score':
            input_data[feat] = ordinal_map[inp['memory_score']]
        elif feat == 'Management or Technical':
            input_data[feat] = mgmt_map[inp['mgmt_tech']]
        elif feat == 'hard/smart worker':
            input_data[feat] = worker_map[inp['worker_type']]
        elif feat == 'certifications':
            input_data[feat] = label_encoders['certifications'].transform([inp['certification']])[0]
        elif feat == 'workshops':
            input_data[feat] = label_encoders['workshops'].transform([inp['workshop']])[0]
        elif feat == 'Interested subjects':
            input_data[feat] = label_encoders['Interested subjects'].transform([inp['interested_subject']])[0]
        elif feat.strip() == 'interested career area':
            input_data[feat] = label_encoders['interested career area '].transform([inp['career_area']])[0]
        elif feat == 'Type of company want to settle in?':
            input_data[feat] = label_encoders['Type of company want to settle in?'].transform([inp['company_type']])[0]
        elif feat == 'Interested Type of Books':
            input_data[feat] = label_encoders['Interested Type of Books'].transform([inp['book_type']])[0]

    input_array = np.array([list(input_data.values())])
    prediction = model.predict(input_array)[0]
    probabilities = model.predict_proba(input_array)[0]
    predicted_role = target_encoder.inverse_transform([prediction])[0]
    desc = role_info.get(predicted_role, 'A great tech career!')

    st.markdown(f"""
    <div class="prediction-box">
        <div class="role-name">{predicted_role}</div>
        <div class="role-desc">{desc}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="step-label">Confidence Breakdown</div>', unsafe_allow_html=True)

    prob_df = pd.DataFrame({
        'Role': target_encoder.classes_,
        'Probability': probabilities
    }).sort_values('Probability', ascending=False)

    for _, row in prob_df.iterrows():
        pct = row['Probability'] * 100
        bar_w = max(pct, 2)
        fill_class = "top" if row['Role'] == predicted_role else "other"
        st.markdown(f"""
        <div class="prob-row">
            <div class="prob-label">{row['Role'].split('(')[0].strip()}</div>
            <div class="prob-track">
                <div class="prob-fill {fill_class}" style="width:{bar_w}%">{pct:.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    if st.button("← Start Over"):
        st.session_state.step = 1
        st.session_state.inputs = {}
        st.rerun()

    st.markdown("""
    <div class="disclaimer-text">
        This prediction is generated by a Machine Learning model trained on a practice dataset.
        Results should be used as a general reference only.
    </div>
    """, unsafe_allow_html=True)
