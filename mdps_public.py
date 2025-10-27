# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 17:00:19 2025

@author: sksou
"""

import streamlit as st
import pickle
import sqlite3
import hashlib
import json
from datetime import datetime
from streamlit_option_menu import option_menu
import google.generativeai as genai
from PIL import Image
import pytesseract

DB_PATH = "healthapp.db"

def db_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    con = db_conn(); cur = con.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      username TEXT UNIQUE NOT NULL,
      password_hash TEXT NOT NULL
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS predictions(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id INTEGER NOT NULL,
      disease TEXT NOT NULL,
      input_values TEXT NOT NULL,
      result TEXT NOT NULL,
      timestamp TEXT NOT NULL,
      FOREIGN KEY(user_id) REFERENCES users(id)
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS chats(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id INTEGER NOT NULL,
      messages TEXT NOT NULL,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      FOREIGN KEY(user_id) REFERENCES users(id)
    )""")

    try:
        cur.execute("ALTER TABLE chats ADD COLUMN title TEXT DEFAULT ''")
    except sqlite3.OperationalError:
        pass
    try:
        cur.execute("ALTER TABLE chats ADD COLUMN type TEXT DEFAULT 'normal'")
    except sqlite3.OperationalError:
        pass

    cur.execute("UPDATE chats SET title = 'Old Session #' || id WHERE title IS NULL OR title = ''")

    con.commit(); con.close()

def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def create_user(username: str, password: str) -> bool:
    try:
        con = db_conn(); cur = con.cursor()
        cur.execute("INSERT INTO users(username,password_hash) VALUES(?,?)",
                    (username, hash_password(password)))
        con.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        con.close()

def login_user(username: str, password: str):
    con = db_conn(); cur = con.cursor()
    cur.execute("SELECT id, password_hash FROM users WHERE username=?", (username,))
    row = cur.fetchone()
    con.close()
    if row and row[1] == hash_password(password):
        return row[0]
    return None

def save_prediction(user_id: int, disease: str, inputs: list, result: str):
    con = db_conn(); cur = con.cursor()
    cur.execute(
        "INSERT INTO predictions(user_id,disease,input_values,result,timestamp) VALUES(?,?,?,?,?)",
        (user_id, disease, json.dumps(inputs), result, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    con.commit(); con.close()

def load_predictions(user_id: int):
    con = db_conn(); cur = con.cursor()
    cur.execute("""
        SELECT disease, input_values, result, timestamp
        FROM predictions
        WHERE user_id=?
        ORDER BY timestamp DESC
    """, (user_id,))
    rows = cur.fetchall()
    con.close()
    return [(d, json.loads(inp), r, ts) for (d, inp, r, ts) in rows]

def create_chat_session(user_id: int, title="New Chat", chat_type="normal") -> int:
    con = db_conn(); cur = con.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur.execute(
        "INSERT INTO chats(user_id,title,type,messages,created_at,updated_at) VALUES(?,?,?,?,?,?)",
        (user_id, title, chat_type, json.dumps([]), now, now)
    )
    chat_id = cur.lastrowid
    con.commit(); con.close()
    return chat_id

def load_chat_sessions(user_id: int):
    con = db_conn(); cur = con.cursor()
    cur.execute("""
        SELECT id, title, type, messages, created_at, updated_at
        FROM chats
        WHERE user_id=?
        ORDER BY updated_at DESC
    """, (user_id,))
    rows = cur.fetchall()
    con.close()
    return [(cid, title, ctype, json.loads(msgs), c_at, u_at) for (cid, title, ctype, msgs, c_at, u_at) in rows]

def save_chat_messages(chat_id: int, messages: list, title=None):
    con = db_conn(); cur = con.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if title is not None:
        cur.execute("UPDATE chats SET messages=?, updated_at=?, title=? WHERE id=?",
                    (json.dumps(messages), now, title, chat_id))
    else:
        cur.execute("UPDATE chats SET messages=?, updated_at=? WHERE id=?",
                    (json.dumps(messages), now, chat_id))
    con.commit(); con.close()

def delete_chat(chat_id: int):
    con = db_conn(); cur = con.cursor()
    cur.execute("DELETE FROM chats WHERE id=?", (chat_id,))
    con.commit(); con.close()

def load_chat_by_id(chat_id: int):
    con = db_conn(); cur = con.cursor()
    cur.execute("SELECT messages FROM chats WHERE id=?", (chat_id,))
    row = cur.fetchone()
    con.close()
    return json.loads(row[0]) if row else []

init_db()
diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
heart_model = pickle.load(open('heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))
st.set_page_config(page_title="Multi-Disease Prediction System", layout="wide")

if "user_id" not in st.session_state:
    st.title("Login / Register")
    tab1, tab2 = st.tabs(["Login", "Register"])
    with tab1:
        login_username = st.text_input("Username (Login)", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", key="login_btn"):
            uid = login_user(login_username, login_password)
            if uid:
                st.session_state.user_id = uid
                st.session_state.username = login_username
                sessions = load_chat_sessions(uid)
                if sessions:
                    st.session_state.chat_session_id = sessions[0][0]
                    st.session_state.chat_history = sessions[0][3]
                else:
                    st.session_state.chat_session_id = create_chat_session(uid)
                    st.session_state.chat_history = []
                st.rerun()
            else:
                st.error("Invalid username or password")
    with tab2:
        reg_username = st.text_input("Username (Register)", key="reg_username")
        reg_password = st.text_input("Password", type="password", key="reg_password")
        if st.button("Register", key="reg_btn"):
            if create_user(reg_username, reg_password):
                st.success("Account created! Please login now.")
            else:
                st.error("Username already exists. Try another.")
    st.stop()

with st.sidebar:
    st.success(f"Logged in as {st.session_state['username']}")
    if st.button("Logout", key="logout_btn"):
        st.session_state.clear(); st.rerun()
    selected = option_menu(
        'Disease Prediction System',
        ['Diabetes Prediction','Heart Disease Prediction','Parkinson’s Prediction',
         'HealthBot Assistant','Upload Health Report','Past Predictions'],
        icons=['activity','heart','person','robot','file-earmark-arrow-up','clock-history'],
        default_index=0
    )

def extract_text_from_image(uploaded_file):
    image = Image.open(uploaded_file)
    text = pytesseract.image_to_string(image)
    return text

if "redirect_to" in st.session_state and st.session_state["redirect_to"]:
    selected = st.session_state["redirect_to"]
    st.session_state["redirect_to"] = None
    
if selected == 'Diabetes Prediction':
    st.title("Diabetes Prediction using ML")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input("Pregnancies")
    with col2:
        Glucose = st.text_input("Glucose Level")
    with col3:
        BloodPressure = st.text_input("Blood Pressure value")
    with col1:
        SkinThickness = st.text_input("Skin Thickness value")
    with col2:
        Insulin = st.text_input("Insulin Level")
    with col3:
        BMI = st.text_input("BMI value")
    with col1:
        DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function value")
    with col2:
        Age = st.text_input("Age")

    if st.button('Diabetes Test Result'):
        user_input_d = [int(Pregnancies), int(Glucose), int(BloodPressure),
                        int(SkinThickness), int(Insulin), float(BMI),
                        float(DiabetesPedigreeFunction), int(Age)]
        diab_prediction = diabetes_model.predict([user_input_d])
        if diab_prediction[0] == 1:
            st.error('The person is likely to have diabetes.')
            diab_status = 'likely to have diabetes'
        else:
            st.success('The person is not diabetic.')
            diab_status = 'not diabetic'
        st.session_state['last_prediction'] = {
            'disease': 'Diabetes',
            'input': user_input_d,
            'result': diab_status
        }
        
if selected == 'Heart Disease Prediction':
    st.title("Heart Disease Prediction using ML")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input('Age')
        sex = st.text_input('Sex (1 = Male, 0 = Female)')
        cp = st.text_input('Chest Pain Type (0–3)')
        trestbps = st.text_input('Resting Blood Pressure')
        chol = st.text_input('Serum Cholesterol (mg/dl)')
    with col2:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)')
        restecg = st.text_input('Resting ECG Results (0–2)')
        thalach = st.text_input('Maximum Heart Rate Achieved')
        exang = st.text_input('Exercise Induced Angina (1 = Yes, 0 = No)')
    with col3:
        oldpeak = st.text_input('Oldpeak (ST Depression by Exercise)')
        slope = st.text_input('Slope of Peak Exercise ST Segment (0–2)')
        ca = st.text_input('Number of Major Vessels (0–3)')
        thal = st.text_input('Thalassemia (0 = Normal, 1 = Fixed, 2 = Reversible)')

    if st.button('Heart Disease Test Result'):
        user_input_h = [
            int(age), int(sex), int(cp), int(trestbps), int(chol),
            int(fbs), int(restecg), int(thalach), int(exang),
            float(oldpeak), int(slope), int(ca), int(thal)
        ]
        heart_prediction = heart_model.predict([user_input_h])
        if heart_prediction[0] == 1:
            st.error('The person is likely to have heart disease.')
            heart_status = 'likely to have heart disease'
        else:
            st.success('The person does not have any heart disease.')
            heart_status = 'does not have any heart disease'
        st.session_state['last_prediction'] = {
            'disease': 'Heart Disease',
            'input': user_input_h,
            'result': heart_status
        }
        
if selected == "Parkinson’s Prediction":
    st.title("Parkinson’s Disease Prediction using ML")

    col1, col2, col3, col4 = st.columns(4)  
    
    with col1:
        fo = st.text_input('Average Vocal Fundamental Frequency (Hz)')
        Jitter_Abs = st.text_input('Jitter (Abs) - Small Frequency Variations')
        Shimmer = st.text_input('Shimmer - Variation in Amplitude')
        APQ = st.text_input('APQ - Average Amplitude Perturbation')
        RPDE = st.text_input('RPDE - Nonlinear Dynamical Complexity')
        D2 = st.text_input('D2 - Dynamical Complexity Measure')
    
    with col2:
        fhi = st.text_input('Maximum Vocal Fundamental Frequency (Hz)')
        RAP = st.text_input('RAP - Relative Average Perturbation')
        Shimmer_dB = st.text_input('Shimmer (dB)')
        DDA = st.text_input('DDA - Average Absolute Difference of Periods')
        DFA = st.text_input('DFA - Signal Fractal Scaling')
        PPE = st.text_input('PPE - Pitch Period Entropy')
    
    with col3:
        flo = st.text_input('Minimum Vocal Fundamental Frequency (Hz)')
        PPQ = st.text_input('PPQ - Pitch Period Perturbation Quotient')
        APQ3 = st.text_input('APQ3 - Amplitude Perturbation (3 cycles)')
        NHR = st.text_input('NHR - Noise to Harmonic Ratio')
        spread1 = st.text_input('Spread1 - Nonlinear Frequency Variation Measure 1')

    with col4:
        Jitter_percent = st.text_input('Jitter (%) - Variation in Frequency')
        DDP = st.text_input('DDP - Difference of Differences of Periods')
        APQ5 = st.text_input('APQ5 - Amplitude Perturbation (5 cycles)')
        HNR = st.text_input('HNR - Harmonic to Noise Ratio')
        spread2 = st.text_input('Spread2 - Nonlinear Frequency Variation Measure 2')
        
    parkinsons_diagnosis = ''

    if st.button("Parkinson's Test Result"):
        try:
            user_input = [
                float(fo), float(fhi), float(flo), float(Jitter_percent),
                float(Jitter_Abs), float(RAP), float(PPQ), float(DDP),
                float(Shimmer), float(Shimmer_dB), float(APQ3), float(APQ5),
                float(APQ), float(DDA), float(NHR), float(HNR), float(RPDE),
                float(DFA), float(spread1), float(spread2), float(D2), float(PPE)
            ]
            parkinsons_prediction = parkinsons_model.predict([user_input])

            if parkinsons_prediction[0] == 1:
                st.error("The person likely has Parkinson’s Disease.")
                park_status = "likely to have Parkinson’s Disease"
            else:
                st.success("The person is healthy.")
                park_status = "does not have Parkinson’s Disease"

            st.session_state['last_prediction'] = {
                'disease': "Parkinson’s Disease",
                'input': user_input,
                'result': park_status
            }

        except ValueError:
            st.error("Please fill all fields with valid numeric values.")

if selected == 'HealthBot Assistant':
    st.title("AI HealthBot Assistant")

    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    except Exception:
        st.error("Gemini API key missing or invalid.")
        st.stop()
        
    def generate_session_title(message: str, chat_type="normal") -> str:
        if chat_type == "report":
            return "Report: " + (message[:25] + "..." if len(message) > 25 else message)
        else:
            return message[:25] + ("..." if len(message) > 25 else "")

    if "chat_session_id" not in st.session_state:
        st.session_state.chat_session_id = create_chat_session(st.session_state.user_id, title="New Chat")
        st.session_state.chat_history = []

    with st.sidebar:
        st.markdown("###Chats")
        sessions = load_chat_sessions(st.session_state.user_id)

        if sessions:
            for cid, title, ctype, msgs, created, updated in sessions:
                tag = "📝" if ctype == "report" else "💬"
                display_title = title if title else "Untitled"
                short_title = display_title[:25] + ("..." if len(display_title) > 25 else "")
                if st.button(f"{tag} {short_title}", key=f"chat_{cid}"):
                    st.session_state.chat_session_id = cid
                    st.session_state.chat_history = msgs
                    st.rerun()

        st.markdown("---")
        if st.button("New Chat", key="new_chat_btn"):
            st.session_state.chat_session_id = create_chat_session(st.session_state.user_id, title="New Chat")
            st.session_state.chat_history = []
            st.rerun()
        if st.button("Clear Current Chat", key="clear_chat_btn"):
            st.session_state.chat_history = []
            save_chat_messages(st.session_state.chat_session_id, [])
            st.rerun()
        if st.button("Delete Current Chat", key="delete_chat_btn"):
            delete_chat(st.session_state.chat_session_id)
            st.session_state.chat_history = []
            st.session_state.chat_session_id = create_chat_session(st.session_state.user_id, title="New Chat")
            st.rerun()

    for msg in st.session_state.chat_history:
        role = "You:" if msg["role"]=="user" else "HealthBot:"
        color = "#1e1e1e" if msg["role"]=="user" else "#2b313e"
        align = "right" if msg["role"]=="user" else "left"
        st.markdown(f"<div style='background:{color};padding:10px;border-radius:12px;margin:8px 0;text-align:{align};color:#fff;'>{role} {msg['content']}</div>", unsafe_allow_html=True)

    user_message = st.chat_input("💬 Type your message...")
    if user_message:
        history = st.session_state.chat_history
        history.append({"role":"user","content":user_message})

        current_id = st.session_state.chat_session_id
        sessions = load_chat_sessions(st.session_state.user_id)
        current_title = [t for (cid, t, tp, m, c, u) in sessions if cid==current_id][0]
        if current_title in ("New Chat", "", None):
            auto_title = generate_session_title(user_message)
            save_chat_messages(current_id, history, title=auto_title)
        else:
            save_chat_messages(current_id, history)

        try:
            gemini_model = genai.GenerativeModel("gemini-2.0-flash-lite-preview")
            response = gemini_model.generate_content(user_message)
            reply = response.text
        except Exception as e:
            reply = f"Gemini API error: {e}"

        history.append({"role":"assistant","content":reply})
        st.session_state.chat_history = history
        save_chat_messages(current_id, history)
        st.rerun()

if selected == "Upload Health Report":
    st.title("Upload Health Report for OCR Analysis")
    uploaded_file = st.file_uploader("Upload health report image", type=["png","jpg","jpeg"], key="ocr_upload")
    if uploaded_file:
        extracted_text = extract_text_from_image(uploaded_file)
        st.subheader("Extracted Text")
        st.text(extracted_text)

        report_session_id = create_chat_session(st.session_state.user_id, title="Report Analysis", chat_type="report")
        history = [{"role":"user","content":extracted_text}]
        try:
            gemini_model = genai.GenerativeModel("gemini-2.0-flash-lite-preview")
            response = gemini_model.generate_content(extracted_text)
            reply = response.text
        except Exception as e:
            reply = f"Gemini API error: {e}"
        history.append({"role":"assistant","content":reply})
        save_chat_messages(report_session_id, history)
        st.session_state.chat_session_id = report_session_id
        st.session_state.chat_history = history
        st.session_state["redirect_to"] = "HealthBot Assistant"
        st.rerun()

if selected == "Past Predictions":
    st.title("Past Predictions History")
    preds = load_predictions(st.session_state.user_id)
    filt = st.selectbox("Filter by disease", ["All","Diabetes","Heart Disease","Parkinson’s Disease"], index=0)
    shown = [p for p in preds if filt=="All" or p[0]==filt]
    if not shown:
        st.info("No past predictions.")
    else:
        for i,(d,vals,res,ts) in enumerate(shown, start=1):
            with st.expander(f"{i}. {d} → {res} ({ts})", expanded=False):
                st.write("**Input Values:**")
                st.code(json.dumps(vals, indent=2))
                st.write("**Result:**", res)


