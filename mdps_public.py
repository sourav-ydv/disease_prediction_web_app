# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 17:00:19 2025

@author: sksou
"""

import streamlit as st
import pickle
import psycopg2
import psycopg2.extras
import hashlib
import json
from datetime import datetime
from streamlit_option_menu import option_menu
import google.generativeai as genai
from PIL import Image
import pytesseract

DB_URL = st.secrets["DATABASE_URL"]

def db_conn():
    return psycopg2.connect(DB_URL, cursor_factory=psycopg2.extras.DictCursor)


def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def create_user(username: str, password: str) -> bool:
    con = None
    try:
        con = db_conn(); cur = con.cursor()
        cur.execute(
            "INSERT INTO users(username,password_hash) VALUES(%s,%s)",
            (username, hash_password(password))
        )
        con.commit()
        return True
    except psycopg2.IntegrityError:
        if con:
            con.rollback()
        return False
    finally:
        if con:
            con.close()

def login_user(username: str, password: str):
    con = None
    try:
        con = db_conn(); cur = con.cursor()
        cur.execute("SELECT id, password_hash FROM users WHERE username=%s", (username,))
        row = cur.fetchone()
        if row and row["password_hash"] == hash_password(password):
            return row["id"]
        return None
    finally:
        if con:
            con.close()

def save_prediction(user_id: int, disease: str, inputs: dict, result: str):
    con = None
    try:
        con = db_conn(); cur = con.cursor()
        cur.execute(
            "INSERT INTO predictions(user_id, disease, input_values, result, timestamp) VALUES (%s, %s, %s, %s, %s)",
            (user_id, disease, json.dumps(inputs), result, datetime.now())
        )
        con.commit()
    finally:
        if con:
            con.close()

def load_predictions(user_id: int):
    con = None
    try:
        con = db_conn(); cur = con.cursor()
        cur.execute("""
            SELECT disease, input_values, result, timestamp
            FROM predictions
            WHERE user_id=%s
            ORDER BY timestamp DESC
        """, (user_id,))
        rows = cur.fetchall()
        results = []
        for r in rows:
            vals = r["input_values"]
            if isinstance(vals, str):
                try:
                    vals = json.loads(vals)
                except Exception:
                    vals = []
            results.append((r["disease"], vals, r["result"], r["timestamp"]))
        return results
    finally:
        if con:
            con.close()

def create_chat_session(user_id: int, title="New Chat", chat_type="normal") -> int:
    con = None
    try:
        con = db_conn(); cur = con.cursor()
        now = datetime.now()
        cur.execute(
            "INSERT INTO chats(user_id,title,type,messages,created_at,updated_at) VALUES(%s,%s,%s,%s,%s,%s) RETURNING id",
            (user_id, title, chat_type, json.dumps([]), now, now)
        )
        chat_id = cur.fetchone()["id"]
        con.commit()
        return chat_id
    finally:
        if con:
            con.close()

def load_chat_sessions(user_id: int):
    con = None
    try:
        con = db_conn(); cur = con.cursor()
        cur.execute("""
            SELECT id, title, type, messages, created_at, updated_at
            FROM chats
            WHERE user_id=%s
            ORDER BY updated_at DESC
        """, (user_id,))
        rows = cur.fetchall()
        sessions = []
        for r in rows:
            msgs = r["messages"]
            if isinstance(msgs, str):
                try:
                    msgs = json.loads(msgs)
                except Exception:
                    msgs = []
            sessions.append((r["id"], r["title"], r["type"], msgs, r["created_at"], r["updated_at"]))
        return sessions
    finally:
        if con:
            con.close()

def save_chat_messages(chat_id: int, messages: list, title=None):
    con = None
    try:
        con = db_conn(); cur = con.cursor()
        now = datetime.now()
        if title is not None:
            cur.execute(
                "UPDATE chats SET messages=%s, updated_at=%s, title=%s WHERE id=%s",
                (json.dumps(messages), now, title, chat_id)
            )
        else:
            cur.execute(
                "UPDATE chats SET messages=%s, updated_at=%s WHERE id=%s",
                (json.dumps(messages), now, chat_id)
            )
        con.commit()
    finally:
        if con:
            con.close()

def delete_chat(chat_id: int):
    con = None
    try:
        con = db_conn(); cur = con.cursor()
        cur.execute("DELETE FROM chats WHERE id=%s", (chat_id,))
        con.commit()
    finally:
        if con:
            con.close()

def load_chat_by_id(chat_id: int):
    con = None
    try:
        con = db_conn(); cur = con.cursor()
        cur.execute("SELECT messages FROM chats WHERE id=%s", (chat_id,))
        row = cur.fetchone()
        return json.loads(row["messages"]) if row else []
    finally:
        if con:
            con.close()

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
        user_input_d = {
            "Pregnancies": int(Pregnancies),
            "Glucose": int(Glucose),
            "BloodPressure": int(BloodPressure),
            "SkinThickness": int(SkinThickness),
            "Insulin": int(Insulin),
            "BMI": float(BMI),
            "DiabetesPedigreeFunction": float(DiabetesPedigreeFunction),
            "Age": int(Age)
        }

        diab_prediction = diabetes_model.predict([list(user_input_d.values())])
        diab_status = 'likely to have diabetes' if diab_prediction[0] == 1 else 'not diabetic'

        if diab_prediction[0] == 1:
            st.error('The person is likely to have diabetes.')
        else:
            st.success('The person is not diabetic.')

        st.session_state['last_prediction'] = {"disease": "Diabetes", "input": user_input_d, "result": diab_status}
        save_prediction(st.session_state.user_id, "Diabetes", user_input_d, diab_status)
      
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
        user_input_h = {
            "Age": int(age),
            "Sex": int(sex),
            "Chest Pain Type": int(cp),
            "Resting Blood Pressure": int(trestbps),
            "Cholesterol": int(chol),
            "Fasting Blood Sugar": int(fbs),
            "Resting ECG": int(restecg),
            "Max Heart Rate": int(thalach),
            "Exercise Angina": int(exang),
            "Oldpeak": float(oldpeak),
            "Slope": int(slope),
            "Major Vessels": int(ca),
            "Thalassemia": int(thal)
        }

        heart_prediction = heart_model.predict([list(user_input_h.values())])
        heart_status = 'likely to have heart disease' if heart_prediction[0] == 1 else 'does not have any heart disease'

        if heart_prediction[0] == 1:
            st.error('The person is likely to have heart disease.')
        else:
            st.success('The person does not have any heart disease.')

        st.session_state['last_prediction'] = {"disease": "Heart Disease", "input": user_input_h, "result": heart_status}
        save_prediction(st.session_state.user_id, "Heart Disease", user_input_h, heart_status)
        
if selected == "Parkinson’s Prediction":
    st.title("Parkinson’s Disease Prediction using ML")

    col1, col2, col3, col4 = st.columns(4)  
    
    with col1:
        fo = st.text_input('Average Vocal Fundamental Frequency (Hz)')
        Jitter_Abs = st.text_input('Jitter (Abs)')
        Shimmer = st.text_input('Shimmer')
        APQ = st.text_input('APQ')
        RPDE = st.text_input('RPDE')
        D2 = st.text_input('D2')
    
    with col2:
        fhi = st.text_input('Maximum Vocal Fundamental Frequency (Hz)')
        RAP = st.text_input('RAP')
        Shimmer_dB = st.text_input('Shimmer (dB)')
        DDA = st.text_input('DDA')
        DFA = st.text_input('DFA')
        PPE = st.text_input('PPE')
    
    with col3:
        flo = st.text_input('Minimum Vocal Fundamental Frequency (Hz)')
        PPQ = st.text_input('PPQ')
        APQ3 = st.text_input('APQ3')
        NHR = st.text_input('NHR')
        spread1 = st.text_input('Spread1')

    with col4:
        Jitter_percent = st.text_input('Jitter (%)')
        DDP = st.text_input('DDP')
        APQ5 = st.text_input('APQ5')
        HNR = st.text_input('HNR')
        spread2 = st.text_input('Spread2')
        
    if st.button("Parkinson's Test Result"):
        try:
            user_input = {
                "fo": float(fo), "fhi": float(fhi), "flo": float(flo),
                "Jitter(%)": float(Jitter_percent), "Jitter(Abs)": float(Jitter_Abs),
                "RAP": float(RAP), "PPQ": float(PPQ), "DDP": float(DDP),
                "Shimmer": float(Shimmer), "Shimmer(dB)": float(Shimmer_dB),
                "APQ3": float(APQ3), "APQ5": float(APQ5), "APQ": float(APQ),
                "DDA": float(DDA), "NHR": float(NHR), "HNR": float(HNR),
                "RPDE": float(RPDE), "DFA": float(DFA),
                "Spread1": float(spread1), "Spread2": float(spread2),
                "D2": float(D2), "PPE": float(PPE)
            }

            parkinsons_prediction = parkinsons_model.predict([list(user_input.values())])
            park_status = "likely to have Parkinson’s Disease" if parkinsons_prediction[0] == 1 else "does not have Parkinson’s Disease"

            if parkinsons_prediction[0] == 1:
                st.error("The person likely has Parkinson’s Disease.")
            else:
                st.success("The person is healthy.")

            st.session_state['last_prediction'] = {"disease": "Parkinson’s Disease", "input": user_input, "result": park_status}
            save_prediction(st.session_state.user_id, "Parkinson’s Disease", user_input, park_status)

        except ValueError:
            st.error("Please fill all fields with valid numeric values.")


if selected == 'HealthBot Assistant':
    st.title("AI HealthBot Assistant")

    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    except Exception:
        st.error("Gemini API key missing or invalid.")
        st.stop()

    # --- helper: build prediction context ---
    def build_prediction_context():
        if "last_prediction" not in st.session_state:
            return None
        pred = st.session_state["last_prediction"]
        disease = pred["disease"]
        result = pred["result"]
        inputs = pred["input"]

        input_details = "\n".join([f"- {k}: {v}" for k, v in inputs.items()])

        context = f"""
        The user previously made a prediction for **{disease}**.
        Result: {result}.
        Input values were:
        {input_details}
        Explain what these numbers mean, what risk factors they indicate,
        and provide suggestions in clear, user-friendly language.
        """
        return context

    # --- ensure chat session ---
    if "chat_session_id" not in st.session_state:
        st.session_state.chat_session_id = create_chat_session(st.session_state.user_id, title="New Chat")
        st.session_state.chat_history = []

    # --- sidebar: chat list ---
    with st.sidebar:
        st.markdown("Chats")
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
        if st.button("➕ New Chat", key="new_chat_btn"):
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

    # --- render chat history ---
    for msg in st.session_state.chat_history:
        role = "You:" if msg["role"]=="user" else "HealthBot:"
        color = "#1e1e1e" if msg["role"]=="user" else "#2b313e"
        align = "right" if msg["role"]=="user" else "left"
        st.markdown(f"<div style='background:{color};padding:10px;border-radius:12px;margin:8px 0;text-align:{align};color:#fff;'>{role} {msg['content']}</div>", unsafe_allow_html=True)

    # --- user input ---
    user_message = st.chat_input("💬 Type your message...")
    if user_message:
        history = st.session_state.chat_history
        history.append({"role":"user","content":user_message})

        current_id = st.session_state.chat_session_id
        sessions = load_chat_sessions(st.session_state.user_id)
        current_title = [t for (cid, t, tp, m, c, u) in sessions if cid==current_id][0]
        if current_title in ("New Chat", "", None):
            auto_title = user_message[:25] + ("..." if len(user_message) > 25 else "")
            save_chat_messages(current_id, history, title=auto_title)
        else:
            save_chat_messages(current_id, history)

        # --- build final prompt with prediction context if needed ---
        context = build_prediction_context()
        if context and any(word in user_message.lower() for word in ["prediction", "numbers", "result", "last test"]):
            prompt = f"""
            User asked: {user_message}

            Here is their last prediction context:
            {context}
            """
        else:
            prompt = user_message

        try:
            gemini_model = genai.GenerativeModel("gemini-2.0-flash-lite-preview")
            response = gemini_model.generate_content(prompt)
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

    filt = st.selectbox("Filter by disease", ["All", "Diabetes", "Heart Disease", "Parkinson’s Disease"], index=0)
    shown = [p for p in preds if filt == "All" or p[0] == filt]

    if not shown:
        st.info("No past predictions.")
    else:
        for i, (d, vals, res, ts) in enumerate(shown, start=1):
            with st.expander(f"{i}. {d} → {res} ({ts})", expanded=False):
                st.write("**Input Values:**")
                if isinstance(vals, dict):
                    for name, value in vals.items():
                        st.write(f"- **{name}:** {value}")
                else:
                    st.code(json.dumps(vals, indent=2))
                st.write("**Result:**", res)


