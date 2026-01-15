import streamlit as st
import json
from data_preprocessor import preprocesser
import torch
from lstm_model import LSTMModelV1
import torch.nn.functional as F
import pandas as pd
from datetime import datetime
import os

#-----------------SAVE PREDICTION ----------------

CSV_PATH = "predictions_log.csv"

label_map = {
    0: "Irrelevant",
    1: "Negative",
    2: "Neutral",
    3: "Positive"
}


def save_prediction_to_csv(text, label, confidence):
    data = {
        "tweet": text,
        "predicted_label_id": label,
        "predicted_sentiment": label_map[label],
        "confidence_percent": f"{confidence*100:.2f}%",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    df_new = pd.DataFrame([data])

    if os.path.exists(CSV_PATH):
        df_existing = pd.read_csv(CSV_PATH)
        df_final = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_final = df_new

    df_final.to_csv(CSV_PATH, index=False)




# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="💬",
    layout="centered"
)
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #141E30, #243B55);
    color: white;
}

.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: bold;
    margin-bottom: 10px;
}

.sub-title {
    text-align: center;
    font-size: 18px;
    color: #dcdcdc;
    margin-bottom: 40px;
}

.input-box {
    border-radius: 12px;
    padding: 12px;
    font-size: 18px;
}

.predict-btn {
    background-color: #00c853;
    color: white;
    font-size: 18px;
    padding: 10px;
    border-radius: 10px;
    width: 100%;
}

.result-card {
    background: rgba(255, 255, 255, 0.1);
    padding: 25px;
    border-radius: 15px;
    margin-top: 30px;
    text-align: center;
}

.footer {
    text-align: center;
    color: #bdbdbd;
    margin-top: 50px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
with open('./data/vocab.json', 'r') as f:
    vocab = json.load(f)

save_model_path = './models/modelV3.pth'
device = torch.device("cpu")

@st.cache_resource
def load_model(model_path):
    model = LSTMModelV1(
        input_size=len(vocab),
        embedding_dim=100,
        hidden_size=64,
        layer_dim=4,
        drop_out=0.2,
        bi_directional=True,
        output_dim=4
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model(save_model_path)

categories = {
    0: '<h2 style="color:gray;">😴 Irrelevant</h2>',
    1: '<h2 style="color:#ff5252;">😠 Negative</h2>',
    2: '<h2 style="color:#40c4ff;">😐 Neutral</h2>',
    3: '<h2 style="color:#69f0ae;">😊 Positive</h2>'
}



def predict(model, text):
    X = torch.tensor(text, dtype=torch.long).unsqueeze(0)

    with torch.inference_mode():
        logits = model(X.to(device))
        probs = F.softmax(logits, dim=1)
        confidence, y_label = torch.max(probs, dim=1)

    return int(y_label), float(confidence)


# ---------------- MAIN APP ----------------
def main():
    st.markdown('<div class="main-title">💬 Sentiment Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Twitter Sentiment Classifier</div>', unsafe_allow_html=True)

    text = st.text_input(
        "Enter text",
        placeholder="Type a tweet or sentence...",
        label_visibility="collapsed"
    )

    if st.button("🔍 Analyze Sentiment", use_container_width=True):
        if text.strip() == "":
            st.warning("⚠️ Please enter some text")
        else:
            processed_text = preprocesser(text)
            predicted, confidence = predict(model, processed_text)

            # Save history
            st.session_state.history.append({
                "Tweet": text,
                "Sentiment": categories[predicted],
                "Confidence": f"{confidence*100:.2f}%"
            })

            # Result card
            st.markdown(
                f"""
                <div class="result-card">
                    <h3>Prediction Result</h3>
                    {categories[predicted]}
                    <p><b>Model Confidence:</b> {confidence*100:.2f}%</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Confidence bar
            st.progress(confidence)
            
            
            # Save prediction to CSV
            save_prediction_to_csv(text, predicted, confidence)

    # ---------------- TWEET HISTORY ----------------
    if st.session_state.history:
        st.markdown("## 🧾 Tweet History")

        for i, item in enumerate(reversed(st.session_state.history[-5:]), 1):
            st.markdown(
                f"""
                <div class="result-card">
                    <b>{i}. Tweet:</b> {item["Tweet"]}<br>
                    <b>Sentiment:</b> {item["Sentiment"]}<br>
                    <b>Confidence:</b> {item["Confidence"]}
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown('<div class="footer">🚀 Built with PyTorch & Streamlit</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
