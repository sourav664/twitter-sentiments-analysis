import streamlit as st
import json
from data_preprocessor import preprocesser
import torch
from lstm_model import LSTMModelV1

with open('./data/vocab.json', 'r') as f:
    vocab = json.load(f)


save_model_path = './models/modelV2.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMModelV1(input_size=len(vocab),
                  embedding_dim=100,
                  hidden_size=48,
                  layer_dim=3,
                  drop_out = 0.2,
                  bi_directional = True,
                  output_dim=4
                  ).to(device)

model.load_state_dict(torch.load(save_model_path))

categories = {
    0: '<h2 style="color:gray;">Irrelevant 💤</h2>',
    1: '<h2 style="color:red;">Negative 😠</h2>',
    2: '<h2 style="color:blue;">Neutral 😐</h2>',
    3: '<h2 style="color:green;">Positive 😊</h2>'
}


# categories = {
#     0: "Irrelevant",
#     1: "Negative",
#     2: "Neutral",
#     3: "Positive"}

def predict(model, text):
    X = torch.tensor(text, dtype=torch.long).unsqueeze(0)
    
    model.eval()
    with torch.inference_mode():
        X = X.to(device)
        logits = model(X)
        y_label = int(torch.argmax(logits, dim=1))
   
   
    return y_label

def main():
    st.title("Sentiment Analysis")
    text = st.text_input("Enter text")
    if st.button("Predict"):
        text = preprocesser(text)
        predicted = predict(model, text)
        st.markdown(categories[predicted], unsafe_allow_html=True)

if __name__ == "__main__":
    main()