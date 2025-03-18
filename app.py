import streamlit as st
import joblib
import os

# Load model and vectorizer safely
try:
    vectorizer_path = "vectorizer.jb"
    model_path = "lr_model.jb"

    if os.path.exists(vectorizer_path) and os.path.exists(model_path):
        vectorizer = joblib.load(vectorizer_path)
        model = joblib.load(model_path)
    else:
        st.error("Model files not found. Please check if 'vectorizer.jb' and 'lr_model.jb' exist.")
        st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Streamlit UI
st.title("📰 Fake News Detector")
st.write("Enter a news article below to check whether it is **Fake** or **Real**.")

inputn = st.text_area("📝 News Article:", "")

if st.button("🔍 Check News"):
    if inputn.strip():
        try:
            # Transform input and predict
            transform_input = vectorizer.transform([inputn])
            prediction = model.predict(transform_input)

            # Adjust class labels if needed (ensure 1=Real, 0=Fake)
            if prediction[0] == 1:
                st.success("✅ The News is **Real**! ")
            else:
                st.error("❌ The News is **Fake**! ")
        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.warning("⚠️ Please enter some text to analyze.")
