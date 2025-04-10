# app.py

import streamlit as st
from predict_external_news import predict_news, get_majority_vote
st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detector")
st.markdown("Enter a news article below and get predictions from multiple ML models.")

# Text input
news_input = st.text_area("Paste your news article here:", height=200)

if st.button("Detect"):
    if not news_input.strip():
        st.warning("Please enter a news article to analyze.")
    else:
        with st.spinner("Analyzing..."):
            results = predict_news(news_input)
            majority = get_majority_vote(results)
            final_label = results["Voting Classifier"]["label"]

        st.subheader("🔍 Model Predictions")
        for model, res in results.items():
            confidence = f"{res['confidence']:.2f}" if isinstance(res['confidence'], float) else res['confidence']
            st.write(f"**{model}**: {res['label']} (Confidence: {confidence})")

        if final_label == "Fake News":
            st.error(f"🛑 Final Decision (Voting Classifier): {final_label}")
        else:
            st.success(f"🟢 Final Decision (Voting Classifier): {final_label}")

        st.info(f"🔵 Manual Majority Vote: {majority}")

st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "© 2025 Fake News Detector | Built with ❤️ by Santosh"
    "</div>",
    unsafe_allow_html=True
)
