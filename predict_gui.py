import streamlit as st
from predict_external_news import predict_news, get_majority_vote

st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detector")
st.markdown("Enter a news article below and get predictions from multiple ML models.")

# Input box
news_input = st.text_area("Paste your news article here:", height=200)

if st.button("Detect"):
    if not news_input.strip():
        st.warning("Please enter a news article to analyze.")
    else:
        with st.spinner("Analyzing..."):
            results = predict_news(news_input)
            majority = get_majority_vote(results)

        st.subheader("🔍 Model Predictions")
        for model, res in results.items():
            st.write(f"**{model}**: {res['label']} (Confidence: {res['confidence']})")

        st.success(f"🟢 Final Decision (Voting Classifier): {results['Voting Classifier']['label']}")
        st.info(f"🔵 Manual Majority Vote: {majority}")
