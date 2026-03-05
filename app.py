import streamlit as st
import joblib

if 'history' not in st.session_state:
    st.session_state.history = []
with st.sidebar:
    st.title("📜 Prediction History")
    if st.session_state.history:
        for item in reversed(st.session_state.history): # Show newest first
            st.write(f"**{item['result']}**: {item['headline'][:30]}...")
            st.caption(f"Confidence: {item['confidence']}")
            st.divider()
    else:
        st.write("No history yet.")
# Load the new model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

st.title("🛡️ Fake News Detector")
st.subheader("Enter a news headline below to check its authenticity")

user_input = st.text_area("Paste Headline or Article Text Here:", height=150)

if st.button("Analyze News"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        # Transform and Predict
        vectorized_text = vectorizer.transform([user_input])
        prediction = model.predict(vectorized_text)
        
        # New: Calculate Confidence (Probability)
        prediction = model.predict(vectorized_text)
        confidence = max(prediction[0]) * 100
        
        # Display Result (1=REAL, 0=FAKE for WELFake)
        if prediction[0] == 0:
            st.success(f"✅ This news appears to be REAL. (Confidence: {confidence:.2f}%)")
        else:
            st.error(f"🚨 This news appears to be FAKE. (Confidence: {confidence:.2f}%)")
           # Add this inside your "Analyze News" button logic
        result_text = "REAL" if prediction[0] == 0 else "FAKE"
        st.session_state.history.append({"headline": user_input, "result": result_text, "confidence": f"{confidence:.2f}%"})
            # --- Footer & Information ---
st.divider()

col1, col2 = st.columns(2)


with col1:
    st.info("""
    ### 💡 How to use:
    * Paste the full headline or first paragraph.
    * Headlines with extreme punctuation (!!!) often score as Fake.
    * Check the confidence—anything near 50% is a 'toss-up'.
    """)

with col2:
    st.warning("""
    ### ⚠️ Disclaimer:
    This AI tool is for educational purposes. Machine learning models can make mistakes. Always verify news with multiple trusted sources.
    """)
