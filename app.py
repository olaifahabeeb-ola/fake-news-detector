import streamlit as st
import joblib

# Initialize history session state
if 'history' not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.title("📜 Prediction History")
    if st.session_state.history:
        for item in reversed(st.session_state.history): # Show newest first
            st.write(f"**{item['result']}** ({item['confidence']}): {item['headline'][:30]}...")
            st.divider()
    else:
        st.write("No history yet.")

# Load the model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

st.title("🛡️ Fake News Detector")
st.subheader("Enter a news headline below to check its authenticity")

user_input = st.text_area("Paste Headline or Article Text Here:", height=150)

if st.button("Analyze News"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        # 1. Transform the input text
        vectorized_text = vectorizer.transform([user_input])
        
        # 2. Make Prediction
        prediction = model.predict(vectorized_text)
        
        # 3. Calculate Confidence (Probability)
        # This works because your training script uses LogisticRegression
        probability = model.predict_proba(vectorized_text)
        confidence_val = max(probability[0]) * 100
        confidence_str = f"{confidence_val:.2f}%"
        
        # 4. Display Result (Matches your training: 1=REAL, 0=FAKE)
        if prediction[0] == 1:
            result_text = "REAL"
            st.success(f"✅ This news appears to be {result_text}. (Confidence: {confidence_str})")
        else:
            result_text = "FAKE"
            st.error(f"🚨 This news appears to be {result_text}. (Confidence: {confidence_str})")
            
        # 5. Update History
        st.session_state.history.append({
            "headline": user_input, 
            "result": result_text, 
            "confidence": confidence_str
        })

# --- Footer & Information ---
st.divider()
col1, col2 = st.columns(2)

with col1:
    st.info("""
    ### 💡 How to use:
    * Paste the full headline or first paragraph.
    * Check the confidence—anything near 50% is a 'toss-up'.
    """)

with col2:
    st.warning("""
    ### ⚠️ Disclaimer:
    This AI tool is for educational purposes. Machine learning models can make mistakes.
    """)


