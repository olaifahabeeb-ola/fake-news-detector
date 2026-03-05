import streamlit as st
import joblib

# Initialize history session state
if 'history' not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.title("📜 Prediction History")
    if st.session_state.history:
        for item in reversed(st.session_state.history): # Show newest first
            st.write(f"**{item['result']}**: {item['headline'][:30]}...")
            st.divider()
    else:
        st.write("No history yet.")

# Load the model and vectorizer
# Ensure these files are uploaded to the same folder on GitHub
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
        
        # Display Result (0=REAL, 1=FAKE based on your code logic)
        if prediction[0] == 0:
            result_text = "REAL"
            st.success(f"✅ This news appears to be {result_text}.")
        else:
            result_text = "FAKE"
            st.error(f"🚨 This news appears to be {result_text}.")
            
        # Update History
        st.session_state.history.append({"headline": user_input, "result": result_text})

# --- Footer & Information ---
st.divider()
col1, col2 = st.columns(2)

with col1:
    st.info("""
    ### 💡 How to use:
    * Paste the full headline or first paragraph.
    * Headlines with extreme punctuation (!!!) often score as Fake.
    """)

with col2:
    st.warning("""
    ### ⚠️ Disclaimer:
    This AI tool is for educational purposes. Always verify news with multiple trusted sources.
    """)
