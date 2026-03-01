import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# 1. Load your dataset
df = pd.read_csv('welfake_data.csv')
df = df.dropna() # Clean up any empty rows

# 2. Setup Features and Labels
# In WELFake: 0 = FAKE, 1 = REAL
X = df['text'] 
y = df['label']

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Vectorize (Turn text into numbers)
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)

# 5. Train a Logistic Regression model (supports confidence scores!)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# 6. Save the new "Brain" and "Dictionary"
# 'compress=3' is a good balance between size and speed
joblib.dump(model, 'fake_news_model.pkl', compress=3)
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Success! New model trained and saved.")