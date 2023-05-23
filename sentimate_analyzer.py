import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Step 1: Dataset Selection
df = pd.read_csv(r'D:\python of ml\numpy\IMDB Dataset.csv')  # Replace 'movie_reviews.csv' with your dataset filename

# Step 2: Data Preprocessing
# Assuming your dataset has 'review' column for the movie reviews and 'sentiment' column for their corresponding labels
reviews = df['review']
sentiments = df['sentiment']

# Step 3: Feature Extraction
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(reviews)

# Step 4: Model Training
X_train, X_test, y_train, y_test = train_test_split(X, sentiments, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred, pos_label='positive')
recall = metrics.recall_score(y_test, y_pred, pos_label='positive')
f1_score = metrics.f1_score(y_test, y_pred, pos_label='positive')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)

# Step 6: Deployment and Testing
while True:
    user_input = input("Enter a movie review: ")
    if user_input == "exit":
        break
    user_input_vectorized = vectorizer.transform([user_input])
    prediction = model.predict(user_input_vectorized)
    print("Predicted sentiment:", prediction[0])
