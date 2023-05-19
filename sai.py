# print("sai0")
# print('''sai0''')
# a=45
# b=56
# print("sai",a+b)
# ashish=[23,"sai","asit",4.9,False]
# print(ashish)
# print(ashish[1])
# print(ashish[1:4])
# print(ashish[1:5])


# ash=[23,67,56,89,34,90,32]
# for i in ash:
#     print(i)

# print(ash[1:6])


# for i in range(100):
#     print(i)


# for i in range(50,100):
#     print(i)s
# i = 0
# while i < 50:
#   print(i)
#   i += 1
# def sai():
#     print("hello")
#     if True:
#         print("Bye")
#         print("sa")
#     else:
#         print("sai prasad")
# sai()
# print("cfr")
import nltk
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

# Download required NLTK resources
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load movie reviews dataset
reviews = [(list(movie_reviews.words(fileid)), category)
           for category in movie_reviews.categories()
           for fileid in movie_reviews.fileids(category)]

# Shuffle the reviews
import random
random.shuffle(reviews)

# Preprocess the reviews
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(review):
    review = ' '.join(review)
    tokens = word_tokenize(review.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing to the reviews
preprocessed_reviews = [(preprocess(review), sentiment) for review, sentiment in reviews]

# Split the dataset into training and testing sets
train_reviews, test_reviews, train_sentiments, test_sentiments = train_test_split(
    [review for review, _ in preprocessed_reviews],
    [sentiment for _, sentiment in preprocessed_reviews],
    test_size=0.2,
    random_state=42
)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()
train_features = vectorizer.fit_transform(train_reviews)
test_features = vectorizer.transform(test_reviews)

# Train the classifier
classifier = LinearSVC()
classifier.fit(train_features, train_sentiments)

# Make predictions
predictions = classifier.predict(test_features)

# Evaluate the classifier
print("Accuracy:", accuracy_score(test_sentiments, predictions))
print(classification_report(test_sentiments, predictions))
