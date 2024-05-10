import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
dataset_path = 'Restaurant reviews.csv'
df = pd.read_csv(dataset_path)

# Text Preprocessing: Clean and tokenize the reviews
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Download NLTK resources
import nltk
nltk.download('stopwords')
nltk.download('punkt')

def preprocess_text(text):
    # Check if the input is a string
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stop words
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        return ' '.join(tokens)
    else:
        # If the input is not a string, return an empty string or handle it based on your needs
        return ''


#'Review' is the column containing raw reviews
df['Cleaned_Review'] = df['Review'].apply(preprocess_text)


# Convert 'Rating' to numeric
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')

# Handle NaN values in 'Rating' column
df = df.dropna(subset=['Rating'])

# Create a threshold for sentiment classification
threshold = 3.5

# Convert numerical ratings to sentiment labels
df['Sentiment_Label'] = df['Rating'].apply(lambda x: 'positive' if x >= threshold else 'negative')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Cleaned_Review'], df['Sentiment_Label'], test_size=0.2, random_state=42)

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_tfidf, y_train)

# Evaluate the model
y_pred = nb_classifier.predict(X_test_tfidf)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))

# Predict sentiments for all reviews in the dataset
df['Predicted_Sentiment'] = nb_classifier.predict(tfidf_vectorizer.transform(df['Cleaned_Review']))

# Group by restaurant name and calculate the average rating
restaurant_stats = df.groupby('Restaurant').agg({
    'Rating': 'mean',
    'Predicted_Sentiment': lambda x: (x == 'positive').mean()  # Calculate the percentage of positive sentiments
}).reset_index()

# Sort by average rating in ascending order
top_restaurants = restaurant_stats.sort_values('Rating', ascending=True).head(10)

# Display the top restaurants
print("Top 10 Recommended Restaurants:")
print(top_restaurants[['Restaurant', 'Rating', 'Predicted_Sentiment']])

