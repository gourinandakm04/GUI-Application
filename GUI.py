import tkinter as tk
from tkinter import messagebox
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import string

# Function to load the trained model and vectorizer
def load_model():
    model = joblib.load("spam_classifier.pkl")  # Load your trained model
    vectorizer = joblib.load("vectorizer.pkl")  # Load your vectorizer (if needed)
    return model, vectorizer

# Function to preprocess the email text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = text.lower()  # Convert text to lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    words = text.split()  # Split text into words
    words = [word for word in words if word not in stop_words]  # Remove stop words
    return ' '.join(words)

# Function to classify the email as spam or ham
def classify_email():
    email_text = email_input.get("1.0", "end-1c")  # Get the input email text
    if email_text.strip() == "":  # If input is empty
        messagebox.showwarning("Input Error", "Please enter some text to classify.")
        return

    # Preprocess the input email text
    processed_text = preprocess_text(email_text)

    # Load the model and vectorizer
    model, vectorizer = load_model()

    # Transform the email text using the vectorizer
    email_features = vectorizer.transform([processed_text])

    # Make a prediction
    prediction = model.predict(email_features)

    # Display the result
    if prediction == 1:  # 1 for spam, 0 for ham (depending on the model)
        result_label.config(text="Prediction: Spam", fg="red")
    else:
        result_label.config(text="Prediction: Ham", fg="green")

# Set up the Tkinter GUI window
root = tk.Tk()
root.title("Spam or Ham Classifier")

# Set up the email input section
email_label = tk.Label(root, text="Enter email text:")
email_label.pack(pady=10)

email_input = tk.Text(root, height=10, width=40)
email_input.pack(pady=10)

# Set up the classify button
classify_button = tk.Button(root, text="Classify", command=classify_email)
classify_button.pack(pady=10)

# Set up the result label
result_label = tk.Label(root, text="Prediction: ", font=("Arial", 14))
result_label.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import joblib

# Example data (you would replace this with your actual dataset)
emails = ["Free money now!", "Meeting at 3 PM tomorrow.", "Congratulations! You won a prize!", "Let's have lunch."]
labels = [1, 0, 1, 0]  # 1 = Spam, 0 = Ham

# Split the data
X_train, X_test, y_train, y_test = train_test_split(emails, labels, test_size=0.2, random_state=42)

# Convert emails to feature vectors
vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)

# Train a model
model = MultinomialNB()
model.fit(X_train_vectors, y_train)

# Save the model and vectorizer
joblib.dump(model, "spam_classifier.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
# Function to predict if an email is spam or ham
def predict_spam_or_ham(email_text):
    # Preprocess the email text
    processed_text = preprocess_text(email_text)

    # Load the model and vectorizer
    model, vectorizer = load_model()

    # Transform the email text using the vectorizer
    email_features = vectorizer.transform([processed_text])

    # Make the prediction
    prediction = model.predict(email_features)

    # Return the result based on the prediction
    if prediction == 1:  # 1 represents spam
        return "Spam"
    else:
        return "Ham"

# Example of using the function:
email_text = "Congratulations, you have won a free iPhone! Claim it now!"
prediction = predict_spam_or_ham(email_text)
print(f"The email is: {prediction}")
