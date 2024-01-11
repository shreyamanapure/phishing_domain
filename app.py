import pickle
import streamlit as st
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer

with open("PhishingDomainDetection.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def preprocess_input(input_text):
    tokenizer = RegexpTokenizer(r'[A-Za-z]+')
    stemmer = SnowballStemmer("english")
    input_tokens = tokenizer.tokenize(input_text)
    input_stemmed = [stemmer.stem(word) for word in input_tokens]
    input_processed = ' '.join(input_stemmed)
    return input_processed

def main():
    st.title("Phishing Domain Detection")
    user_input = st.text_input("Enter the domain to be checked:")
    if st.button("Check"):
        processed_input = preprocess_input(user_input)
        input_features = vectorizer.transform([processed_input])
        prediction = model.predict(input_features)
        if prediction[0] == 0:
            st.write("This is Non-Malicious Website.")
        else:
            st.write("This is Malicious Website")

if __name__ == "__main__":
    main()
