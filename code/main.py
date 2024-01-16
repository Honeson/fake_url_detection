import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import re

import tldextract
from urllib.parse import urlparse
from googlesearch import search
import pickle
import joblib

import streamlit as st


st.title('Fake Urls Detection by Yusuf (UNIPORT)')


def letter_count(url):
    letters = []
    for i in url:
        if i.isalpha():
            letters.append(i)
    return letters


def digit_count(url):
    digits = []
    for i in url:
        if i.isnumeric():
            digits.append(i)
    return digits

def google_index(url):
    site = search(url, 5)
    return True if site else False


def extract_features(url):
    parsed_url = urlparse(url)
    # Extract domain features using tldextract
    domain_extract = tldextract.extract(url)
    domain = domain_extract.domain
    suffix = domain_extract.suffix
    letters = letter_count(url)
    digits = digit_count(url)
    google_searchable = google_index(url)

    # Features to extract
    features = {
        'url_length': len(url), #Length of URL
        'domain_length': len(domain), #Length of domain
        'dot_count': domain.count('.'), #Number of dots in the domain
        'is_ip_address': domain.replace('.', '').isdigit(), #IP address in the domain
        'special_chars_in_domain': any(char.isnumeric() or not char.isalnum() for char in domain), #Presence of special characters in the domain
        'tld_length': len(suffix), #Length of the top-level domain (e.g., '.com', '.org')
        'hyphen_in_domain': '-' in domain, #Presence of hyphen in the domain
        'at_symbol': '@' in parsed_url.netloc, #Presence of '@' in the URL
        'https': 'https' in parsed_url.scheme, #Check if site is secure with https
        'letter_count': len(letters), #Length of alphabets in url
        'digit_count': len(digits), #Length of numbers(digits) in url
        'google_search': google_searchable
    }

    return features


#Check is the  url is valid or not
def is_valid_url(url):
    # Regular expression pattern for a more lenient URL validation
    pattern = re.compile(r'^(https?://)?(www\.)?[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    url = str(url)
    # Check if the provided URL matches the pattern
    return bool(pattern.match(url))


def model_predict(model, url):
    class_mapping = {
        0: 'Scam',
        1: 'Not Scam'
    
    }
    formated_url = extract_features(url)
    
    
    prediction = model.predict(np.array(list(formated_url.values())).reshape(1, -1))[0]
    prediction = class_mapping.get(prediction, 'Unkown')
    return prediction
    

def main():

    user_model = st.selectbox('Choose your model from the list', ['Select Model', 'Logistic Regression', 'Decision Tree', 'Random Forest'])

    if user_model != 'Select Model':
        # Load the model from the .h5 file
        model_path = None

        if user_model == 'Logistic Regression':
            model_path = 'trained_model/Logistic_Regression.h5'
        elif user_model == 'Decision Tree':
            model_path = 'trained_model/Decision_Tree.h5'
        elif user_model == 'Random Forest':
            model_path = 'trained_model/Random_Forest.h5'

        if model_path:
            model = joblib.load(model_path)
            input_url = st.text_input('Enter a website URL and our AI will tell you if it is scam or not')
            

            if st.button('Check URL'):
                if input_url:
                    st.write('Hang tight while we check if the URL is scam or not scam')

                    is_valid = is_valid_url(input_url)
                    if is_valid:

                        prediction = model_predict(model, input_url)

                        st.write(f'Entered URL: {input_url}')
                        st.write(f'Result: {prediction}')
                    else:
                        st.warning('Inavlid url detected: Please enter a valid url...')

if __name__ == "__main__":
    main()
