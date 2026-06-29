import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

st.set_page_config(
    page_title="Spam Detection System",
    page_icon="🛡️",
    layout="centered"
)

st.markdown("""
<style>

.stApp{
    background: linear-gradient(
        135deg,
        #0F172A 0%,
        #172554 50%,
        #312E81 100%
    );
}

.main .block-container{

    max-width:700px;

    margin-top:8vh;

    padding:40px;

    border-radius:25px;

    background:rgba(255,255,255,0.05);

    backdrop-filter:blur(15px);

    box-shadow:
        0 8px 32px rgba(0,0,0,0.35);

}

/* Hide Streamlit UI */

header{
    visibility:hidden;
}

footer{
    visibility:hidden;
}

#MainMenu{
    visibility:hidden;
}

/* TITLE */

h1{

    text-align:center;

    color:white !important;

    font-size:3rem;

    font-weight:700;

}

/* LABEL */

label{

    color:white !important;

    font-size:16px;

    font-weight:500;

}

/* INPUT */

.stTextArea textarea{

    border-radius:12px;

    border:none;

    padding:14px;

    background:white;

    color:black !important;

}

/* Placeholder */

.stTextArea textarea::placeholder{

    color:#6B7280;

}

/* BUTTON */

.stButton>button{

    background:#10B981;

    color:white;

    border:none;

    border-radius:12px;

    padding:12px 25px;

    font-weight:bold;

}

.stButton>button:hover{

    background:#059669;

}

</style>

""", unsafe_allow_html=True)

ps = PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Spam Detection System")

input_sms = st.text_area(
    "Enter your message",
    height=150
)

def transform_text(text):

    text = text.lower()

    text = nltk.word_tokenize(text)

    y = []

    for i in text:

        if i.isalnum():

            y.append(i)

    text = y[:]

    y.clear()

    for i in text:

        if i not in stopwords.words('english') and i not in string.punctuation:

            y.append(i)

    text = y[:]

    y.clear()

    for i in text:

        y.append(ps.stem(i))

    return " ".join(y)


if st.button("Predict"):

    transformed_sms = transform_text(input_sms)

    vector_input = tfidf.transform([transformed_sms])

    result = model.predict(vector_input)[0]

    if result == 1:

        st.error("Spam Message")

    else:

        st.success("Not Spam")