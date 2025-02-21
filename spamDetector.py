import pickle
import streamlit as st
from PIL import Image
#from win32com.client import Dispatch
#we can You this package for Audio for Attension of Users
#def speak(text) :
   #speak = Dispatch(("SAPI.SpVoice"))
   #speak.Speak(text)
model = pickle.load(open("spam.pkl", "rb"))
cv = pickle.load(open("vectorizer.pkl", "rb"))

#this command will us to go localhost
def main():
    st.title("Email Spam Detection Using Machine Learning")
    st.subheader("Build With Streamlit & Python")
    msg = st.text_input("Enter a Text :- ")
    if st.button("Predict"):
        data = [msg]
        vect = cv.transform(data).toarray()
        prediction = model.predict(vect)
        result = prediction[0]
        if result == 1 :
            st.error("This is a Spam Mail")
            #speak("This is a Spam Mail")
        else:
            st.success("This is Safe Mail")

main()