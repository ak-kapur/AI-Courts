
import google.generativeai as genai
import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai

import webbrowser
GOOGLE_API_KEY = "AIzaSyAmJBHy_GJ3Olj2W9ESWbGaHvat0BpZKF4"
genai.configure(api_key=GOOGLE_API_KEY)

# Load Gemini Model
model = genai.GenerativeModel("gemini-1.5-flash")

# st.sidebar.title("Navigation")

# # Button to open app6(Judge).py
# if st.sidebar.button("See the Suggested Decision"):
#     os.system("streamlit run app6(Judge).py")  # Runs the Judge app

# # Button to open index.html
# index_path = r"C:\Users\aryam\Desktop\AI-court\templates\index.html"  # Get absolute path
# if st.sidebar.button("Dashboard"):
#     webbrowser.open(f"file://{index_path}")  # Open the HTML file in a new tab
# if st.sidebar.button("Scan Documents"):
#     os.system("streamlit run ocr1.py")


# Streamlit App UI
st.set_page_config(page_title="AI Chatbot", layout="centered")
st.title("🤖 AI Chatbot")
st.write("Ask me anything!")


st.sidebar.title("Navigation")

# Button to open app6(Judge).py
if st.sidebar.button("See the Suggested Decision"):
    os.system("streamlit run app6(Judge).py")  # Runs the Judge app

# Button to open index.html
index_path = r"C:\Users\aryam\Desktop\Minor Project\templates\index.html"  # Get absolute path
if st.sidebar.button("Dashboard"):
    webbrowser.open(f"file://{index_path}")  # Open the HTML file in a new tab
if st.sidebar.button("Scan Documents"):
    os.system("streamlit run ocr1.py")

# Initialize chat history if not exists
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
user_input = st.chat_input("Type your message...")
if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get chatbot response
    try:
        response = model.generate_content(user_input)
        bot_reply = response.text
    except Exception as e:
        bot_reply = f"Error: {str(e)}"
    
    # Display bot response
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
