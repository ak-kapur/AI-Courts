import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
import google.generativeai as genai
import os
import webbrowser
import streamlit as st

import webbrowser
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import os
# Configure the Google API key for Gemini
GOOGLE_API_KEY = "AIzaSyDW6StuDRPI-HkMFK0XaCboNABSpe8VUbY"
genai.configure(api_key=GOOGLE_API_KEY)

def get_focused_points(case_details):
    """
    Use the Google Gemini model to suggest key points a lawyer should focus on based on the case details.
    """
    focus_prompt_template = """
    Based on the following case details and Indian Law Acts and Sections suggest the key points a lawyer should focus on for the case.

    Case Details: {case_details}

    Key Points to Focus:
    """
    
    # Create the prompt with the provided case details
    prompt = PromptTemplate(template=focus_prompt_template, input_variables=["case_details"])
    formatted_prompt = prompt.format(case_details=case_details)
    
    # Initialize the ChatGoogleGenerativeAI with the specified Gemini model
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=GOOGLE_API_KEY)
    
    # Create a HumanMessage with the formatted prompt
    input_message = HumanMessage(content=formatted_prompt)
    
    # Generate the focused points
    focus_points = model([input_message])
    
    return focus_points.content

# Streamlit App
def main():
    st.title("Lawyer's Focus Point Generator")
    st.write("This tool suggests key points a lawyer should focus on based on the case details under the Indian Law .")
    st.sidebar.title("Navigation")
    if st.sidebar.button("AI Research Engine"):
        os.system("streamlit run trial.py")  # Runs the Judge app
    index_path = r"C:\Users\aryam\Desktop\Minor Project\templates\lawyer.html"
    if st.sidebar.button("Dashboard"):
        webbrowser.open(f"file://{index_path}")  # Open the HTML file in a new tab
    if st.sidebar.button("Scan Documents"):
        os.system("streamlit run ocr1.py")
    if st.sidebar.button("Chatbot"):
        os.system("streamlit run chatbot.py")

    # Text area for the user to input case details
    case_details = st.text_area("Enter Case Details:", height=300)
    
    # Generate focused points on button click
    if st.button("Get Focus Points"):
        if case_details.strip():
            with st.spinner("Generating focus points..."):
                focus_points = get_focused_points(case_details)
                st.write(f"**Key Points to Focus:** {focus_points}")
        else:
            st.error("Please enter the case details.")

if __name__ == "__main__":
    main()
