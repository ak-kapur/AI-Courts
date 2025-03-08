import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
import google.generativeai as genai
import os
import webbrowser
# Configure the Google API key for Gemini
GOOGLE_API_KEY = "AIzaSyDW6StuDRPI-HkMFK0XaCboNABSpe8VUbY"
genai.configure(api_key=GOOGLE_API_KEY)

def get_suggested_decision(case_details):
    """
    Use the Google Gemini model to suggest a decision based on the case details.
    """
    decision_prompt_template = """
    Based on the following case details and the Indian Law Acts and Section, suggest an appropriate decision.

    Case Details: {case_details}

    Suggested Decision:
    """
    
    # Create the prompt with the provided case details
    prompt = PromptTemplate(template=decision_prompt_template, input_variables=["case_details"])
    formatted_prompt = prompt.format(case_details=case_details)
    
    # Initialize the ChatGoogleGenerativeAI with the specified Gemini model
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=GOOGLE_API_KEY)
    
    # Create a HumanMessage with the formatted prompt
    input_message = HumanMessage(content=formatted_prompt) 
    
    # Generate the suggested decision
    decision = model([input_message])
    
    return decision.content
st.sidebar.title("Navigation")

# Button to open app6(Judge).py
if st.sidebar.button("Chatbot"):
    os.system("streamlit run chatbot.py")  # Runs the Judge app

# Button to open index.html
index_path = r"C:\Users\aryam\Desktop\Minor Project\templates\index.html"  # Get absolute path
if st.sidebar.button("Dashboard"):
    webbrowser.open(f"file://{index_path}")  # Open the HTML file in a new tab
if st.sidebar.button("Scan Documents"):
    os.system("streamlit run ocr1.py")
# Streamlit App
def main():
    st.title("Suggested Decision ")
    st.write("This tool generates a suggested decision for cases under the Indian Legal Acts, based on the details provided.")

    # Text area for the user to input case details
    case_details = st.text_area("Enter Case Details:", height=300)
    
    # Generate suggested decision on button click
    if st.button("Get Suggested Decision"):
        if case_details.strip():
            with st.spinner("Generating suggested decision..."):
                decision = get_suggested_decision(case_details)
                st.write(f"**Suggested Decision:** {decision}")
        else:
            st.error("Please enter the case details.")

if __name__ == "__main__":
    main()
