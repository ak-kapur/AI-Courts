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

GOOGLE_API_KEY = "AIzaSyDW6StuDRPI-HkMFK0XaCboNABSpe8VUbY"
genai.configure(api_key=GOOGLE_API_KEY)

# Function to open app6(Judge).py   
def open_judge_app():
    if os.name == "nt":  # Windows
        os.system("start cmd /k streamlit run app6(Judge).py")
    else:  # macOS/Linux
        os.system("python3 app6(Judge).py &")


# Sidebar for navigation
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

if st.sidebar.button("Chatbot"):
    os.system("streamlit run chatbot.py")



@st.cache_data
def search_indian_kanoon(query):
    search_url = f"https://indiankanoon.org/search/?formInput={query.replace(' ', '+')}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(search_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        results = []
        for result in soup.find_all('div', class_='result_title'):
            title = result.find('a').get_text()
            link = "https://indiankanoon.org" + result.find('a')['href']
            results.append({'title': title, 'link': link})
        return results
    except requests.RequestException as e:
        st.error(f"Error searching Indian Kanoon: {str(e)}")
        return []



def search_indian_kanoon(query):
    search_url = f"https://indiankanoon.org/search/?formInput={query.replace(' ', '+')}"
    response = requests.get(search_url)
    
    print(f"Searching: {search_url}")  # Debugging
    
    soup = BeautifulSoup(response.content, "html.parser")

    results = []
    for result in soup.find_all('div', class_='result_title'):
        title = result.find('a').get_text()
        link = "https://indiankanoon.org" + result.find('a')['href']
        results.append({'title': title, 'link': link})
    
    print("Results found:", results)  # Debugging
    
    return results
# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into smaller chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create FAISS vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Function to set up Langchain QA chain
def get_conversational_chain():
    prompt_template = """
  You are a legal assistant trained to extract information from legal documents.
    Answer the question based on the provided context as accurately as possible. If the information is missing or the question doesn't match the document's context, provide a general answer or contextually relevant response instead of saying the information is missing.
"
    
    Context: {context}
    Question: {question}
    
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=GOOGLE_API_KEY)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    return chain

# Function to process user input and query the PDFs
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    
    # Load the FAISS index
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question, k=5)  # Fetch more relevant documents

    chain = get_conversational_chain()

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    return response["output_text"]

# Improve the summarization logic
def get_case_summary():
    prompt_template = "Summarize the case in 20-30 words with key details about the issue and involved parties."
    # Use the conversational chain to generate a summary
    response = user_input(prompt_template)
    return response
    
# def get_case_summary():
#     return "What is this case about in 20 words?"
@st.cache_resource
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store
# Streamlit App
def main():
    st.title("Legal Ease for Judges")

    # Step 1: Upload PDFs
    pdf_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    
    if pdf_files:
        with st.spinner("Processing PDFs..."):
            raw_text = get_pdf_text(pdf_files)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("PDFs processed successfully! Now ask a question or search for similar cases on Indian Kanoon.")
    
    # Step 2: Choose to ask from PDFs or search Indian Kanoon
    option = st.radio(
        "Would you like to ask a question from your PDFs or search Indian Kanoon?",
        ('Ask PDF', )
    )

    question = st.text_input("Ask a question:")

    if option == 'Ask PDF':
        if st.button("Get Answer from PDF"):
            if not pdf_files:
                st.error("Please upload a PDF first.")
            elif question:
                with st.spinner("Searching your PDFs..."):
                    answer = user_input(question)
                    st.write(f"**Answer from PDF:** {answer}")
            else:
                st.error("Please ask a question.")
    elif option == 'Search Database':
        if st.button("Get Search Results"):
            if question:
                with st.spinner("Searching Database"):
                    results = search_indian_kanoon(question)
                    if results:
                        st.write(f"**Found {len(results)} similar cases:**")
                        for result in results:
                            st.write(f"[{result['title']}]({result['link']})")
                    else:
                        st.write("No results found.")
            else:
                st.error("Please ask a question.")



    if st.button("Get Summary"):
        if pdf_files:
            with st.spinner("Generating summary..."):
                summary = get_case_summary()
                st.write(f"**Case Summary:** {summary}")
        else:
            st.error("Please upload a PDF first.")

    
    # if st.button("Search Similar Cases"):
    #     if st.session_state.vector_store:
    #         with st.spinner("Generating case summary..."):
    #             summary_question = get_case_summary()
    #             summary_answer = user_input(summary_question, st.session_state.vector_store)
    #             st.write(f"*Generated Case Summary:* {summary_answer}")
                
    #             with st.spinner("Searching..."):
    #                 results = search_indian_kanoon(summary_answer)
    #                 if results:
    #                     st.write(f"*Found {len(results)} similar cases:*")
    #                     for result in results:
    #                         st.write(f"[{result['title']}]({result['link']})")
    #                 else:
    #                     st.write("No results found.")
    #     else:
    #         st.error("Please upload a PDF first.")

    # # New Feature: Search Indian Kanoon with the generated case summary
    # if st.button("Search Similar Cases on Database"):
    #     if pdf_files:
    #         with st.spinner("Generating case summary..."):
    #             summary_question = get_case_summary()
    #             summary_answer = user_input(summary_question)
    #             st.write(f"**Generated Case Summary:** {summary_answer}")
                
    #             with st.spinner("Searching Indian Kanoon..."):
    #                 results = search_indian_kanoon(summary_answer)
    #                 if results:
    #                     st.write(f"**Found {len(results)} similar cases:**")
    #                     for result in results:
    #                         st.write(f"[{result['title']}]({result['link']})")
    #                 else:
    #                     st.write("No results found.")
    #     else:
    #         st.error("Please upload a PDF first.")

if __name__ == "__main__":
    main()
