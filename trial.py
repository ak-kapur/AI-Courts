# import streamlit as st
# import requests
# from bs4 import BeautifulSoup
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# import google.generativeai as genai
# import os

# # Set up Google API key for Generative AI
# GOOGLE_API_KEY = "AIzaSyDW6StuDRPI-HkMFK0XaCboNABSpe8VUbY"
# genai.configure(api_key=GOOGLE_API_KEY)

# @st.cache_data
# def search_indian_kanoon(query):
#     search_url = f"https://indiankanoon.org/search/?formInput={query.replace(' ', '+')}"
#     headers = {
#         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
#     }
    
#     try:
#         response = requests.get(search_url, headers=headers)
#         response.raise_for_status()
#         soup = BeautifulSoup(response.content, "html.parser")
#         results = []
#         for result in soup.find_all('div', class_='result_title'):
#             title = result.find('a').get_text()
#             link = "https://indiankanoon.org" + result.find('a')['href']
#             results.append({'title': title, 'link': link})
#         return results
#     except requests.RequestException as e:
#         st.error(f"Error searching Indian Kanoon: {str(e)}")
#         return []

# @st.cache_data
# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         try:
#             pdf_reader = PdfReader(pdf)
#             for page in pdf_reader.pages:
#                 text += page.extract_text()
#         except Exception as e:
#             st.error(f"Error processing PDF {pdf.name}: {str(e)}")
#     return text

# @st.cache_data
# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
#     return text_splitter.split_text(text)

# @st.cache_resource
# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     return vector_store

# @st.cache_resource
# def get_conversational_chain():
#     prompt_template = """
#     You are a legal assistant tasked with extracting specific details from a case document. 
#     Answer the question as accurately as possible based on the provided context. 
#     If the answer is not available, say, "Answer is not available in the context."
    
#     Context: {context}
#     Question: {question}
    
#     Answer:
#     """
    
#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=GOOGLE_API_KEY)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#     return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# def user_input(user_question, vector_store):
#     docs = vector_store.similarity_search(user_question, k=5)
#     chain = get_conversational_chain()
#     response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
#     return response["output_text"]

# def get_case_summary():
#     return "What is this case about in 20 words?"

# def main():
#     st.set_page_config(page_title="Legal Ease", page_icon="⚖", layout="wide")
#     st.title("Legal Ease")
#     st.markdown("### AI Research Assistant: A tool to streamline legal research and enhance productivity.")

#     if 'vector_store' not in st.session_state:
#         st.session_state.vector_store = None

#     pdf_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    
#     if pdf_files:
#         with st.spinner("Processing PDFs..."):
#             raw_text = get_pdf_text(pdf_files)
#             text_chunks = get_text_chunks(raw_text)
#             st.session_state.vector_store = get_vector_store(text_chunks)
#             st.success("PDFs processed successfully! Now ask a question or search for similar cases.")

#     option = st.radio("Would you like to ask a question from your PDFs or search?", ('Ask PDF', 'Search'))

#     question = st.text_input("Ask a question:")

#     if option == 'Ask PDF':
#         if st.button("Get Answer from PDF"):
#             if not st.session_state.vector_store:
#                 st.error("Please upload a PDF first.")
#             elif question:
#                 with st.spinner("Searching your PDFs..."):
#                     answer = user_input(question, st.session_state.vector_store)
#                     st.write(f"*Answer from PDF:* {answer}")
#             else:
#                 st.error("Please ask a question.")
#     elif option == 'Search':
#         if st.button("Get Search Results"):
#             if question:
#                 with st.spinner("Searching..."):
#                     results = search_indian_kanoon(question)
#                     if results:
#                         st.write(f"*Found {len(results)} similar cases:*")
#                         for result in results:
#                             st.write(f"[{result['title']}]({result['link']})")
#                     else:
#                         st.write("No results found.")
#             else:
#                 st.error("Please ask a question.")
    
#     if st.button("Search Similar Cases"):
#         if st.session_state.vector_store:
#             with st.spinner("Generating case summary..."):
#                 summary_question = get_case_summary()
#                 summary_answer = user_input(summary_question, st.session_state.vector_store)
#                 st.write(f"*Generated Case Summary:* {summary_answer}")
                
#                 with st.spinner("Searching..."):
#                     results = search_indian_kanoon(summary_answer)
#                     if results:
#                         st.write(f"*Found {len(results)} similar cases:*")
#                         for result in results:
#                             st.write(f"[{result['title']}]({result['link']})")
#                     else:
#                         st.write("No results found.")
#         else:
#             st.error("Please upload a PDF first.")

# if __name__ == "__main__":
#     main()


import streamlit as st
import requests
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

# Set up Google API key for Generative AI
GOOGLE_API_KEY = "AIzaSyDW6StuDRPI-HkMFK0XaCboNABSpe8VUbY"
genai.configure(api_key=GOOGLE_API_KEY)


@st.cache_data
def search_indian_kanoon(query):
    search_url = f"https://indiankanoon.org/search/?formInput={query.replace(' ', '+')}"
    try:
        webbrowser.open(search_url)  # Open the URL in the browser
        return [{'title': 'Click here to view search results', 'link': search_url}]
    except Exception as e:
        return []

@st.cache_data
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.error(f"Error processing PDF {pdf.name}: {str(e)}")
    return text

@st.cache_data
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return text_splitter.split_text(text)

@st.cache_resource
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

@st.cache_resource
def get_conversational_chain():
    prompt_template = """
    You are a legal assistant trained to extract information from legal documents.
    Answer the question based on the provided context as accurately as possible. If the information is missing or the question doesn't match the document's context, provide a general answer or contextually relevant response instead of saying the information is missing.
""
    
    Context: {context}
    Question: {question}
    
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=GOOGLE_API_KEY)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question, vector_store):
    docs = vector_store.similarity_search(user_question, k=5)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def get_case_summary():
    return "What is this case about in 20 words?"

def main():
    st.set_page_config(page_title="Legal Ease", page_icon="⚖", layout="wide")
    st.title("Legal Ease")
    st.markdown("### AI Research Assistant: A tool to streamline legal research and enhance productivity.")
    st.sidebar.title("Navigation")
    if st.sidebar.button("See the Strong Points for Preparation"):
        os.system("streamlit run app8.py")  # Runs the Judge app
    index_path = r"C:\Users\aryam\Desktop\Minor Project\templates\lawyer.html"
    if st.sidebar.button("Dashboard"):
        webbrowser.open(f"file://{index_path}")  # Open the HTML file in a new tab
    if st.sidebar.button("Scan Documents"):
        os.system("streamlit run ocr1.py")
    if st.sidebar.button("Chatbot"):
        os.system("streamlit run chatbot.py")

# Button to open index.html


    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None

    pdf_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    
    if pdf_files:
        with st.spinner("Processing PDFs..."):
            raw_text = get_pdf_text(pdf_files)
            text_chunks = get_text_chunks(raw_text)
            st.session_state.vector_store = get_vector_store(text_chunks)
            st.success("PDFs processed successfully! Now ask a question or search for similar cases.")

    option = st.radio("Would you like to ask a question from your PDFs or search?", ('Ask PDF', 'Search'))

    question = st.text_input("Ask a question:")

    if option == 'Ask PDF':
        if st.button("Get Answer from PDF"):
            if not st.session_state.vector_store:
                st.error("Please upload a PDF first.")
            elif question:
                with st.spinner("Searching your PDFs..."):
                    answer = user_input(question, st.session_state.vector_store)
                    st.write(f"*Answer from PDF:* {answer}")
            else:
                st.error("Please ask a question.")
    elif option == 'Search':
        if st.button("Get Search Results"):
            if question:
                with st.spinner("Opening search results..."):
                    results = search_indian_kanoon(question)
                    if results:
                        for result in results:
                            st.write(f"[{result['title']}]({result['link']})")
            else:
                st.error("Please enter a search query.")
    
    if st.button("Search Similar Cases"):
        if st.session_state.vector_store:
            with st.spinner("Generating case summary..."):
                summary_question = get_case_summary()
                summary_answer = user_input(summary_question, st.session_state.vector_store)
                st.write(f"*Generated Case Summary:* {summary_answer}")
                
                with st.spinner("Opening search results..."):
                    results = search_indian_kanoon(summary_answer)
                    if results:
                        for result in results:
                            st.write(f"[{result['title']}]({result['link']})")
        else:
            st.error("Please upload a PDF first.")

if __name__ == "__main__":
    main()
