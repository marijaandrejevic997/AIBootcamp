import sys
sys.modules['sqlite3'] = __import__('pysqlite3')
from langchain_community.document_loaders import PyPDFLoader; 
from langchain_text_splitters import TokenTextSplitter; 
from langchain_core.messages import SystemMessage;
from langchain_core.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough
import streamlit as st

class MockEmbeddings(Embeddings):
    """Mock embeddings for testing - returns random vectors instead of calling OpenAI"""
    
    def __init__(self, embedding_dim: int = 1536):
        self.embedding_dim = embedding_dim
    
    def embed_documents(self, texts):
        """Mock embedding for multiple documents"""
        # Returns a list of random vectors
        return [np.random.randn(self.embedding_dim).tolist() for _ in texts]
    
    def embed_query(self, text):
        """Mock embedding for a single query"""
        # Returns a single random vector
        return np.random.randn(self.embedding_dim).tolist()



class MockChatOpenAI(Runnable):
    """Mock ChatOpenAI for testing - returns fake responses instead of calling OpenAI"""
    
    def __init__(self, model_name="gpt-3.5-turbo", temperature=0.2, api_key=None):
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key
    
    def invoke(self, input, config=None):
        """Mock invoke method - returns a fake response"""
        return "This is a mock response from the chatbot. The main topic of the PDF is information retrieval and language models."
    
    @property
    def InputType(self):
        return str
    
    @property
    def OutputType(self):
        return str

# load the PDF
loader_pdf = PyPDFLoader("example.pdf")
# split the PDF into chunks
docs_list = loader_pdf.load()
# split the chunks into smaller chunks up to 200 tokens
token_splitter = TokenTextSplitter(encoding_name="cl100k_base", chunk_size=200, chunk_overlap=40)
docs_list_tokens_split = token_splitter.split_documents(docs_list)

# create the vectorstore from the chunks (can be used OpenAIEmbeddings or MockEmbeddings for testing)
#embeeding = OpenAIEmbeddings(model="text-embedding-3-small", api_key="{{OPENAI_API_KEY}}")
embeeding = MockEmbeddings()
vectorstore = FAISS.from_documents(docs_list_tokens_split, embeeding)
retriever = vectorstore.as_retriever(search_type = "mmr", search_kwargs={"k": 1, "lambda_mult": 0.5})

# prompts
PROMPT_RETRIEVING_S = '''You will receive a question from a student taking the Intro to AI course. Answer the question using only the provided context. '''
PROMPT_TEMPLATE_RETRIEVING_H = '''This is the question: {question} This is the context: {context}'''
prompt_retrieving_s = SystemMessage(content=PROMPT_RETRIEVING_S)
prompt_template_retrieving_h = HumanMessagePromptTemplate.from_template(PROMPT_TEMPLATE_RETRIEVING_H)
chat_prompt_template_retrieving = ChatPromptTemplate.from_messages([prompt_retrieving_s, prompt_template_retrieving_h])

# create the chain and get the response (Using MockChatOpenAI for testing, can be replaced with ChatOpenAI for real API calls)
#chat = ChatOpenAI(model_name="gpt_40", temperature=0.2, api_key="{{OPENAI_API_KEY}}")
chat = MockChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2, api_key="mock-key")
str_output_parser = StrOutputParser()
chain = ({'context': retriever, 
'question': RunnablePassthrough ()} | chat_prompt_template_retrieving | chat | str_output_parser)


st.header("PDF Question Answering", divider = True)
question = st.text_input("\nAsk a question about the PDF:\n")
if (st.button("Ask")):
    if question:
        response_placeholder = st.empty()
        response_txt = ""
        result = chain.stream(question)

        for chunk in result: 
            response_txt += chunk
            response_placeholder.markdown(response_txt)
    else: 
        st.warning("Please enter a question before clicking the Ask button.")