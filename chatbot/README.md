# PDF Question Answering Chatbot

## Purpose

`chatbot_simple.py` is an example of the simple chatbot built with LangChain and Streamlit. It allows users to ask questions about PDF documents and receive AI-powered answers based on the content of those documents. 

This is an educational project for the Intro to AI course that demonstrates:
- Document loading and processing
- Text chunking and tokenization
- Vector embeddings and similarity search
- LLM-powered question answering
- Streamlit web interface development
- Mock testing without API costs

## Features

✨ **Key Features:**
- **PDF Loading**: Loads and processes PDF documents using LangChain
- **Document Chunking**: Splits large documents into manageable chunks (200 tokens each)
- **Vector Search**: Uses FAISS for fast similarity-based document retrieval
- **RAG Pipeline**: Combines retrieved context with an LLM to generate accurate answers
- **Streamlit UI**: Interactive web interface for asking questions
- **Mock Testing**: Includes mock embeddings and LLM for cost-free testing
- **Production Ready**: Easily switch to real OpenAI APIs with one comment change

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install langchain langchain-community langchain-openai streamlit faiss-cpu pysqlite3-binary numpy PyPDF2
```

2. **Place your PDF file** in the project directory as `example.pdf`

## Usage

### Running the Chatbot

Start the Streamlit app:
```bash
streamlit run chatbot_simple.py
```

The app will open in your browser at `http://localhost:8501`

## Configuration

### Using Mock Mode (Default - Free Testing)
The chatbot uses mock embeddings and LLM by default:
```python
embedding = MockEmbeddings()  # Mock embeddings
chat = MockChatOpenAI()       # Mock LLM
```

### Switching to Real OpenAI APIs

1. **Set your OpenAI API key:**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

2. **Uncomment these lines** in `chatbot_simple.py`:
```python
# Uncomment these:
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
chat = ChatOpenAI(model_name="gpt-4", temperature=0.2)

# Comment these:
# embedding = MockEmbeddings()
# chat = MockChatOpenAI()
```

3. **Add your API key** from https://platform.openai.com/api-keys