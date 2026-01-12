# PDF Q&A Assistant

A Streamlit application that enables interactive question-answering over PDF documents using advanced language models and vector embeddings.

## Features

- 📄 **PDF Upload**: Upload PDF files and process them for Q&A
- 🤖 **AI-Powered Responses**: Uses Google's Gemini 2.5 Flash Lite model for intelligent answers
- 🧬 **Vector Embeddings**: Leverages HuggingFace embeddings for semantic search
- 💾 **Vector Store**: Chroma vector database for efficient document retrieval
- 💬 **Conversation History**: Maintains chat history within the session

## Requirements

- Python 3.8+
- Streamlit
- LangChain ecosystem libraries
- Google API key for Gemini access

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Project_III
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/Scripts/activate  # On Windows
# or
source venv/bin/activate      # On macOS/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root and add your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

The app will open in your browser. Use the sidebar to:
1. Upload a PDF file
2. Ask questions about the document's content
3. View the AI-generated responses

## Technologies Used

- **Streamlit**: Web application framework
- **LangChain**: LLM orchestration and document processing
- **Google Gemini API**: Large language model
- **HuggingFace**: Sentence transformers for embeddings
- **Chroma**: Vector database for semantic search
- **PyPDF**: PDF document loading and processing

## Project Structure

```
Project_III/
├── app.py              # Main Streamlit application
├── requirements.txt    # Project dependencies
├── README.md          # This file
└── .env               # Environment variables (not committed)
```

## Notes

- Ensure your `.env` file is added to `.gitignore` to avoid committing sensitive API keys
- The application uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings
- Gemini model temperature is set to 0.7 for balanced responses

