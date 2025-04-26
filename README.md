# Document Chat with Gemini and FAISS

This project implements a document-based question-answering system using Google's Gemini AI and FAISS for efficient similarity search. It allows users to upload documents (txt, pdf, docx) and ask questions about their content.

## Features

- Document upload support for multiple formats (txt, pdf, docx)
- Text extraction and chunking
- Semantic search using FAISS
- Question answering powered by Google's Gemini AI
- Interactive chat interface with Streamlit
- Chat history management
- Document context-aware responses

## Prerequisites

- Python 3.8 or higher
- Google API key for Gemini AI
- Required Python packages (see Installation)

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

## Usage

1. Start the application:
```bash
streamlit run main.py
```

2. Upload a document through the web interface
3. Ask questions about the document content
4. Use the "New Chat" button to start a new conversation
5. Use the "Clear Chat" button to clear the chat history

## Project Structure

- `main.py`: Main application file containing the Streamlit interface and core functionality
- `.env`: Configuration file for API keys and environment variables
- `requirements.txt`: List of Python package dependencies

## Dependencies

- streamlit: Web application framework
- langchain-google-genai: Google Gemini AI integration
- faiss: Efficient similarity search
- PyPDF2: PDF text extraction
- python-docx: Word document processing
- python-dotenv: Environment variable management
- numpy: Numerical computations

## How It Works

1. Document Processing:
   - Uploaded documents are processed to extract text
   - Text is split into manageable chunks
   - Each chunk is embedded using Gemini's embedding model

2. Question Answering:
   - User questions are embedded using the same model
   - FAISS finds the most relevant document chunks
   - Gemini AI generates answers based on the retrieved context

## License

This project is open-source and available under the MIT License.

Made with ❤️ by Rahul Jignesh Ganatra

Check out our demo video: [RAG-MODEL Demo](https://drive.google.com/file/d/1Bv2-ueE--G32h9UDbozPDFAlLbaeYVYW/view?usp=sharing)