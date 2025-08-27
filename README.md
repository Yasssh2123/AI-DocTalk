# Q&A Chatbot with File Upload

A powerful multi-user chatbot that allows users to upload PDF files and ask questions about their content using Gemini AI and HuggingFace embeddings with optional GPU acceleration.

## ✨ Features

- **Multi-User Support**: Create profiles and manage separate document collections
- **PDF Processing**: Upload and extract text from PDF documents
- **RAG-based Q&A**: Intelligent question answering using Retrieval-Augmented Generation
- **Duplicate Handling**: Smart duplicate detection with replace/update options
- **Conversation Memory**: Maintains chat history across sessions
- **GPU Acceleration**: Automatic GPU detection for faster embeddings (optional)
- **Document Management**: View, switch between, and delete uploaded PDFs
- **Persistent Storage**: All data saved locally with automatic loading

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up Gemini API Key
Create a `.env` file in the project directory:
```env
GEMINI_API_KEY=your_actual_api_key_here
```

### 3. Run the Application
```bash
streamlit run app.py
```

### 4. Start Using
1. Create a user profile or select existing one
2. Upload a PDF file using the sidebar
3. Ask questions about the uploaded content
4. Switch between different PDFs anytime

## 🔧 Optional GPU Setup

For faster embedding processing on GPU-enabled systems:

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

The app automatically detects and uses GPU when available, showing "🖥️ GPU Embeddings" in the user profile.

## 📁 Project Structure

```
Digiqt_Assignment/
├── app.py              # Main application
├── requirements.txt    # Dependencies
├── README.md          # This file
├── .env               # API keys (create this)
├── users.json         # User data (auto-created)
└── user_data/         # User documents & embeddings (auto-created)
    └── {username}/
        ├── {pdf_name}_embeddings/  # FAISS embeddings
        └── {pdf_name}_chat.json    # Chat history
```

## 🎯 Usage Guide

### Creating Users
- First time: Enter your name to create a profile
- Returning: Select your profile from the dropdown

### Managing PDFs
- **Upload**: Use sidebar file uploader
- **Duplicate**: Choose "Replace/Update" or "Cancel"
- **Switch**: Click any PDF from "Your Documents" list
- **Delete**: Use 🗑️ button next to each PDF

### Chatting
- Ask questions about the active PDF content
- View conversation history
- Clear chat history with "Clear Chat" button
- Questions unrelated to PDF content are filtered out

## 🛠️ Technical Details

- **Frontend**: Streamlit web interface
- **AI Model**: Google Gemini 1.5 Flash
- **Embeddings**: HuggingFace sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: FAISS for similarity search
- **Text Processing**: RecursiveCharacterTextSplitter for optimal chunking
- **Storage**: Local JSON and FAISS files

## 🔒 Data Privacy

- All data stored locally on your machine
- No data sent to external services except Gemini API for responses
- User documents and chat history remain private

## 🐛 Troubleshooting

**"No text found in PDF"**: PDF might be image-based or corrupted
**"Please set GEMINI_API_KEY"**: Check your .env file
**"Failed to load embeddings"**: Try re-uploading the PDF
**Slow processing**: Consider GPU setup for faster embeddings

## 📋 Requirements

- Python 3.8+
- Gemini API key
- 2GB+ RAM for embeddings
- Optional: CUDA-compatible GPU for acceleration

## 🤝 Contributing

Feel free to submit issues and enhancement requests!