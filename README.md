# 🤖 Free AI Document Assistant

A powerful, completely free AI-powered document assistant that allows you to upload documents and ask intelligent questions about their content. Built with Streamlit and powered by free AI models.

## ✨ Features

- **📄 Multi-format Support**: PDF, TXT, CSV, and DOCX files
- **🤖 Free AI Processing**: No API costs required for basic functionality
- **🔍 Intelligent Search**: Vector-based document search using FAISS
- **💬 Interactive Chat**: Natural language Q&A interface
- **📊 Document Analytics**: Chunk analysis and similarity scoring
- **🎨 Modern UI**: Clean, responsive interface with custom styling
- **💾 Export Options**: Download chat history as CSV
- **🔄 Real-time Processing**: Instant document analysis and responses

## 🚀 Live Demo

**Deploy your own instance:**
- [Streamlit Cloud](https://share.streamlit.io) (Recommended)
- [Hugging Face Spaces](https://huggingface.co/spaces)
- [Render](https://render.com)

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Quick Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/ai-document-assistant.git
cd ai-document-assistant
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
streamlit run streamlit_app.py
```

4. **Open in browser:**
Navigate to `http://localhost:8501`

## 📦 Dependencies

```python
streamlit==1.31.0
langchain-community==0.0.20
sentence-transformers==2.2.2
faiss-cpu==1.7.4
numpy==1.24.3
pandas==2.0.3
PyPDF2==3.0.1
python-docx==0.8.11
requests==2.31.0
```

## 🎯 Usage

### Basic Usage (100% Free)

1. **Upload Document**: Drag and drop or select a file (PDF, TXT, CSV, DOCX)
2. **Wait for Processing**: The app will automatically chunk and index your document
3. **Ask Questions**: Type natural language questions about your document
4. **Get Answers**: Receive AI-generated responses based on document content

### Advanced Usage (Free with API Token)

1. **Get Hugging Face API Token**:
   - Visit [Hugging Face](https://huggingface.co/settings/tokens)
   - Create a free account
   - Generate an API token (free tier: 30,000 characters/month)

2. **Configure API**:
   - Select "Hugging Face API" in the sidebar
   - Enter your API token
   - Enjoy enhanced AI responses

## 🔧 Configuration

### Streamlit Configuration

Create `.streamlit/config.toml`:
```toml
[global]
fileWatcherType = "auto"
runOnSave = true

[server]
headless = true
port = 8501
maxUploadSize = 200

[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

### Environment Variables

For deployment, set these secrets:
```bash
HF_API_KEY=your_hugging_face_api_key_here
```

## 🚀 Deployment Guide

### Streamlit Cloud (Recommended)

1. **Push to GitHub**:
```bash
git add .
git commit -m "Ready for deployment"
git push origin main
```

2. **Deploy**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file: `streamlit_app.py`
   - Click "Deploy"

3. **Configure Secrets** (Optional):
   - In app settings, add: `HF_API_KEY = "your_key"`

### Hugging Face Spaces

1. **Create Space**:
   - Go to [Hugging Face Spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Choose "Streamlit" as SDK

2. **Upload Files**:
   - Upload `streamlit_app.py` as `app.py`
   - Upload `requirements.txt`

### Other Platforms

- **Render**: Connect GitHub repo, set build command
- **Railway**: One-click deployment from GitHub
- **Heroku**: Use `setup.sh` and `Procfile`

## 🎨 Features Overview

### Document Processing
- **Smart Chunking**: Recursive text splitting for optimal context
- **Vector Embeddings**: Using SentenceTransformers for semantic search
- **FAISS Indexing**: Fast similarity search and retrieval
- **Multi-format Support**: PDF, TXT, CSV, DOCX handling

### AI Integration
- **Local Processing**: Rule-based responses (completely free)
- **Hugging Face API**: Advanced AI models (free tier available)
- **Fallback System**: Automatic fallback to local processing
- **Context-aware**: Responses based on relevant document chunks

### User Interface
- **Responsive Design**: Works on desktop and mobile
- **Interactive Chat**: Real-time Q&A interface
- **Progress Indicators**: Loading states and progress bars
- **Error Handling**: Graceful error messages and recovery

## 🔍 How It Works

1. **Document Upload**: Files are processed and temporarily stored
2. **Text Extraction**: Content extracted based on file type
3. **Chunking**: Documents split into manageable, overlapping chunks
4. **Embedding**: Text converted to vector representations
5. **Indexing**: FAISS creates searchable vector index
6. **Query Processing**: User questions converted to vectors
7. **Similarity Search**: Most relevant chunks identified
8. **AI Response**: Context-aware answer generation

## 🛡️ Privacy & Security

- **No Data Storage**: Documents processed in memory only
- **Temporary Files**: Uploaded files automatically deleted
- **Local Processing**: Basic mode works completely offline
- **API Security**: Optional API keys stored securely
- **Open Source**: Full code transparency

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -m "Add new feature"`
4. Push to branch: `git push origin feature/new-feature`
5. Submit a pull request

## 📋 Roadmap

- [ ] **Multi-language Support**: Support for non-English documents
- [ ] **Advanced AI Models**: Integration with more free AI APIs
- [ ] **Batch Processing**: Multiple document analysis
- [ ] **Export Formats**: PDF and Word export options
- [ ] **Collaboration**: Share documents and conversations
- [ ] **Analytics**: Usage statistics and insights

## 🐛 Troubleshooting

### Common Issues

**"Error loading embedding model"**
- Check internet connection
- Restart the application
- Clear browser cache

**"No chunks created from document"**
- Ensure document contains readable text
- Try a different file format
- Check file size (max 200MB)

**"API Error with Hugging Face"**
- Verify API token is correct
- Check API quota limits
- Try local processing mode

### Performance Tips

- **Large Documents**: Use smaller context length for faster processing
- **Memory Issues**: Process documents in smaller chunks
- **API Limits**: Use local processing for extensive usage

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Streamlit**: For the amazing web framework
- **Hugging Face**: For free AI model access
- **LangChain**: For document processing utilities
- **FAISS**: For efficient similarity search
- **SentenceTransformers**: For text embeddings

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/ai-document-assistant/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/ai-document-assistant/discussions)
- **Email**: your.email@example.com

## 🌟 Show Your Support

If you find this project helpful, please consider:
- ⭐ Starring the repository
- 🍴 Forking and contributing
- 📢 Sharing with others
- 🐛 Reporting issues

---

**Made with ❤️ and powered by free AI models**

*Deploy your own instance today and start analyzing documents with AI - completely free!*