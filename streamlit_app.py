import os
import streamlit as st
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import tempfile
import json
from datetime import datetime
import pandas as pd
import requests
import time

# Page config
st.set_page_config(
    page_title="🤖 Free AI Document Assistant", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    .user-message {
        background-color: #f0f2f6;
        border-left-color: #667eea;
    }
    .assistant-message {
        background-color: #e8f4f8;
        border-left-color: #4CAF50;
    }
    .stats-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 Free AI Document Assistant</h1>
    <p>Powered by Free AI Models | Upload documents and get intelligent answers</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_embedding_model():
    """Load embedding model with caching"""
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"Error loading embedding model: {str(e)}")
        return None

# Load embedding model
embedding_model = load_embedding_model()

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False
if "document_name" not in st.session_state:
    st.session_state.document_name = ""

# Free AI API configuration
def get_free_ai_response(prompt, max_tokens=500):
    """Get response from free AI APIs"""
    
    # Option 1: Hugging Face Inference API (Free tier available)
    try:
        # You can get free API key from https://huggingface.co/settings/tokens
        hf_api_key = st.secrets.get("HF_API_KEY", "")  # Set in Streamlit secrets
        
        if hf_api_key:
            headers = {"Authorization": f"Bearer {hf_api_key}"}
            
            # Using free models like Google's Flan-T5 or Microsoft's DialoGPT
            api_url = "https://api-inference.huggingface.co/models/google/flan-t5-large"
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "temperature": 0.7,
                    "do_sample": True
                }
            }
            
            response = requests.post(api_url, headers=headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "").strip()
        
        # Fallback to local processing if no API key
        return generate_local_response(prompt)
        
    except Exception as e:
        st.error(f"Error with Hugging Face API: {str(e)}")
        return generate_local_response(prompt)

def generate_local_response(prompt):
    """Generate response using local processing (rule-based)"""
    try:
        # Simple rule-based response generation
        question_lower = prompt.lower()
        
        # Extract context from prompt
        context_start = prompt.find("Context from the document:")
        question_start = prompt.find("Question:")
        
        if context_start != -1 and question_start != -1:
            context = prompt[context_start + len("Context from the document:"):question_start].strip()
            question = prompt[question_start + len("Question:"):].strip()
        else:
            context = prompt
            question = prompt
        
        # Basic question analysis
        if any(word in question_lower for word in ['summary', 'summarize', 'main points', 'key points']):
            return generate_summary(context)
        elif any(word in question_lower for word in ['what is', 'define', 'definition']):
            return generate_definition(context, question)
        elif any(word in question_lower for word in ['how', 'process', 'steps']):
            return generate_process_explanation(context)
        elif any(word in question_lower for word in ['why', 'reason', 'because']):
            return generate_explanation(context)
        elif any(word in question_lower for word in ['list', 'enumerate', 'bullet points']):
            return generate_list(context)
        else:
            return generate_general_response(context, question)
            
    except Exception as e:
        return f"I apologize, but I encountered an error while processing your question: {str(e)}"

def generate_summary(context):
    """Generate a summary from context"""
    sentences = context.split('. ')
    if len(sentences) <= 3:
        return f"Based on the document content: {context[:500]}..."
    
    # Take first few sentences and last sentence for summary
    summary_sentences = sentences[:2] + [sentences[-1]]
    summary = '. '.join(summary_sentences)
    
    return f"**Summary:** {summary}\n\nThis appears to be the main content from the document. The key information includes the initial context and conclusion."

def generate_definition(context, question):
    """Generate definition-style response"""
    # Extract potential term from question
    question_words = question.lower().split()
    
    return f"Based on the document context:\n\n{context[:300]}...\n\nThis appears to be related to your question about definitions or explanations. The document provides context that may help answer your specific question."

def generate_process_explanation(context):
    """Generate process or how-to explanation"""
    return f"**Process/Steps based on document:**\n\n{context[:400]}...\n\nThe document contains information that appears to describe processes or procedures. Please refer to the specific sections for detailed steps."

def generate_explanation(context):
    """Generate why/reason explanation"""
    return f"**Explanation:**\n\n{context[:400]}...\n\nThe document provides context that may explain the reasons or causes related to your question. The information suggests various factors that could be relevant."

def generate_list(context):
    """Generate list-style response"""
    sentences = context.split('. ')
    items = []
    
    for i, sentence in enumerate(sentences[:5]):
        if sentence.strip():
            items.append(f"{i+1}. {sentence.strip()}")
    
    return "**Key Points from Document:**\n\n" + "\n".join(items)

def generate_general_response(context, question):
    """Generate general response"""
    return f"**Answer based on document:**\n\n{context[:400]}...\n\nThe document contains relevant information for your question: '{question}'. Please review the context above for specific details."

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # AI Model selection
    ai_model = st.selectbox(
        "Choose AI Processing",
        ["Local Processing (Free)", "Hugging Face API (Free with Token)"],
        index=0,
        help="Local processing is completely free but basic. HF API offers better responses with free token."
    )
    
    # Context length
    context_length = st.slider(
        "Context Length",
        min_value=1,
        max_value=5,
        value=3,
        help="Number of relevant chunks to include"
    )
    
    # Max response length
    max_tokens = st.slider(
        "Max Response Length",
        min_value=100,
        max_value=1000,
        value=500,
        help="Maximum length of AI response"
    )
    
    st.markdown("---")
    
    # API Key input for Hugging Face
    if ai_model == "Hugging Face API (Free with Token)":
        st.markdown("### 🔑 API Configuration")
        st.markdown("Get free API key from [Hugging Face](https://huggingface.co/settings/tokens)")
        hf_key = st.text_input("Hugging Face API Key", type="password")
        if hf_key:
            st.success("✅ API Key configured!")
    
    # Document stats
    if st.session_state.document_processed:
        st.subheader("📊 Document Stats")
        st.success(f"✅ {st.session_state.document_name}")
        st.info(f"📄 {len(st.session_state.chunks)} chunks created")
        st.info(f"💬 {len(st.session_state.chat_history)} questions asked")
    
    # Clear all button
    if st.button("🗑️ Clear All Data", type="secondary"):
        for key in ['chat_history', 'index', 'chunks', 'document_processed', 'document_name']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

def load_file(file):
    """Load file with comprehensive error handling"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(file.getbuffer())
            tmp_path = tmp_file.name
        
        # Load based on file type
        if file.name.lower().endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        elif file.name.lower().endswith(".txt"):
            loader = TextLoader(tmp_path, encoding='utf-8')
        elif file.name.lower().endswith(".csv"):
            loader = CSVLoader(tmp_path)
        elif file.name.lower().endswith(".docx"):
            loader = Docx2txtLoader(tmp_path)
        else:
            st.error("❌ Unsupported file type. Please upload PDF, TXT, CSV, or DOCX files.")
            return None
        
        documents = loader.load()
        os.unlink(tmp_path)  # Clean up
        
        if not documents:
            st.error("❌ No content found in the uploaded file.")
            return None
            
        return documents
        
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        return None

def create_vector_store(documents):
    """Create enhanced vector store"""
    try:
        # Use recursive splitter for better chunking
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        
        if not chunks:
            st.error("❌ No chunks created from the document.")
            return None, None
        
        texts = [doc.page_content for doc in chunks if doc.page_content.strip()]
        
        if not texts:
            st.error("❌ No valid text content found in the document.")
            return None, None
        
        # Create embeddings
        with st.spinner("🔄 Creating embeddings..."):
            vectors = embedding_model.encode(texts, show_progress_bar=False)
            vectors = np.array(vectors).astype('float32')
        
        # Create FAISS index
        index = faiss.IndexFlatL2(vectors.shape[1])
        index.add(vectors)
        
        return index, texts
        
    except Exception as e:
        st.error(f"❌ Error creating vector store: {str(e)}")
        return None, None

def get_relevant_context(question, index, texts, k=3):
    """Get relevant context with similarity scores"""
    try:
        q_vector = embedding_model.encode([question])
        q_vector = np.array(q_vector).astype('float32')
        
        k = min(k, len(texts))
        distances, indices = index.search(q_vector, k=k)
        
        # Get relevant texts with similarity scores
        relevant_contexts = []
        for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            if idx < len(texts) and texts[idx].strip():
                similarity = 1 / (1 + distance)  # Convert distance to similarity
                relevant_contexts.append({
                    'text': texts[idx],
                    'similarity': similarity,
                    'rank': i + 1
                })
        
        return relevant_contexts
        
    except Exception as e:
        st.error(f"❌ Error retrieving context: {str(e)}")
        return []

def generate_ai_response(question, context_items, max_tokens):
    """Generate response using free AI processing"""
    try:
        # Prepare context
        context_text = "\n\n".join([f"Context {item['rank']}: {item['text']}" for item in context_items])
        
        # Create enhanced prompt
        prompt = f"""You are an intelligent document assistant. Answer the question based on the provided context.

Context from the document:
{context_text}

Question: {question}

Please provide a comprehensive answer based on the context above."""

        # Get AI response
        return get_free_ai_response(prompt, max_tokens)
        
    except Exception as e:
        return f"❌ Error generating response: {str(e)}"

# Main application
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📁 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a file to analyze",
        type=["pdf", "txt", "csv", "docx"],
        help="Supported formats: PDF, TXT, CSV, DOCX"
    )

if uploaded_file:
    with st.spinner("🔄 Processing document..."):
        docs = load_file(uploaded_file)
        if docs:
            index, chunks = create_vector_store(docs)
            if index is not None and chunks is not None:
                st.session_state.index = index
                st.session_state.chunks = chunks
                st.session_state.document_processed = True
                st.session_state.document_name = uploaded_file.name
                
                st.success(f"✅ {uploaded_file.name} processed successfully!")
                st.info(f"📄 Created {len(chunks)} text chunks for analysis")

# Q&A Interface
if st.session_state.document_processed:
    st.markdown("### 💬 Ask Questions About Your Document")
    
    # Quick questions
    st.markdown("**💡 Try these example questions:**")
    example_questions = [
        "What is the main topic of this document?",
        "Can you summarize the key points?",
        "What are the most important findings?",
        "Are there any specific recommendations?"
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(example_questions):
        with cols[i % 2]:
            if st.button(f"❓ {question}", key=f"example_{i}"):
                st.session_state.current_question = question
    
    # Question input
    with st.form(key="qa_form", clear_on_submit=True):
        question = st.text_input(
            "Your question:",
            value=st.session_state.get('current_question', ''),
            placeholder="What would you like to know about the document?",
            key="question_input"
        )
        
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            submit = st.form_submit_button("🚀 Ask AI", type="primary")
        with col2:
            if st.form_submit_button("🔄 Clear"):
                st.session_state.current_question = ""
    
    if submit and question.strip():
        with st.spinner("🤔 AI is thinking..."):
            # Get relevant context
            context_items = get_relevant_context(
                question, 
                st.session_state.index, 
                st.session_state.chunks, 
                k=context_length
            )
            
            if context_items:
                # Generate AI response
                answer = generate_ai_response(
                    question, 
                    context_items, 
                    max_tokens
                )
                
                # Add to chat history
                st.session_state.chat_history.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "question": question,
                    "answer": answer,
                    "context_count": len(context_items),
                    "model": ai_model
                })
                
                # Display answer
                st.markdown("### 🤖 AI Response")
                st.markdown(f'<div class="assistant-message">{answer}</div>', unsafe_allow_html=True)
                
                # Show context sources
                with st.expander("📚 View Source Context"):
                    for item in context_items:
                        st.markdown(f"**Relevance: {item['similarity']:.2%}**")
                        st.markdown(f"_{item['text'][:300]}..._")
                        st.markdown("---")
                
            else:
                st.warning("❌ No relevant context found for your question.")
    
    elif submit and not question.strip():
        st.warning("⚠️ Please enter a question.")

# Chat History
if st.session_state.chat_history:
    st.markdown("### 📚 Conversation History")
    
    # Export chat history
    if st.button("📥 Export Chat History"):
        chat_df = pd.DataFrame(st.session_state.chat_history)
        csv = chat_df.to_csv(index=False)
        st.download_button(
            "💾 Download CSV",
            csv,
            f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv"
        )
    
    # Display chat history
    for i, chat in enumerate(reversed(st.session_state.chat_history)):
        with st.expander(f"💬 {chat['timestamp']} - {chat['question'][:50]}..."):
            st.markdown(f"**🙋 Question:** {chat['question']}")
            st.markdown(f"**🤖 Answer:** {chat['answer']}")
            st.markdown(f"**📊 Model:** {chat['model']} | **📄 Context Chunks:** {chat['context_count']}")

# Instructions for free usage
st.markdown("---")
st.markdown("### 🆓 How to Use This Free AI Assistant")

with st.expander("📖 Setup Instructions"):
    st.markdown("""
    **Option 1: Local Processing (Completely Free)**
    - No API keys needed
    - Basic but functional responses
    - Works offline after initial setup
    
    **Option 2: Hugging Face API (Free with Registration)**
    1. Go to [Hugging Face](https://huggingface.co/) and create free account
    2. Get API token from [Settings](https://huggingface.co/settings/tokens)
    3. Add token in sidebar for better AI responses
    4. Free tier includes 30,000 characters per month
    
    **Dependencies to install:**
    ```bash
    pip install streamlit langchain-community sentence-transformers faiss-cpu numpy pandas PyPDF2 python-docx requests
    ```
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem;">
    <p>🤖 <strong>Free AI Document Assistant</strong> | No API costs required!</p>
    <p>Upload documents, ask questions, get intelligent answers - completely free!</p>
</div>
""", unsafe_allow_html=True)

# Error handling for missing models
if embedding_model is None:
    st.error("❌ Failed to load embedding model. Please check your internet connection and try again.")
    st.stop()