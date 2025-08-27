import streamlit as st
import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2
import os
import json
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Personal Q&A Chatbot", 
    page_icon="🤖",
    layout="wide"
)

_embeddings_cache = None

def get_embeddings():
    global _embeddings_cache
    if _embeddings_cache is None:
        # Check for GPU availability
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            device = 'cpu'
        
        _embeddings_cache = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        st.session_state.embedding_device = device
    return _embeddings_cache

def load_user_data():
    if os.path.exists("users.json"):
        with open("users.json", "r") as f:
            return json.load(f)
    return {}

def save_user_data(data):
    with open("users.json", "w") as f:
        json.dump(data, f, indent=2)

def save_chat_history(username, pdf_name, messages):
    try:
        user_dir = f"user_data/{username}"
        os.makedirs(user_dir, exist_ok=True)
        chat_file = f"{user_dir}/{pdf_name}_chat.json"
        with open(chat_file, "w") as f:
            json.dump(messages, f, indent=2)
    except Exception as e:
        st.error(f"Error saving chat: {str(e)}")

def load_chat_history(username, pdf_name):
    try:
        chat_file = f"user_data/{username}/{pdf_name}_chat.json"
        if os.path.exists(chat_file):
            with open(chat_file, "r") as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Error loading chat: {str(e)}")
    return []

def save_user_embeddings(username, vectorstore, filename):
    try:
        user_dir = f"user_data/{username}"
        os.makedirs(user_dir, exist_ok=True)
        embeddings_path = f"{user_dir}/{filename}_embeddings"
        vectorstore.save_local(embeddings_path)
        return True
    except Exception:
        return False 

def check_embeddings_exist(username, filename):
    embeddings_path = f"user_data/{username}/{filename}_embeddings"
    # Check if directory exists and contains required FAISS files
    if os.path.exists(embeddings_path):
        required_files = ['index.faiss', 'index.pkl']
        return all(os.path.exists(os.path.join(embeddings_path, f)) for f in required_files)
    return False

def load_user_embeddings(username, filename):
    try:
        embeddings_path = f"user_data/{username}/{filename}_embeddings"
        if check_embeddings_exist(username, filename):
            embeddings = get_embeddings()
            vectorstore = FAISS.load_local(embeddings_path, embeddings, allow_dangerous_deserialization=True)
            
            st.session_state.cached_vectorstore = vectorstore
            st.session_state.cached_pdf_name = filename
            
            return vectorstore
        else:
            return None
    except Exception as e:
        st.error(f"Error loading embeddings: {str(e)}")
        return None

if "user_name" not in st.session_state:
    st.session_state.user_name = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None
if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

users = load_user_data()

if not st.session_state.user_name:
    st.title("🤖 Personal Q&A Chatbot")
    st.subheader("Your intelligent document assistant")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if users:
            st.write("### 👋 Welcome Back!")
            selected_user = st.selectbox("Select your profile:", [""] + list(users.keys()))
            
            if selected_user:
                st.session_state.user_name = selected_user
                st.rerun()
            col_a, col_b, col_c = st.columns([1, 1, 1])
            with col_b:
                if st.button("🆕 New User", key="new_user_btn"):
                    st.session_state.show_new_user = True
        else:
            st.session_state.show_new_user = True
        
        if st.session_state.get('show_new_user', False):
            st.write("**Create New Profile**")
            new_name = st.text_input("Enter your name:", placeholder="e.g., John Doe")
            
            col_x, col_y = st.columns(2)
            with col_x:
                if st.button("🚀 Start Chatting", use_container_width=True) and new_name:
                    st.session_state.user_name = new_name
                    if new_name not in users:
                        users[new_name] = {"pdfs": {}}
                        save_user_data(users)
                    st.session_state.show_new_user = False
                    st.rerun()
            with col_y:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.show_new_user = False
                    st.rerun()

else:
    # User is logged in
    user_data = users.get(st.session_state.user_name, {"pdfs": {}})
    
    # Greeting
    if not st.session_state.messages and not st.session_state.pdf_loaded:
        if user_data["pdfs"]:
            greeting = f"Hey {st.session_state.user_name}! 👋 Welcome back! Would you like to chat about your previous PDFs or upload a new one?"
        else:
            greeting = f"Hi {st.session_state.user_name}! 👋 Welcome! Please upload a PDF file and I'll help answer your questions about it."
        st.session_state.messages.append({"role": "assistant", "content": greeting})


    with st.sidebar:
        st.markdown("### 👤 User Profile")
        device_info = f"🖥️ {st.session_state.get('embedding_device', 'cpu').upper()}" if 'embedding_device' in st.session_state else ""
        st.info(f"**{st.session_state.user_name}**\n\n📊 {len(user_data['pdfs'])} PDFs stored\n{device_info} Embeddings")
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.clear()
            st.rerun()
        
        st.markdown("---")
        
        st.markdown("### 📚 Your Documents")
        
        if user_data["pdfs"]:
            for pdf_name in user_data["pdfs"]:
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        word_count = user_data["pdfs"][pdf_name].get("word_count", "Unknown")
                        
                        # Show active indicator with different styling
                        is_active = st.session_state.current_pdf == pdf_name
                        if is_active:
                            st.markdown(f"**📄 {pdf_name}** ✅")
                            st.success("Currently Active")
                        else:
                            st.markdown(f"**📄 {pdf_name}**")
                        
                        st.caption(f"📝 {word_count} words")
                        
                        embeddings_exist = check_embeddings_exist(st.session_state.user_name, pdf_name)
                        button_text = f"💬 Chat" if embeddings_exist else f"⚠️ Missing"
                        
                        if st.button(button_text, key=f"load_{pdf_name}", use_container_width=True, disabled=is_active):
                            if embeddings_exist:
                                # Check if already cached
                                if (hasattr(st.session_state, 'cached_pdf_name') and 
                                    st.session_state.cached_pdf_name == pdf_name and
                                    hasattr(st.session_state, 'cached_vectorstore')):
                                    # Use cached version - instant load
                                    st.session_state.vectorstore = st.session_state.cached_vectorstore
                                    st.session_state.current_pdf = pdf_name
                                    st.session_state.pdf_loaded = True
                                    st.session_state.messages = load_chat_history(st.session_state.user_name, pdf_name)
                                    st.success(f"✅ Loaded {pdf_name} (cached)")
                                    st.rerun()
                                else:
                                    # Load from disk
                                    with st.spinner(f"Loading {pdf_name} from disk..."):
                                        # Save current chat before switching
                                        if st.session_state.current_pdf and st.session_state.messages:
                                            save_chat_history(st.session_state.user_name, st.session_state.current_pdf, st.session_state.messages)
                                        
                                        vectorstore = load_user_embeddings(st.session_state.user_name, pdf_name)
                                        if vectorstore:
                                            st.session_state.vectorstore = vectorstore
                                            st.session_state.current_pdf = pdf_name
                                            st.session_state.pdf_loaded = True
                                            
                                            # Load chat history for this PDF
                                            st.session_state.messages = load_chat_history(st.session_state.user_name, pdf_name)
                                            
                                            st.success(f"✅ Loaded {pdf_name}")
                                            st.rerun()
                                        else:
                                            st.error(f"❌ Failed to load {pdf_name}")
                            else:
                                st.warning(f"⚠️ Please re-upload {pdf_name}")
                    
                    with col2:
                        if st.button("🗑️", key=f"del_{pdf_name}", help="Delete"):
                            del users[st.session_state.user_name]["pdfs"][pdf_name]
                            save_user_data(users)
                            import shutil
                            try:
                                # Delete embeddings and chat history
                                shutil.rmtree(f"user_data/{st.session_state.user_name}/{pdf_name}_embeddings")
                                chat_file = f"user_data/{st.session_state.user_name}/{pdf_name}_chat.json"
                                if os.path.exists(chat_file):
                                    os.remove(chat_file)
                            except:
                                pass
                            st.success(f"🗑️ Deleted {pdf_name}")
                            st.rerun()
                    
                    st.markdown("---")
        else:
            st.info("📭 No PDFs uploaded yet")
        
        st.markdown("---")
        
        # Upload Section
        st.markdown("### 📁 Upload New Document")
        if 'upload_counter' not in st.session_state:
            st.session_state.upload_counter = 0
        uploader_key = f"uploader_{st.session_state.upload_counter}"
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key=uploader_key)
        
        if uploaded_file:
            pdf_name = uploaded_file.name.replace(".pdf", "")
            
            # Check if this PDF already exists
            if pdf_name in users.get(st.session_state.user_name, {}).get("pdfs", {}) and not st.session_state.get('replace_existing', False):
                st.warning(f"⚠️ {pdf_name} already exists!")
                
                col_replace, col_cancel = st.columns(2)
                with col_replace:
                    if st.button("🔄 Replace/Update", key="replace_pdf", use_container_width=True):
                        # Set flag to proceed with replacement
                        st.session_state.replace_existing = True
                        st.rerun()
                with col_cancel:
                    if st.button("❌ Cancel Upload", key="cancel_upload", use_container_width=True):
                        # Increment counter to clear file uploader
                        st.session_state.upload_counter += 1
                        st.info("Upload cancelled. Select existing PDF from the list above.")
                        st.rerun()
            
            # Process the PDF (either new or replacement)
            if pdf_name not in users.get(st.session_state.user_name, {}).get("pdfs", {}) or st.session_state.get('replace_existing', False):
                api_key = os.getenv("GEMINI_API_KEY")
                
                if api_key:
                    genai.configure(api_key=api_key)
                    
                    with st.spinner(f"Processing {pdf_name}..."):
                        try:
                            # Extract text
                            reader = PyPDF2.PdfReader(uploaded_file)
                            text = ""
                            for page in reader.pages:
                                text += page.extract_text() + "\n"
                            
                            if not text.strip():
                                st.error("❌ No text found in PDF")
                                st.stop()
                            
                            #chunking with RecursiveCharacterTextSplitter
                            text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=1000,
                                chunk_overlap=200,
                                length_function=len,
                                separators=["\n\n", "\n", ". ", " ", ""]
                            )
                            chunks = text_splitter.split_text(text)
                            chunks = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 100]
                            
                            if not chunks:
                                st.error("❌ No meaningful content found")
                                st.stop()
                            
                            # Create embeddings and vectorstore
                            embeddings = get_embeddings()
                            vectorstore = FAISS.from_texts(chunks, embeddings)
                            
                            # Save and activate
                            if save_user_embeddings(st.session_state.user_name, vectorstore, pdf_name):
                                users[st.session_state.user_name]["pdfs"][pdf_name] = {
                                    "word_count": len(text.split()),
                                    "chunks": len(chunks)
                                }
                                save_user_data(users)
                                
                                st.session_state.vectorstore = vectorstore
                                st.session_state.current_pdf = pdf_name
                                st.session_state.pdf_loaded = True
                                st.session_state.messages = []
                                
                                action = "updated" if st.session_state.get('replace_existing', False) else "ready"
                                
                                if 'replace_existing' in st.session_state:
                                    del st.session_state.replace_existing
                                st.session_state.upload_counter += 1
                                
                                st.success(f"✅ {pdf_name} {action}! ({len(chunks)} chunks)")
                                st.rerun()
                            else:
                                st.error("❌ Failed to save")
                                
                        except Exception as e:
                            st.error(f"❌ Error processing PDF: {str(e)}")
                else:
                    st.error("❌ Please set GEMINI_API_KEY in .env file")

    st.title("🤖 Personal Q&A Assistant")
    
    if st.session_state.current_pdf:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.success(f"📄 Active Document: **{st.session_state.current_pdf}**")
        with col2:
            if st.button("🗑️ Clear Chat", key="clear_chat"):
                st.session_state.messages = []
                chat_file = f"user_data/{st.session_state.user_name}/{st.session_state.current_pdf}_chat.json"
                if os.path.exists(chat_file):
                    os.remove(chat_file)
                st.success("Chat cleared!")
                st.rerun()
    else:
        st.info("📋 Select or upload a PDF from the sidebar to start chatting!")
    

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if st.session_state.current_pdf and st.session_state.vectorstore:
        if prompt := st.chat_input(f"Ask about {st.session_state.current_pdf}..."):
            api_key = os.getenv("GEMINI_API_KEY")
            
            if not api_key:
                st.error("❌ Please set GEMINI_API_KEY in .env file")
                st.stop()
            
            # Configure Gemini
            genai.configure(api_key=api_key)
            
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            
            docs = st.session_state.vectorstore.similarity_search(prompt, k=3)
            context = "\n".join([doc.page_content for doc in docs])
            
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            
            conversation_history = ""
            if len(st.session_state.messages) > 1:  
                conversation_history = "\n\nPrevious conversation:\n"
                for msg in st.session_state.messages[-50:]:  
                    if msg["role"] == "user":
                        conversation_history += f"User: {msg['content']}\n"
                    else:
                        conversation_history += f"Assistant: {msg['content']}\n"
            
            system_prompt = f"""You are a helpful Q&A assistant for {st.session_state.user_name}. Answer questions based ONLY on the provided context from their uploaded PDF: {st.session_state.current_pdf}.

RULES:
1. Give short, clear answers (2-3 sentences max)
2. Only answer questions related to the PDF content
3. Be conversational and helpful
4. Remember the conversation history and maintain context
5. Prevent harmful/inappropriate outputs.
6. If question is not about the PDF, say "This question is not related to your uploaded document."

Context from {st.session_state.current_pdf}:
{context}{conversation_history}

Current Question: {prompt}

Answer:"""
            
            try:
                response_obj = model.generate_content(system_prompt)
                response = response_obj.text.strip()
            except Exception as e:
                st.error(f"Error details: {str(e)}")
                response = "Sorry, I couldn't process your question. Please try again."
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.write(response)
            
            # Auto-save chat history after each interaction
            save_chat_history(st.session_state.user_name, st.session_state.current_pdf, st.session_state.messages)
    else:
        st.info("🔒 Chat disabled - Please select a PDF first")
