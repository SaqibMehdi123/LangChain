import streamlit as st
import tempfile
import os
import json
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

# Import all your RAG and data handling functions from the logic file
# Make sure 'rag_core.py' is in the same folder as this file.
from rag_core import (
    get_or_create_vector_store,
    get_rag_response,
    add_documents_to_store,
    get_processed_files_from_store,
    load_chat_history,
    save_chat_history,
    create_default_chat,
    generate_chat_title
)

# ---
# 1. PAGE CONFIGURATION
# Must be the first Streamlit command
# ---
try:
    st.set_page_config(
        page_title="Academic RAG Assistant",
        page_icon="🎓",
        layout="wide",
    )
except st.errors.StreamlitAPIException:
    pass

# ---
# 2. AUTHENTICATION SETUP
# ---
# Load the config.yaml file
try:
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)
except FileNotFoundError:
    st.error("FATAL: 'config.yaml' not found. Run 'generate_hashes.py' first.")
    st.stop()
except Exception as e:
    st.error(f"Error loading config.yaml: {e}")
    st.stop()

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# ---
# 3. MAIN APP LOGIC
# ---

# We initialize session state for authentication
if "authentication_status" not in st.session_state:
    st.session_state.authentication_status = None

# ---
# --- THE FIX IS HERE ---
# We check the login status *first*.
# We only render the login/register tabs if the user is NOT logged in.
# ---

# --- A. IF LOGGED IN (Show the App) ---
if st.session_state["authentication_status"]:
    
    # Get user details from session state
    username = st.session_state["username"]
    name = st.session_state["name"]

    st.title("🎓 Academic RAG Assistant")
    st.markdown("Powered by **Gemini 2.5 Pro** & Your Custom Documents")
    st.markdown("---")

    # --- Session State Initialization ---
    if "api_key" not in st.session_state:
        try:
            st.session_state.api_key = st.secrets["GOOGLE_API_KEY"]
        except Exception:
            st.session_state.api_key = None
            st.error("API Key not found in secrets.toml. App cannot function.")
            st.stop()
            
    if "api_key_loaded_toast_shown" not in st.session_state:
        st.session_state.api_key_loaded_toast_shown = False

    # --- Load USER-SPECIFIC Vector Store ---
    if "vector_store" not in st.session_state:
        with st.spinner("Loading your persistent vector store..."):
            try:
                st.session_state.vector_store = get_or_create_vector_store(
                    st.session_state.api_key, 
                    collection_name=username
                )
                st.session_state.processed_files = get_processed_files_from_store(
                    st.session_state.vector_store
                )
                print(f"Loaded {len(st.session_state.processed_files)} files for user {username}.")
            except Exception as e:
                st.error(f"Failed to load vector store: {e}")
                st.session_state.vector_store = None
                st.session_state.processed_files = []
    
    # --- Load USER-SPECIFIC Chat History ---
    if "chat_history" not in st.session_state:
        with st.spinner("Loading your chat history..."):
            chat_data = load_chat_history(username)
            st.session_state.chat_history = {k: v for k, v in chat_data.items() if k != "chat_id_counter"}
            st.session_state.chat_id_counter = chat_data.get("chat_id_counter", 1)
            st.session_state.active_chat_id = list(st.session_state.chat_history.keys())[0]

    # --- Sidebar ---
    with st.sidebar:
        st.markdown(f"# 🎓 Welcome, {name}")
        authenticator.logout('Logout', 'sidebar') # Add logout button
        st.markdown("---")

        if st.session_state.api_key:
            if not st.session_state.api_key_loaded_toast_shown:
                st.toast("API Key loaded successfully!", icon="✅")
                st.session_state.api_key_loaded_toast_shown = True
        else:
            st.error("API Key not found.")
            st.stop()

        st.markdown("### 1. My Chats")
        
        if st.button("➕ New Chat", use_container_width=True):
            new_chat_id = f"chat_{st.session_state.chat_id_counter}"
            new_chat_title = "New Chat" 
            st.session_state.chat_id_counter += 1
            
            st.session_state.chat_history[new_chat_id] = {
                "title": new_chat_title,
                "messages": [{"role": "assistant", "content": "New chat started. Ask me anything."}]
            }
            st.session_state.active_chat_id = new_chat_id
            
            save_data = st.session_state.chat_history.copy()
            save_data["chat_id_counter"] = st.session_state.chat_id_counter
            save_chat_history(save_data, username)
            st.rerun()

        st.markdown("---")

        chat_container = st.container(height=250)
        with chat_container:
            for chat_id, chat_data in list(st.session_state.chat_history.items()):
                col1, col2 = st.columns([0.85, 0.15])
                with col1:
                    btn_type = "primary" if chat_id == st.session_state.active_chat_id else "secondary"
                    if st.button(chat_data["title"], key=f"select_{chat_id}", use_container_width=True, type=btn_type):
                        st.session_state.active_chat_id = chat_id
                        st.rerun() 
                with col2:
                    if st.button("🗑️", key=f"delete_{chat_id}", use_container_width=True):
                        if len(st.session_state.chat_history) == 1:
                            st.sidebar.warning("Cannot delete the last chat.")
                        else:
                            del st.session_state.chat_history[chat_id]
                            st.session_state.active_chat_id = list(st.session_state.chat_history.keys())[0]
                            
                            save_data = st.session_state.chat_history.copy()
                            save_data["chat_id_counter"] = st.session_state.chat_id_counter
                            save_chat_history(save_data, username)
                            st.rerun()

        st.markdown("---")
        st.markdown("### 2. Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDFs, TXT, or DOCX files",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )

        if st.button("Process Documents", use_container_width=True, type="primary"):
            if uploaded_files:
                new_files_to_process = []
                skipped_file_names = []
                
                for file in uploaded_files:
                    if file.name not in st.session_state.processed_files:
                        new_files_to_process.append(file)
                    else:
                        skipped_file_names.append(file.name)

                if skipped_file_names:
                    st.info(f"Skipped {len(skipped_file_names)} file(s) (already in db): {', '.join(skipped_file_names)}")

                if new_files_to_process:
                    temp_files_info = []
                    new_file_names = [file.name for file in new_files_to_process]
                    
                    with st.spinner(f"Processing {len(new_files_to_process)} new document(s)..."):
                        try:
                            for uploaded_file in new_files_to_process:
                                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                                    tmp_file.write(uploaded_file.getvalue())
                                    temp_files_info.append((tmp_file.name, uploaded_file.name))
                            
                            temp_file_paths_with_original_names = {info[0]: info[1] for info in temp_files_info}
                            
                            st.session_state.vector_store = add_documents_to_store(
                                vector_store=st.session_state.vector_store,
                                file_paths_with_original_names=temp_file_paths_with_original_names,
                                api_key=st.session_state.api_key
                            )
                            
                            st.session_state.processed_files.extend(new_file_names)
                            
                            processed_msg = f"Processed and saved: **{', '.join(new_file_names)}**."
                            st.session_state.chat_history[st.session_state.active_chat_id]["messages"].append({
                                "role": "assistant",
                                "content": processed_msg
                            })
                            
                            save_data = st.session_state.chat_history.copy()
                            save_data["chat_id_counter"] = st.session_state.chat_id_counter
                            save_chat_history(save_data, username)
                            
                            st.success(f"Processed {len(new_file_names)} new document(s)!")
                        except Exception as e:
                            st.error(f"An error occurred: {e}")
                        finally:
                            for file_path, _ in temp_files_info:
                                if os.path.exists(file_path):
                                    os.remove(file_path)
                elif not skipped_file_names:
                    st.warning("Please upload at least one document.")
            else:
                st.warning("Please upload at least one document.")


        st.markdown("---")
        st.markdown("### 3. Your Knowledge Base")
        if st.session_state.processed_files:
            st.info(f"**Files in your DB:**\n" + "\n".join(f"- 📄 {name}" for name in st.session_state.processed_files))
        else:
            st.info("Upload documents to begin.")

    # --- Main Chat Interface ---
    try:
        active_messages = st.session_state.chat_history[st.session_state.active_chat_id]["messages"]
    except KeyError:
        st.session_state.active_chat_id = list(st.session_state.chat_history.keys())[0]
        active_messages = st.session_state.chat_history[st.session_state.active_chat_id]["messages"]

    for message in active_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"]) 
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("Show Sources"):
                    for source in message["sources"]:
                        try:
                            filename = source['metadata'].get('source', 'Unknown File')
                            page_num = source['metadata'].get('page', '?')
                            page_content = source.get('page_content', '')
                            
                            st.markdown(f"**File:** `{filename}` (Page: {page_num + 1})")
                            st.code(f"Content: {page_content[:200]}...", language=None)
                        except Exception as e:
                            st.markdown(f"- Error displaying source: {e}")

    # --- Chat Input Logic ---
    if prompt := st.chat_input("e.g., 'What is supervised learning based on the text?'"):
        
        new_title_generated = False
        try:
            active_chat_data = st.session_state.chat_history[st.session_state.active_chat_id]
            if len(active_chat_data["messages"]) == 1 and (active_chat_data["title"] == "New Chat" or active_chat_data["title"].startswith("Chat ")):
                new_title = generate_chat_title(prompt, st.session_state.api_key)
                st.session_state.chat_history[st.session_state.active_chat_id]["title"] = new_title
                new_title_generated = True
        except Exception as e:
            print(f"Error generating title: {e}")
        
        st.session_state.chat_history[st.session_state.active_chat_id]["messages"].append({"role": "user", "content": prompt})
        
        save_data = st.session_state.chat_history.copy()
        save_data["chat_id_counter"] = st.session_state.chat_id_counter
        save_chat_history(save_data, username)
        
        with st.chat_message("user"):
            st.markdown(prompt)

        if st.session_state.vector_store is None:
            with st.chat_message("assistant"):
                st.warning("Vector store not loaded. Please check API key and restart.")
        else:
            with st.chat_message("assistant"):
                with st.spinner("Analyzing documents..."):
                    try:
                        response_data = get_rag_response(
                            vector_store=st.session_state.vector_store,
                            question=prompt,
                            api_key=st.session_state.api_key
                        )
                        answer = response_data["answer"]
                        sources = response_data["sources"]
                    
                    except Exception as e:
                        st.error(f"Error generating response: {e}")
                        error_msg = f"An error occurred: {e}"
                        st.session_state.chat_history[st.session_state.active_chat_id]["messages"].append({"role": "assistant", "content": error_msg})
                        save_chat_history(st.session_state.chat_history, username)
                        st.rerun()
                        
                st.markdown(answer)
                
                serializable_sources = []
                for source in sources:
                    serializable_sources.append({
                        "page_content": source.page_content,
                        "metadata": source.metadata
                    })

                with st.expander("Show Sources"):
                    for source in serializable_sources:
                        try:
                            filename = source['metadata'].get('source', 'Unknown File')
                            page_num = source['metadata'].get('page', '?')
                            page_content = source.get('page_content', '')
                            
                            st.markdown(f"**File:** `{filename}` (Page: {page_num + 1})")
                            st.code(f"Content: {page_content[:200]}...", language=None)
                        except Exception as e:
                            st.markdown(f"- Error displaying source: {e}")

                st.session_state.chat_history[st.session_state.active_chat_id]["messages"].append({
                            "role": "assistant",
                            "content": answer,
                            "sources": serializable_sources
                        })
                
                save_data = st.session_state.chat_history.copy()
                save_data["chat_id_counter"] = st.session_state.chat_id_counter
                save_chat_history(save_data, username)
                
        if new_title_generated:
            st.rerun()

# --- B. IF NOT LOGGED IN (Show Login/Register Tabs) ---
else:
    login_tab, register_tab = st.tabs(["Login", "Register"])
    
    with login_tab:
        # Call login() here, *inside* the tab
        authenticator.login(location='main')
        if st.session_state["authentication_status"] is False:
            st.error('Username/password is incorrect')
        elif st.session_state["authentication_status"] is None:
            st.info('Please enter your username and password')

    with register_tab:
        try:
            if authenticator.register_user(location='main'):
                st.success('User registered successfully! Please go to the Login tab to sign in.')
                # Save the new user to the config file
                with open('config.yaml', 'w') as file:
                    yaml.dump(config, file, default_flow_style=False)
        except Exception as e:
            st.error(f"Error during registration: {e}")