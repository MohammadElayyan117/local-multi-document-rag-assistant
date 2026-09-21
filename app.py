import os
import json
import hashlib
import shutil

import streamlit as st

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma


# --------------------------------------------------
# Project settings
# --------------------------------------------------

UPLOAD_DIR = "uploaded_pdfs"
CHROMA_DIR = "chroma_uploads"

REGISTRY_FILE = "document_registry.json"
CHAT_HISTORY_FILE = "chat_histories.json"

MAX_CHAT_MESSAGES = 50


os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)

os.makedirs(
    CHROMA_DIR,
    exist_ok=True
)


# --------------------------------------------------
# Page settings
# --------------------------------------------------

st.set_page_config(
    page_title="Local Multi-Document RAG Assistant",
    page_icon="🤖",
    layout="centered"
)


# --------------------------------------------------
# Document registry
# --------------------------------------------------

def load_registry():

    if not os.path.exists(
        REGISTRY_FILE
    ):
        return {}

    try:

        with open(
            REGISTRY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception:

        return {}


def save_registry(
    registry_data
):

    with open(
        REGISTRY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            registry_data,
            file,
            ensure_ascii=False,
            indent=2
        )


registry = load_registry()


# --------------------------------------------------
# Chat history
# --------------------------------------------------

def load_chat_histories():

    if not os.path.exists(
        CHAT_HISTORY_FILE
    ):
        return {}

    try:

        with open(
            CHAT_HISTORY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            histories = json.load(file)

        # Keep only the latest 50 messages
        for document_hash in histories:

            histories[document_hash] = (
                histories[document_hash][
                    -MAX_CHAT_MESSAGES:
                ]
            )

        return histories

    except Exception:

        return {}


def save_chat_histories():

    with open(
        CHAT_HISTORY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            st.session_state.chat_histories,
            file,
            ensure_ascii=False,
            indent=2
        )


def append_chat_message(
    document_hash,
    message
):

    if (
        document_hash
        not in st.session_state.chat_histories
    ):

        st.session_state.chat_histories[
            document_hash
        ] = []


    st.session_state.chat_histories[
        document_hash
    ].append(
        message
    )


    # Keep only latest 50 messages
    st.session_state.chat_histories[
        document_hash
    ] = (
        st.session_state.chat_histories[
            document_hash
        ][
            -MAX_CHAT_MESSAGES:
        ]
    )


    save_chat_histories()


# --------------------------------------------------
# Load local AI models
# --------------------------------------------------

@st.cache_resource
def load_models():

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text:latest"
    )

    llm = ChatOllama(
        model="llama3.1:8b",
        temperature=0
    )

    return embeddings, llm


embeddings, llm = load_models()


# --------------------------------------------------
# Save uploaded PDF
# --------------------------------------------------

def save_uploaded_pdf(
    uploaded_file
):

    file_bytes = (
        uploaded_file.getvalue()
    )


    file_hash = hashlib.sha256(
        file_bytes
    ).hexdigest()[:12]


    file_path = os.path.join(
        UPLOAD_DIR,
        f"{file_hash}.pdf"
    )


    if not os.path.exists(
        file_path
    ):

        with open(
            file_path,
            "wb"
        ) as file:

            file.write(
                file_bytes
            )


    if file_hash not in registry:

        registry[file_hash] = {
            "name": uploaded_file.name,
            "path": file_path
        }


        save_registry(
            registry
        )


    return file_hash


# --------------------------------------------------
# Delete document
# --------------------------------------------------

def delete_document(
    document_hash
):

    document_info = registry.get(
        document_hash
    )


    # Delete saved PDF
    if document_info:

        file_path = document_info.get(
            "path"
        )

        if (
            file_path
            and os.path.exists(file_path)
        ):

            try:

                os.remove(
                    file_path
                )

            except Exception:

                pass


    # Delete ChromaDB folder
    chroma_path = os.path.join(
        CHROMA_DIR,
        document_hash
    )


    if os.path.exists(
        chroma_path
    ):

        try:

            shutil.rmtree(
                chroma_path
            )

        except Exception:

            pass


    # Remove from registry
    registry.pop(
        document_hash,
        None
    )


    save_registry(
        registry
    )


    # Remove chat history
    st.session_state.chat_histories.pop(
        document_hash,
        None
    )


    save_chat_histories()


# --------------------------------------------------
# Load or build ChromaDB
# --------------------------------------------------

def load_document_vector_store(
    file_hash,
    file_path,
    file_name
):

    persist_directory = os.path.join(
        CHROMA_DIR,
        file_hash
    )


    collection_name = (
        f"pdf_{file_hash}"
    )


    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory
    )


    existing_count = (
        vector_store._collection.count()
    )


    reader = PdfReader(
        file_path
    )


    page_count = len(
        reader.pages
    )


    # Already indexed
    if existing_count > 0:

        return (
            vector_store,
            existing_count,
            page_count
        )


    documents = []


    # Extract text page by page
    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = (
            page.extract_text()
            or ""
        )


        if text.strip():

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "page_label": str(
                            page_number
                        ),
                        "source": file_name
                    }
                )
            )


    # Split text into chunks
    text_splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
    )


    chunks = (
        text_splitter.split_documents(
            documents
        )
    )


    # Store chunks
    if chunks:

        vector_store.add_documents(
            chunks
        )


    chunk_count = (
        vector_store._collection.count()
    )


    return (
        vector_store,
        chunk_count,
        page_count
    )


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "chat_histories" not in st.session_state:

    st.session_state.chat_histories = (
        load_chat_histories()
    )


if "selected_doc_hash" not in st.session_state:

    st.session_state.selected_doc_hash = None


if "uploader_version" not in st.session_state:

    st.session_state.uploader_version = 0


# Defaults
vector_store = None
active_document_name = None
active_document_hash = None


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header(
        "RAG System"
    )


    # Upload PDF
    uploaded_file = st.file_uploader(
        "Upload New PDF",
        type=["pdf"],
        key=(
            f"pdf_uploader_"
            f"{st.session_state.uploader_version}"
        )
    )


    if uploaded_file is not None:

        uploaded_hash = hashlib.sha256(
            uploaded_file.getvalue()
        ).hexdigest()[:12]


        # New document
        if uploaded_hash not in registry:

            new_hash = save_uploaded_pdf(
                uploaded_file
            )


            registry = load_registry()


            st.session_state.selected_doc_hash = (
                new_hash
            )


            st.session_state.chat_histories.setdefault(
                new_hash,
                []
            )


            save_chat_histories()


    st.divider()


    # --------------------------------------------------
    # Saved documents
    # --------------------------------------------------

    if registry:

        document_hashes = list(
            registry.keys()
        )


        if (
            st.session_state.selected_doc_hash
            not in document_hashes
        ):

            st.session_state.selected_doc_hash = (
                document_hashes[-1]
            )


        # Synchronize selectbox
        if (
            "saved_document_selector"
            not in st.session_state
            or
            st.session_state.saved_document_selector
            not in document_hashes
        ):

            st.session_state.saved_document_selector = (
                st.session_state.selected_doc_hash
            )


        selected_hash = st.selectbox(
            "Saved Documents",
            options=document_hashes,
            format_func=lambda value: (
                registry[value]["name"]
            ),
            key="saved_document_selector"
        )


        st.session_state.selected_doc_hash = (
            selected_hash
        )


        active_document_hash = (
            selected_hash
        )


        # Create chat history if missing
        st.session_state.chat_histories.setdefault(
            active_document_hash,
            []
        )


        active_document = registry[
            active_document_hash
        ]


        active_document_name = (
            active_document["name"]
        )


        active_document_path = (
            active_document["path"]
        )


        # --------------------------------------------------
        # Delete document
        # --------------------------------------------------

        confirm_delete = st.checkbox(
            "Confirm document deletion"
        )


        if st.button(
            "Delete Current Document",
            use_container_width=True,
            disabled=not confirm_delete
        ):

            delete_document(
                active_document_hash
            )


            st.session_state.selected_doc_hash = (
                None
            )


            st.session_state.saved_document_selector = (
                None
            )


            # Reset file uploader
            st.session_state.uploader_version += 1


            st.rerun()


        # --------------------------------------------------
        # Load document database
        # --------------------------------------------------

        with st.spinner(
            "Loading document..."
        ):

            (
                vector_store,
                chunk_count,
                page_count
            ) = load_document_vector_store(
                active_document_hash,
                active_document_path,
                active_document_name
            )


        if chunk_count > 0:

            st.success(
                "PDF ready for questions"
            )

        else:

            st.error(
                "No readable text was found in this PDF."
            )

            vector_store = None


        st.write(
            "Document:"
        )


        st.code(
            active_document_name
        )


        st.caption(
            f"Pages: {page_count}"
        )


        st.caption(
            f"Chunks in ChromaDB: {chunk_count}"
        )


        current_message_count = len(
            st.session_state.chat_histories[
                active_document_hash
            ]
        )


        st.caption(
            f"Saved chat messages: "
            f"{current_message_count} / "
            f"{MAX_CHAT_MESSAGES}"
        )


    else:

        st.info(
            "Upload a PDF to start."
        )


    # --------------------------------------------------
    # System information
    # --------------------------------------------------

    st.divider()


    with st.expander(
        "⚙️ System Details"
    ):

        st.markdown(
            "**LLM**  \n"
            "Llama 3.1 8B"
        )

        st.markdown(
            "**Embedding Model**  \n"
            "nomic-embed-text"
        )

        st.markdown(
            "**Vector Database**  \n"
            "ChromaDB"
        )

        st.markdown(
            "**Retrieval Strategy**  \n"
            "MMR"
        )

        st.markdown(
            "**Chat History**  \n"
            "Persistent per document"
        )


    st.divider()


    # Clear chat for current document only
    if st.button(
        "Clear Current Chat",
        use_container_width=True,
        disabled=active_document_hash is None
    ):

        st.session_state.chat_histories[
            active_document_hash
        ] = []


        save_chat_histories()


        st.rerun()


# --------------------------------------------------
# Main page
# --------------------------------------------------

st.title(
    "🤖 Local Multi-Document RAG Assistant"
)


st.write(
    "A fully local RAG application for uploading, indexing, managing, and querying multiple PDF documents with persistent chat history and source-grounded answers."
)

with st.expander(
    "✨ Project Highlights"
):

    st.markdown(
        "- **Multi-PDF document management**\n"
        "- **Fully local LLM and embeddings**\n"
        "- **Persistent per-document chat history**\n"
        "- **Source-grounded answers with page references**\n"
        "- **MMR retrieval with ChromaDB**\n"
        "- **RAGAS evaluation for response quality**"
    )


# --------------------------------------------------
# Current document chat
# --------------------------------------------------

if active_document_hash is not None:

    current_messages = (
        st.session_state.chat_histories[
            active_document_hash
        ]
    )

else:

    current_messages = []


# --------------------------------------------------
# Display chat
# --------------------------------------------------

for message in current_messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


        if "sources" in message:

            with st.expander(
                "Sources"
            ):

                for index, source in enumerate(
                    message["sources"],
                    start=1
                ):

                    st.markdown(
                        f"**Source {index} — Page {source['page']}**"
                    )


                    st.write(
                        source["text"]
                    )


                    st.divider()


# --------------------------------------------------
# Chat input
# --------------------------------------------------

question = st.chat_input(
    "Ask a question about the selected PDF...",
    disabled=vector_store is None
)


# --------------------------------------------------
# RAG pipeline
# --------------------------------------------------

if (
    question
    and vector_store is not None
    and active_document_hash is not None
):


    # Save user message
    append_chat_message(
        active_document_hash,
        {
            "role": "user",
            "content": question
        }
    )


    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )


    # Retrieve candidate chunks
    candidate_results = (
        vector_store.max_marginal_relevance_search(
            question,
            k=8,
            fetch_k=15,
            lambda_mult=0.8
        )
    )


    # Remove very short chunks
    results = [
        doc
        for doc in candidate_results
        if len(
            doc.page_content.strip()
        ) >= 120
    ][:4]


    # Fallback
    if not results:

        results = (
            candidate_results[:4]
        )


    # Build context
    context_parts = []


    for doc in results:

        page = doc.metadata.get(
            "page_label",
            "Unknown"
        )


        context_parts.append(
            f"[Page {page}]\n"
            f"{doc.page_content}"
        )


    context = "\n\n".join(
        context_parts
    )


    # --------------------------------------------------
    # Grounded prompt
    # --------------------------------------------------

    prompt = f"""
You are a RAG assistant answering questions about an uploaded PDF document.

Answer the user's exact question using ONLY the context provided below.

Do not use outside knowledge.

Prioritize context that directly answers the question.
Ignore retrieved context that is only loosely related.

Start with a direct answer to the question.

If the question asks for multiple items, include all distinct items supported by the retrieved context and avoid duplicates.

If the answer cannot be found in the context, say:
"I could not find this information in the document."

Keep the answer clear and concise.

Document:
{active_document_name}

Question:
{question}

Context:
{context}

Answer:
"""


    # --------------------------------------------------
    # Generate answer
    # --------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Searching the document..."
        ):

            response = llm.invoke(
                prompt
            )


        answer = (
            response.content
        )


        st.markdown(
            answer
        )


        # Sources
        source_details = []


        for doc in results:

            source_details.append(
                {
                    "page": (
                        doc.metadata.get(
                            "page_label",
                            "Unknown"
                        )
                    ),
                    "text": (
                        doc.page_content
                    )
                }
            )


        with st.expander(
            "Sources"
        ):

            for index, source in enumerate(
                source_details,
                start=1
            ):

                st.markdown(
                    f"**Source {index} — Page {source['page']}**"
                )


                st.write(
                    source["text"]
                )


                st.divider()


    # Save assistant response
    append_chat_message(
        active_document_hash,
        {
            "role": "assistant",
            "content": answer,
            "sources": source_details
        }
    )


    # --------------------------------------------------
    # Save latest interaction for RAGAS
    # --------------------------------------------------

    evaluation_data = {

        "document": (
            active_document_name
        ),

        "question": question,

        "answer": answer,

        "contexts": [
            doc.page_content
            for doc in results
        ],

        "sources": [
            doc.metadata.get(
                "page_label",
                "Unknown"
            )
            for doc in results
        ]
    }


    with open(
        "evaluation_sample.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            evaluation_data,
            file,
            ensure_ascii=False,
            indent=2
        )