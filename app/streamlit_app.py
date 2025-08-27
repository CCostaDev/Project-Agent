import os
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# --- LLM selection (Azure OpenAI or OpenAI) -----------------------------------
def get_chat_llm():
    """
    Returns a chat LLM for Azure OpenAI (if AZURE_* vars and chat deployment set)
    or standard OpenAI (if OPENAI_API_KEY is set).
    """
    az_key = os.getenv("AZURE_OPENAI_API_KEY")
    az_ep = os.getenv("AZURE_OPENAI_ENDPOINT")
    az_ver = os.getenv("AZURE_OPENAI_API_VERSION")
    az_chat_deploy = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

    if az_key and az_ep and az_ver and az_chat_deploy:
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            api_key=az_key,
            azure_endpoint=az_ep,
            api_version=az_ver,
            deployment_name=az_chat_deploy,
            temperature=0,
        )

    # Fallback: standard OpenAI
    std_key = os.getenv("OPENAI_API_KEY")
    if std_key:
        from langchain_openai import ChatOpenAI
        model = os.getenv("CHAT_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model, temperature=0)

    return None

# --- Embeddings wrapper for retriever -----------------------------------------
def get_embeddings():
    az_key = os.getenv("AZURE_OPENAI_API_KEY")
    az_ep = os.getenv("AZURE_OPENAI_ENDPOINT")
    az_ver = os.getenv("AZURE_OPENAI_API_VERSION")
    az_embed_deploy = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

    if az_key and az_ep and az_ver and az_embed_deploy:
        from langchain_openai import AzureOpenAIEmbeddings
        return AzureOpenAIEmbeddings(
            api_key=az_key,
            azure_endpoint=az_ep,
            api_version=az_ver,
            deployment=az_embed_deploy,
        )

    std_key = os.getenv("OPENAI_API_KEY")
    if std_key:
        from langchain_openai import OpenAIEmbeddings
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddings(model=model)

    return None

# --- App ----------------------------------------------------------------------
def format_docs(docs):
    lines = []
    for d in docs:
        name = d.metadata.get("name", "source")
        lines.append(f"[{name}]\n{d.page_content}")
    return "\n\n".join(lines)

def main():
    load_dotenv()

    st.set_page_config(page_title="Training Support Chatbot", layout="centered")
    st.title("Training Support Chatbot (Local Prototype)")

    # Sidebar helpers
    with st.sidebar:
        st.subheader("Search settings")
        top_k = st.slider("Top-K results", 1, 10, 4, help="How many chunks to retrieve from Chroma")
        st.caption("Make sure you ran the embedding script first: `python pipelines/embed_local.py`.")

    # Connect to Chroma with the same embeddings you used for indexing
    embeddings = get_embeddings()
    if embeddings is None:
        st.error(
            "No embedding provider configured.\n\n"
            "Set Azure vars (AZURE_OPENAI_* + AZURE_OPENAI_EMBEDDING_DEPLOYMENT) "
            "or OPENAI_API_KEY, then re-run the embed script."
        )
        return

    chroma_path = os.getenv("CHROMA_DB_PATH", ".chroma")
    db = Chroma(persist_directory=chroma_path, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": top_k})

    llm = get_chat_llm()
    if llm is None:
        st.warning(
            "No chat model configured. Answers won’t generate.\n\n"
            "Set Azure chat deployment (AZURE_OPENAI_* + AZURE_OPENAI_CHAT_DEPLOYMENT) "
            "or OPENAI_API_KEY + CHAT_MODEL."
        )

    SYSTEM = (
        "You are an internal training assistant. "
        "ONLY use the provided context to answer. "
        "If the answer is not in the context, say you don't know."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM),
        ("human", "Question: {question}\n\nContext:\n{context}")
    ])

    q = st.chat_input("Ask a training question…")
    if q:
        with st.spinner("Searching…"):
            docs = retriever.invoke(q)

        if not docs:
            st.info("No results found in the local index. Try another question or re-run the embedding step.")
            return

        context = format_docs(docs)

        if llm is None:
            # Show the top chunk to prove retrieval works even if LLM isn’t set.
            st.markdown("### Retrieved context (no LLM configured yet)")
            st.code(context[:1200] + ("…" if len(context) > 1200 else ""))
            return

        with st.spinner("Generating answer…"):
            answer = llm.invoke(prompt.format(question=q, context=context))

        st.markdown("### Answer")
        st.write(answer.content)

        with st.expander("Sources"):
            for i, d in enumerate(docs, start=1):
                name = d.metadata.get("name", f"chunk {i}")
                st.markdown(f"- **{name}**")

if __name__ == "__main__":
    main()