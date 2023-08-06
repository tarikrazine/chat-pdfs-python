import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import os

load_dotenv()


def get_pdf_text(pdf_files):
    text = ""

    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)

        for page in pdf_reader.pages:
            text += page.extract_text()

    return text


def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(raw_text)

    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

    return conversation_chain


def main():

    st.set_page_config(page_title="Chat with multiple PDF's 📚")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("Chat with multiple PDF's 📚")

    user_question = st.text_input("Ask question about your documents")

    if user_question:
        handle_user_input(user_question)

    st.write(user_template.replace(
        "{{MSG}}", "Hello robot"), unsafe_allow_html=True)
    st.write(bot_template.replace(
        "{{MSG}}", "hello human"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your documents")

        pdf_files = st.file_uploader(
            "Upload your PDF's here and click on 'Process'", accept_multiple_files=True)

        if st.button("Process"):
            with st.spinner("Processing..."):
                # Get pdf text
                raw_text = get_pdf_text(pdf_files)

                # Get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # Create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == "__main__":
    main()
