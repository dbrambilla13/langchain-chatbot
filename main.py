import os
import utils
import streamlit as st
from streaming import StreamHandler

import bs4

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch

from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma


from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


st.set_page_config(page_title="Chat", page_icon="favicon.ico")
st.header("Chat")
st.write("some description")


@st.spinner("Setup LLM")
@st.cache_resource
def setup_llm(openai_model):

    llm = ChatOpenAI(model_name=openai_model, temperature=0, streaming=True)
    return llm


@st.spinner("Analyzing documents..")
@st.cache_resource
def setup_document_retrival(web_sources_list):

    # Load, chunk and index the contents of the papers at the following links

    # searching for works with concept hypoxia in OpenAlex
    # https://api.openalex.org/works?filter=concept.id:C7836513,from_publication_date:2022-04-29&per-page=200

    loader = WebBaseLoader(web_paths=web_sources_list)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    return retriever


def setup_qa_chain(llm, retriever):

    # Setup memory for contextual conversation
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Setup QA chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm, retriever=retriever, memory=memory, verbose=True
    )
    return qa_chain


@utils.enable_chat_history
def main():

    example_web_sources_list = [
        "https://doi.org/10.1172/jci159839",
        "https://doi.org/10.1186/s13045-022-01292-6",
        "https://doi.org/10.1038/s41581-022-00587-8",
        "https://doi.org/10.1038/s41392-022-01080-1",
        "https://doi.org/10.1038/s41392-023-01332-8",
        "https://doi.org/10.1016/j.semcancer.2020.09.011",
        "https://doi.org/10.1016/j.redox.2022.102312",
        "https://doi.org/10.1126/science.abg9302",
        "https://doi.org/10.1186/s12943-022-01645-2",
        "https://doi.org/10.1038/s41590-022-01379-9",
    ]

    openai_model = utils.configure_openai()

    user_query = st.chat_input(placeholder="Ask me anything!")

    llm = setup_llm(openai_model)

    retriever = setup_document_retrival(example_web_sources_list)

    qa_chain = setup_qa_chain(llm=llm, retriever=retriever)

    if user_query:

        utils.display_msg(user_query, "user")

        with st.chat_message("assistant"):
            st_cb = StreamHandler(st.empty())
            result = qa_chain.invoke({"question": user_query}, {"callbacks": [st_cb]})
            response = result["answer"]
            st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":

    main()
