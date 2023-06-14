import openai
import pinecone
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

embeddings = OpenAIEmbeddings()

pinecone.init(
    api_key=str(os.environ['PINECONE_API_KEY']),
    environment=str(os.environ['PINECONE_ENV'])
)
index_name = str(os.environ['PINECONE_INDEX_NAME'])

def load_chain():
    docsearch = Pinecone.from_existing_index(index_name, embeddings)
    return docsearch


chain = load_chain()
# From here down is all the StreamLit UI.
st.header("ChatBot Demo")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

user_input = st.text_input("You: ", "Hello, how are you?", key="input")

if user_input:
    docs = chain.similarity_search(user_input)
    output = docs[0].page_content

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
