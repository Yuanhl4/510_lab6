from tempfile import NamedTemporaryFile
import os
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.readers.file import PDFReader

load_dotenv()

st.set_page_config(
    page_title="Song Generation",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

# set api key and model
llm = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE"),
    model="gpt-3.5-turbo",
    temperature=0.7,  # Adjust temperature for creativity
    system_prompt="You are Taylor Swift, a famous singer known for your emotional and catchy lyrics. You will use the provided lyrics as inspiration to generate a new song. Let's create some magic!",
)

# start
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

# initialize
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "What lyrics do you already have? Please upload a PDF with the lyrics."}
    ]

# show the messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

uploaded_file = st.file_uploader("Upload your lyrics in PDF version")
if uploaded_file:
    bytes_data = uploaded_file.read()
    with NamedTemporaryFile(delete=False) as tmp:
        tmp.write(bytes_data)
        with st.spinner(text="Bot is thinking..."):
            reader = PDFReader()
            docs = reader.load_data(tmp.name)
            index = VectorStoreIndex.from_documents(docs)
    os.remove(tmp.name)

    if "chat_engine" not in st.session_state.keys():
        st.session_state.chat_engine = index.as_chat_engine(
            chat_mode="condense_question", verbose=False, llm=llm
        )

# user inputs
if prompt := st.chat_input("What's your question?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.stream_chat(prompt)
            st.write_stream(response.response_gen)
            message = {"role": "assistant", "content": response.response}
    st.session_state.messages.append({"role": "assistant", "content": response})
