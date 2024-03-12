from tempfile import NamedTemporaryFile
import os
import streamlit as st
import fitz
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

st.title("Text Extractor")

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_BASE")
)

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = next(stream)["choices"][0]["message"]["content"]
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# File upload
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    pdf = fitz.open(stream=uploaded_file.read())
    extracted_text = ""

    for page in pdf:
        extracted_text += page.get_text()
    pdf.close()
    st.text_area("Extracted Text", extracted_text, height=300)
