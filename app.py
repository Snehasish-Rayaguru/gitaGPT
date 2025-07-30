import streamlit as st
from gita import get_qa_chain


st.set_page_config(page_title="GitaGPT", page_icon="📿")
st.title("🕉️ GitaGPT - Ask Krishna")

st.markdown("Ask spiritual or philosophical questions. Answers are based on the **Bhagavad Gita**.")

@st.cache_resource
def load_qa():
    return get_qa_chain()

qa_chain = load_qa()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask your question to Krishna..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking like Krishna..."):
            response = qa_chain.invoke({"query": prompt})
            st.markdown(response['result'])
            st.session_state.messages.append({"role": "assistant", "content": response['result']})
