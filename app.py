import streamlit as st
from src.rag_pipeline import ask

st.set_page_config(
    page_title="CrediTrust Complaint Analysis Assistant",
    page_icon="💳"
)

st.title("CrediTrust Complaint Analysis Assistant")

# Session state
if "question" not in st.session_state:
    st.session_state.question = ""

question = st.text_input(
    "Ask a question about customer complaints",
    key="question"
)

col1, col2 = st.columns(2)

with col1:
    ask_button = st.button("Ask")

with col2:
    clear_button = st.button("Clear")

if clear_button:
    st.session_state.question = ""
    st.rerun()

if ask_button and question.strip():

    try:
        answer, sources = ask(question)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Retrieved Sources")

        for i, source in enumerate(sources, 1):
            st.markdown(f"**Source {i}**")
            st.write(source[:500] + "...")
            st.divider()

    except Exception as e:
        st.error(f"Error: {e}")