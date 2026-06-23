import streamlit as st
from src.rag_pipeline import ask

st.title(
    "CrediTrust Complaint Analysis Assistant"
)

question = st.text_input(
    "Ask a question about customer complaints"
)

if st.button("Ask"):

    answer, sources = ask(question)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")

    for i, source in enumerate(sources, 1):
        st.markdown(f"**Source {i}**")
        st.write(source[:500] + "...")


if st.button("Clear"):
    st.session_state.clear()