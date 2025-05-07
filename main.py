# to run:  python -m streamlit run main.py


import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db

st.title("Coursera's Q&A 🌱")

# Use session state to prevent repeated DB creation
if "db_created" not in st.session_state:
    st.session_state["db_created"] = False

if not st.session_state["db_created"]:
    if st.button("Create Knowledgebase"):
        create_vector_db()
        st.session_state["db_created"] = True
        st.success("Knowledgebase created!")
else:
    st.info("Knowledgebase already created.")

question = st.text_input("Question: ")

if question:
    chain = get_qa_chain()
    response = chain.invoke({"query": question})
    st.header("Answer")
    st.write(response["result"])
