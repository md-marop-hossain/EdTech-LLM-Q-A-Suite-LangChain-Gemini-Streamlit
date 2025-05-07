# installed packages:
# pip install --upgrade --quiet langchain-google-genai langchain faiss-cpu langchain-community
# pip install -qU huggingface-hub==0.25.2 InstructorEmbedding==1.0.1 sentence-transformers==2.2.2 transformers==4.37.2


from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import RetrievalQA
# from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
# from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS

import os
from dotenv import load_dotenv

load_dotenv()

llm = GoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"],
    temperature=0.1,
    max_output_tokens=1024
)

embeddings = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-large",
    model_kwargs={"device": "cpu"}  # Explicit device configuration
)

vectordb_file_path = "faiss_index"


def create_vector_db():
    loader = CSVLoader(file_path='codebasics_faqs.csv', source_column="prompt", encoding='latin-1')
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data, embedding=embeddings)
    vectordb.save_local(vectordb_file_path)


def get_qa_chain():
    # Load with corrected embeddings variable and deserialization flag
    vectordb = FAISS.load_local(
        vectordb_file_path,
        embeddings,
        allow_dangerous_deserialization=True  # Required in newer LangChain versions
    )
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return chain


if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()
    print(chain.invoke({"query": "Do you guys provide internship and also do you offer EMI payments?"}))
    print(chain.invoke({"query": "should I learn power bi or tableau?"}))
    print(chain.invoke({"query": "I've a MAC computer. Can I use powerbi on it?"}))

