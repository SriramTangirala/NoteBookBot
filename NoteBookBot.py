import streamlit as st
from PyPDF2 import PdfReader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

st.header("NoteBookBot")

with st.sidebar:
    st.title("My Notes")
    file = st.file_uploader("Upload the textbook here", type = "pdf")

# Extracting the text from the pdf file
if file is not None:
    my_pdf = PdfReader(file)
    text = ""
    for page in my_pdf.pages:
        text += page.extract_text()
        st.write(text)


# Breaking the text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size = 250, chunk_overlap = 50)
    chunks = splitter.split_text(text)
    st.write(chunks)

    # Creating an object of OpenAIEmbeddings class that let us connect with OpenAI's Embedding models
    embeddings = OpenAIEmbeddings(api_key = "Add API key here")
    # Creating Vector Database and storing embeddings into the database
    vector_store = FAISS.from_texts(chunks, embeddings)
    # Get User Query
    user_query = st.text_input("Enter you Query here")
    # Semantic search from Vector store
    if user_query:
        matching_chunks = vector_store.similarity_search(user_query)
        # It converts the user query into embeddings and then perform similarity search
        # Define LLM (GPT 3.5 Turbo)
        LLM = ChatOpenAI(
            api_key= "Add API key here",
            max_tokens = 300,
            temperature = 0,
            model = "gpt-3.5-turbo"
        )

        # Method 1

        # Generate Response
        chain = load_qa_chain(LLM, chain_type = "stuff")
        # (chain_type = "stuff") means concatenating the user query and relevant chunks and passing to the LLM
        output = chain.run(question = user_query, input_documents = matching_chunks)
        st.write(output)


        # # Method 2
        # customized_prompt = ChatPromptTemplate.from_template(
        #     """You are my assistant tutor. Answer the questions based on the following context and if you did not get the context, say "I am unaware of this":
        #     {context}
        #     Question: {input}
        #     """
        # )
        #
        # chain = create_stuff_documents_chain(LLM, customized_prompt)
        # output = chain.invoke({"input": user_query, "context": matching_chunks})
        # st.write(output)