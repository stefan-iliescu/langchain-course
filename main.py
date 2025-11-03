import os
from dotenv import load_dotenv

from langchain.prompts.prompt import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

load_dotenv()

if __name__ == "__main__":
    print("Retrieving...")

    # Initialize embeddings and LLM
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI()

    # Your query
    query = "What is Pinecone in machine learning?"

    # Create a simple prompt template for QA
    prompt_template = """Answer the question based on the context below.
Context: {context}
Question: {question}
Answer:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Create the vector store
    vectorstore = Pinecone.from_existing_index(
        index_name=os.environ["INDEX_NAME"],
        embedding=embeddings
    )

    # Create a chain to combine documents
    combine_docs_chain = StuffDocumentsChain(llm=llm, prompt=prompt)

    # Create retrieval QA chain
    retrieval_chain = RetrievalQA(
        retriever=vectorstore.as_retriever(),
        combine_documents_chain=combine_docs_chain,
        return_source_documents=True  # optional
    )

    # Run the query
    result = retrieval_chain.run(query)

    print("Answer:", result)
