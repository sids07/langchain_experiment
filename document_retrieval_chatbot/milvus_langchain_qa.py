import os

from langchain.vectorstores import Milvus
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.prompts import PromptTemplate


if __name__ == "__main__":
    
    os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEy"
    embedding = OpenAIEmbeddings()
    
    vector_db = Milvus(
        embedding_function =embedding,
        connection_args = {
            "host": "localhost",
            "port": "19530"
        }
    )
    
    llm = ChatOpenAI(
        temperature=0
    )
    
    # question = "who won FIFA world cup 2010?"
    question = "What is the role of scrum master?"
    
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vector_db.as_retriever(),
    )
    result = qa_chain(
        {
            "query" : question
        }
    )
    print("-"*50)
    print(result)
    
    # Build prompt
    template = """
    Use the following pieces of context to answer the question at the end. If the answer is not present in the context, just say that it is beyond my knowledge base, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
    {context}
    Question: {question}
    Helpful Answer:
    """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    
    # Run chain
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vector_db.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )
    print("-"*50)
    result = qa_chain(
        {
            "query" : question
        }
    )
    
    print(result)
    
    """
    In case of large number of documents, models token limit may be reached. So, chaining can be used to solved this problem.
    1. Map-Reduce: Here, each relevant document is sent to the model and ask to answer independent of other documents. 
                And finally all the answers are sent to LLM and then try to intrepret final answer.
    2. Refine: Here, each document is sent sequentially along with answers from previous documents. 
                First document is sent and asked for answer. And going to next document previous answer is also sent and LLM is asked to improve the answers 
                if given second context is useful.
                
    Time-Consuming as have to call LLM multiple times
    """
    
    qa_chain_mr = RetrievalQA.from_chain_type(
    llm,
    retriever=vector_db.as_retriever(),
    chain_type="map_reduce"
    )
    print("-"*50)
    result = qa_chain_mr({"query": question})
    result["result"]
    
    qa_chain_mr = RetrievalQA.from_chain_type(
    llm,
    retriever=vector_db.as_retriever(),
    chain_type="refine"
    )
    print("-"*50)
    result = qa_chain_mr({"query": question})
    result["result"]
    
    
    ## USE CONTEXTUAL COMPRESSION RETRIEVER
    compressor = LLMChainExtractor.from_llm(llm)
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vector_db.as_retriever()
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=compression_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )
    print("-"*50)
    result = qa_chain(
        {
            "query" : question
        }
    )
    
    print(result)
    