import os

from langchain.vectorstores import Milvus
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import LLMChain


if __name__ == "__main__":
    
    os.environ["OPENAI_API_KEY"] = "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"
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
    
    template = """Use the following pieces of context to answer the question at the end. If the answer is not present in the context, just say that it is beyond my knowledge base, don't try to answer beyond the context. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
    {context}
    Question: {question}
    Helpful Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template=template)
    
    doc_chain = load_qa_with_sources_chain(
        llm = llm,
        prompt = QA_CHAIN_PROMPT
    )
    custom_template = """
    Given following conversation and new question, repharse the new question to be standalone question, in its original language only if new question is a follow up question from the conversation chat history:

    chat history:
    {chat_history}

    new_input:
    {question}
    """
    
    CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

    question_generator = LLMChain(
        llm = ChatOpenAI(
            temperature=0,
            prompt = CUSTOM_QUESTION_PROMPT
        )
    )
    
    compressor = LLMChainExtractor.from_llm(llm)
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vector_db.as_retriever()
    )
    
    chain = ConversationalRetrievalChain(
        retriever=compression_retriever,
        question_generator=question_generator,
        combine_docs_chain=doc_chain,
    )
    chat_history = []
    for i in range(5):
        query = input("Enter your question: ")
        print(chat_history)
        result = chain({
            "question":query,
            "chat_history":chat_history
        })
        print(result["answer"])
        print(result["source"])
        relevant_source = ' '.join(list(set([doc.metadata['source'] for doc in result['source_documents']])))
        print(relevant_source)
        chat_history.extend([(query, result["answer"])])