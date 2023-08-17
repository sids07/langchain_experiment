"""
1.  Given following conversation and new question, repharse the new question to be standalone question, in its original language only if new question is a follow up question from the conversation chat history:

    chat history:
    {history}

    new_input:
    {query}


2.  Now new standalone question is passed to document retriever and relevant docs are retrieved.
3.  And finally, LLM is called with relevant context.
"""
import os

from langchain.vectorstores import Milvus
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
# from langchain.chains.question_answering import load_qa_chain
# from langchain.chains import LLMChain
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
    temperature=0,
    model="gpt-4"
)

template = """
Strictly only use the following pieces of context to answer the question at the end. \
If the answer is not present in the context, just say that it is beyond my knowledge base, don't try to answer beyond the context. \
If no context is obtained just say it is beyond my knowledge base. \
Keep the answer as concise as possible.
{context}
Question: {question}
Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)

custom_template = """
Given following conversation and new question, repharse the new question to be standalone question, in its original language only if new question is a follow up question from the conversation chat history:

chat history:
{chat_history}

new_input:
{question}
"""

CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vector_db.as_retriever()
)

chat_history = []
# memory = ConversationBufferMemory(
#     memory_key="chat_history",
#     return_messages=True
# )

bots = ConversationalRetrievalChain.from_llm(
    llm,
    # memory=memory,
    retriever= compression_retriever,
    condense_question_llm = ChatOpenAI(
        temperature = 0,
        model = "gpt-3.5-turbo"
    ),
    condense_question_prompt = CUSTOM_QUESTION_PROMPT,
    combine_docs_chain_kwargs = {
        "prompt": QA_CHAIN_PROMPT
    },
    return_source_documents=True,
    return_generated_question=True,
    # verbose= True
    )

def respond(query, chat_history):
    langchain_chat_history = [(lst[0],lst[1]) for lst in chat_history]
    print(langchain_chat_history)
    bot_message = bots({
        "question":query,
        "chat_history":langchain_chat_history
    })
    if len(bot_message["source_documents"])>0:
        relevant_source = ' '.join(list(set([doc.metadata['source'] for doc in bot_message['source_documents']])))
        answer = bot_message["answer"]+ " Source: "+ relevant_source
    else:
        answer = bot_message["answer"]
    chat_history.append((query, answer))
    return "",chat_history

if __name__ == "__main__":
    
    # os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEy" #upacare
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
        temperature=0,
        model="gpt-4"
    )
    
    template = """
    Strictly only use the following pieces of context to answer the question at the end. \
    If the answer is not present in the context, just say that it is beyond my knowledge base, don't try to answer beyond the context. \
    If no context is obtained just say it is beyond my knowledge base. \
    Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
    {context}
    Question: {question}
    Helpful Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)
    
    custom_template = """
    Given following conversation and new question, repharse the new question to be standalone question, in its original language only if new question is a follow up question from the conversation chat history:

    chat history:
    {chat_history}

    new_input:
    {question}
    """
    
    CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

    compressor = LLMChainExtractor.from_llm(llm)
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vector_db.as_retriever()
    )
    
    chat_history = []
    # memory = ConversationBufferMemory(
    #     memory_key="chat_history",
    #     return_messages=True
    # )
    
    bots = ConversationalRetrievalChain.from_llm(
        llm,
        # memory=memory,
        retriever= compression_retriever,
        condense_question_llm = ChatOpenAI(
            temperature = 0,
            model = "gpt-3.5-turbo"
        ),
        condense_question_prompt = CUSTOM_QUESTION_PROMPT,
        combine_docs_chain_kwargs = {
            "prompt": QA_CHAIN_PROMPT
        },
        return_source_documents=True,
        return_generated_question=True,
        # verbose= True
        )
        
    for i in range(5):
        query = input("Enter your question: ")
        print(chat_history)
        result = bots({
            "question":query,
            "chat_history":chat_history
        })
        # print(result)
        print(result["answer"])
        # print(result["source"])
        relevant_source = ' '.join(list(set([doc.metadata['source'] for doc in result['source_documents']])))
        print(relevant_source)
        # if len(relevant_source)<
        chat_history.extend([(query, result["answer"])])
        # print(result)
        # print(result["answer"])