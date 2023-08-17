from langchain.vectorstores import Milvus
import openai
import os
from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.retrievers.self_query.base import SelfQueryRetriever
# from langchain.chains.query_constructor.base import AttributeInfo
from langchain.llms import OpenAI

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor


if __name__ == "__main__":

    openai.api_key = "OPENAI_API_KEy"
    os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEy"
    
    embedding = OpenAIEmbeddings()

    
    ### QUERYING DATA IN MILVUS
    
    vector_db = Milvus(
        embedding_function =embedding,
        connection_args = {
            "host": "localhost",
            "port": "19530"
        }
    )
    
    ## SIMPLE SEMANTIC SIMILARITY SEARCH
    
    questions = "what is the role of scrum master?"
    print("*"*10)
    result_docs = vector_db.similarity_search(
        questions,
        k = 4
    )
    
    for res in result_docs:
        print(res.metadata)
        print(res.page_content)
        
    """
    MAX MARGINAL RELEVANCE SEARCH : EXTRACT NOT ONLY MOST RELEVANT DOCUMENT BUT ALSO DIVERSE ONE THAT MAY HAVE INFORMATION OF THE QUERY.
    It does this by finding the examples with the embeddings that have the greatest cosine similarity with the inputs, 
    and then iteratively adding them while penalizing them for closeness to already selected examples.
    """
    print("*"*10)
    result_mmr = vector_db.max_marginal_relevance_search(
        questions,
        k=3,
        fetch_k=4
    )
    
    for ress in result_mmr:
        print(ress.metadata)
        print(ress.page_content)
        
    
    ## FILTER DOCS WITH METADATA
    
    print("*"*10)
    result_docs = vector_db.similarity_search(
        questions,
        k = 4,
        expr = f'title==\"Mastering the Scrum Master Role: What It Takes to Elevate in the Role with Angela Johnson - YouTube\"'
    )
    
    for res in result_docs:
        print(res.metadata)
        print(res.page_content)
        

    ## LLM to extract metadata info itself (Not supported in milvus)
    
    # metadata_field_info = [
    #     AttributeInfo(
    #         name="title",
    #         description = "The youtube the chunk is from, should be one of `Mastering the Scrum Master Role: What It Takes to Elevate in the Role with Angela Johnson - YouTube`, `Abbie DeMartino: From Sports coach to Scrum Master to Dev Manager - YouTube` or `The HOW to the WHAT: Leading Cross-Functional and Agile Teams with Mikaylah McCarty - YouTube` ",
    #         type="string"
    #     )
    # ]
    
    # document_content_description = "Youtube notes"
    
    # llm = OpenAI(
    #     temperature=0
    # )
    
    # retriever = SelfQueryRetriever.from_llm(
    #     llm,
    #     vector_db,
    #     document_content_description,
    #     metadata_field_info,
    #     verbose=True
    # )
    
    # result_docs = retriever.get_relevant_documents(questions)
    
    # for d in result_docs:
    #     print(d.metadata)
    
    
    ### CONTEXTUAL COMPRESSION
    """
    One challenge with retrieval is that usually you don't know the specific queries your document storage system will face when you ingest data into the system. 
    This means that the information most relevant to a query may be buried in a document with a lot of irrelevant text. 
    Passing that full document through your application can lead to more expensive LLM calls and poorer responses.
    Contextual compression is meant to fix this. The idea is simple: instead of immediately returning retrieved documents as-is, 
    you can compress them using the context of the given query, so that only the relevant information is returned. 
    “Compressing” here refers to both compressing the contents of an individual document and filtering out documents wholesale.
    """
    
    def pretty_print_docs(docs):
        print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))
        
    # Wrap our vectorstore
    llm = OpenAI(temperature=0)
    compressor = LLMChainExtractor.from_llm(llm)
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vector_db.as_retriever()
    )
    
    compressed_docs = compression_retriever.get_relevant_documents(questions)
    print("*"*10)
    pretty_print_docs(compressed_docs)
    
    ## COMBINING MULTIPLE RETRIEVER ACCORDING TO USECASE CONTEXTUAL AND MMR
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vector_db.as_retriever(search_type = "mmr")
    )
    compressed_docs = compression_retriever.get_relevant_documents(questions)
    print("*"*10)
    pretty_print_docs(compressed_docs)
    
