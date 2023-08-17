import os 
import openai
import argparse
from dotenv import load_dotenv, find_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationTokenBufferMemory, ConversationSummaryBufferMemory

from typing import Optional

def parser_args():
    parser = argparse.ArgumentParser(
        description="Arguments to pass for using memory in langchain"
    )
    
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["conversation_buffer_memory", "conversation_buffer_window_memory", "conversation_token_buffer_memory","conversation_summary_buffer_memory"],
        help="Enter the type of memory you want to use for langchain conversation"
    )
    
    parser.add_argument(
        "--buffer-window-size",
        default=2,
        help="If type is Conversation Buffer Window you must enter window size"
    )
    
    parser.add_argument(
        "--max_token_limit",
        type = int,
        default=30,
        help="Enter token limit for Conversation Token Buffer Memory"
    )
    
    return parser.parse_args()
    
    
    
class LangchainMemory:
    
    def __init__(
        self, 
        memory_type: str , 
        buffer_window_size:Optional[int] = None, 
        max_token_limit:Optional[int]= None, 
        model_name:str="gpt-3.5-turbo"
        ) -> None:
        
        key = self._get_openai_key()
        openai.api_key = key
        
        self.chat = ChatOpenAI(
            temperature = 0.0,
            model_name =model_name
        )
        
        self.memory_type = memory_type
        self.buffer_window_size = buffer_window_size
        self.max_token_limit = max_token_limit
        
        self.memory = self._initialize_memory()
        
        self.conversation = ConversationChain(
            llm=self.chat, 
            memory = self.memory,
            verbose=True
        )
    
    @staticmethod
    def _get_openai_key() -> str:
        _ = load_dotenv(find_dotenv())
        return os.environ['OPENAI_API_KEY']
    
    def _initialize_memory(self):
        if self.memory_type == "conversation_buffer_memory":
            memory = ConversationBufferMemory()
            
        elif self.memory_type == "conversation_buffer_window_memory":
            if self.buffer_window_size is None:
                raise ValueError("Buffer Window Size must be specified")
            
            else:
                memory = ConversationBufferWindowMemory(
                    k = self.buffer_window_size
                )
        else:
            if self.max_token_limit is None:
                raise ValueError("Max token limit must be specified")     
            elif self.memory_type == "conversation_token_buffer_memory":
                memory = ConversationTokenBufferMemory(
                    llm = self.chat,
                    max_token_limit = self.max_token_limit
                )   
            else:
                memory = ConversationSummaryBufferMemory(
                    llm=self.chat, 
                    max_token_limit=self.max_token_limit
                )
        
        return memory
    
    def save_context(self, context: dict):
        return self.memory.save_context(context)
    
    def predict(self, input_query):
        return self.conversation.predict(input=input_query)
         