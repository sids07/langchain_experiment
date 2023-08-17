from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
import openai
import os
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser,ResponseSchema, StructuredOutputParser
from pydantic import BaseModel, Field
from typing import Any
import argparse

class OpenAIOutputParser:
    def __init__(self, model_name="gpt-3.5-turbo"):
        key = self._get_openai_key()
        openai.api_key = key
        self.chat = ChatOpenAI(
            temperature = 0.0,
            model_name =model_name
        )
        
    @staticmethod
    def _get_openai_key():
        _ = load_dotenv(find_dotenv())
        return os.environ['OPENAI_API_KEY']

    def get_response(self, message):
        response = self.chat(
            message
        )
        return response.content
    
    def get_structured_output_parser(self,name2description):
        response_schema = [ResponseSchema(name=name,description=description) for name, description in name2description.items()]
        return StructuredOutputParser(
            response_schemas = response_schema
        )
    
    def get_pydantic_output_parser(self, pydantic_object):
        return PydanticOutputParser(
            pydantic_object = pydantic_object
        )
        
def parse_args():
    parser = argparse.ArgumentParser(
        description="Output Parsers in Langchain"
    )

    parser.add_argument(
        "--type",
        type=str,
        choices=["structured","pydantic"],
        default="structured",
        help="Enter your choice of output parser"
    )
    
    args = parser.parse_args()
    return args
        
if __name__ == "__main__":
    args = parse_args()
    
    openai_output_parser = OpenAIOutputParser()
    
    template = """ \
    For the following text, extract the information about gift, delivery_days, price_value:
    text: {text}
    {format_instructions}
    """
    
    prompt_template = ChatPromptTemplate.from_template(
        template
    )
    
    text = """ \
    I have bought a new car for my wife as a gift for our 5th year anniversary. \
    It arrived after 3 days of waiting. Though the price of the car was not so high since it was average car. \
    But my wife was more than happy and excited to have it.
    """
    
    if args.type == "structured":
        name2description = {
        "gift":"Was the item purchased as gift for someone else? Answer True if yes, False if not or unknown.",
        "delivery_days":"How many days did it take for the product to deliver? If the information is not found output -1.",
        "price_value":"Extract any sentences about values or price. and output them as comman separated python list"
        }
        
        output_parser = openai_output_parser.get_structured_output_parser(
            name2description= name2description
        )
    
    elif args.type == "pydantic":
        class Information(BaseModel):
            gift: Any = Field(description="Was the item purchased as gift for someone else? Answer True if yes, False if not or unknown.")
            delivery_days: Any = Field(description="How many days did it take for the product to deliver? If the information is not found output -1.")
            price_value: Any = Field(description="Extract any sentences about values or price. and output them as comman separated python list")

        output_parser = openai_output_parser.get_pydantic_output_parser(
            pydantic_object= Information
        )
    
    message = prompt_template.format_messages(
        text = text,
        format_instructions = output_parser.get_format_instructions()
    )
    
    response = openai_output_parser.get_response(
        message= message
    )
    
    formatted_response = output_parser.parse(response)
    print(formatted_response)
    