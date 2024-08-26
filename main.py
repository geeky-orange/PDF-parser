import streamlit as st
import pandas as pd
import os
import json
from io import StringIO
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    download_loader,
    RAKEKeywordTableIndex,
)
import ast
import re
from llama_parse.base import ResultType, Language
from llama_parse import LlamaParse
from dotenv import load_dotenv
load_dotenv()

# bring in deps
import nest_asyncio
nest_asyncio.apply()


os.environ["OPENAI_API_KEY"] = "sk-6ktjARpTQuanwCV3s8qTwzCeXe2orB7UZgz80Jy6N3bUBEma"
os.environ["OPENAI_API_BASE"] = "https://api.chatanywhere.tech/v1"
os.environ['LLAMA_CLOUD_API_KEY']= "llx-TAOQ6V2eepq1Pj7L2o4Y2cZgtbdmsWFn2vP9D6TCg29zKlmM"




# Streamlit app
st.title("PDF Parser")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
parser = LlamaParse(result_type=ResultType.MD,language=Language.ENGLISH)


def object_to_dict(obj):
    if isinstance(obj, list):
        return [object_to_dict(item) for item in obj]
    elif hasattr(obj, "__dict__"):
        return {key: object_to_dict(value) for key, value in obj.__dict__.items()}
    else:
        return obj

def parse_string_data(string):
    data_list = []
    responses = re.findall(r'"Response(\d+)":\s*\[(.*?)\]', string, re.DOTALL)
    for i, response in responses:
        items = re.findall(r'(\w+): (.*?)(?=,\n|$)', response)
        response_dict = {key: value.strip() for key, value in items}
        
        # Extract the "Question" key-value pair
        question_match = re.search(r'Question: (.*?)\n', response)
        if question_match:
            response_dict['Question'] = question_match.group(1).strip()
        data_list.append(response_dict)
    return data_list

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Process the PDF file
    documents = parser.load_data("temp.pdf")


    embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")
    llm = OpenAI(model="gpt-4o-mini", temperature=0)
    # create an index from the parsed markdown
    index = VectorStoreIndex.from_documents(documents, llm=llm, embed_model=embed_model)

    # create a query engine for the index
    query_engine = index.as_query_engine(response_mode="accumulate")

    query = """Give atleast 6 Market event and its Causal Relationship ( what leeds to what ) in the text and related question and answer pair of the causal relationship
Please order the response in the following order in JSON Format that I can easily put into a CSV file

{"Response":[Market Event: [Relevant Market Event to Causal Relationship],Causal Relationship: [Response],Question: [Related Question to Causal Relationship],Answer: [Answer to the above question]]}
"""


    response = query_engine.query(query)
    response = object_to_dict(response)
    df = pd.DataFrame(response.items()) 
    
    
    # String manipulation to extract the two responses
    response1 , response2 = df[1][0].split('---------------------')
    resp1 = json.loads(response1.strip("Response 1: "))["Response"]
    resp2 = json.loads(response2.replace("Response 2: ", ""))["Response"]
    
    # Commented out for now but used to test json conversion
    # st.write(resp1)
    # st.write(resp2)

    # Put resp1 and resp2 into a DataFrame
    df1 = pd.DataFrame(resp1)
    df2 = pd.DataFrame(resp2)

    # Concatenate the two DataFrames
    df = pd.concat([df1, df2], ignore_index=True)


    

    st.write(df)
    df.to_csv('output.csv', index=False)



    st.download_button(
        label="Download CSV",
        data=df.to_csv().encode('utf-8'),
        file_name='output.csv',
        mime='text/csv'
    )



