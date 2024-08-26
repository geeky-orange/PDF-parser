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


# Assume this function processes a PDF and returns a DataFrame
def process_pdf(file_path):
    # Your LLM model processing code here
    # For demonstration, we'll just create a dummy DataFrame
    data = {'Column1': [1, 2, 3], 'Column2': ['A', 'B', 'C']}
    df = pd.DataFrame(data)
    return df

# Streamlit app
st.title("PDF Parser")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
parser = LlamaParse(result_type=ResultType.MD,language=Language.ENGLISH)
# parser = LlamaParse(result_type="markdown")

def object_to_dict(obj):
    if isinstance(obj, list):
        return [object_to_dict(item) for item in obj]
    elif hasattr(obj, "__dict__"):
        return {key: object_to_dict(value) for key, value in obj.__dict__.items()}
    else:
        return obj

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Process the PDF file
    documents = parser.load_data("temp.pdf")

    # use SimpleDirectoryReader to parse our file
    # file_extractor = {".pdf": parser}
    # documents = SimpleDirectoryReader(input_files=['temp.pdf'], file_extractor=file_extractor).load_data()
    # print(documents)

    # # Display parsed content
    # st.write("Parsed Content:")
    # if documents:
    #     for doc in documents:
    #         st.markdown(doc)
    # else:
    #     st.write("No content found")


    embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")
    llm = OpenAI(model="gpt-4o-mini", temperature=0)
    # create an index from the parsed markdown
    index = VectorStoreIndex.from_documents(documents, llm=llm, embed_model=embed_model)

    # create a query engine for the index
    query_engine = index.as_query_engine(response_mode="accumulate")

    query = """Give atleast 6 Market event and its Causal Relationship ( what leeds to what ) in the text and related question and answer pair of the causal relationship
Please order the response in the following order in JSON Format

{
[
Market Event: [Relevant Market Event to Causal Relationship],
Causal Relationship: [Response],
Question: [Related Question to Causal Relationship],
Answer: [Answer to the above question],
],

6 More Later
}

"""
    response = query_engine.query(query)
    
    print(response)
    print(type(response))
    response = object_to_dict(response)
    print(type(response))

    print(response)

    df = pd.DataFrame(response.items())  
    st.write(df)
    
    df.to_csv('output.csv', index=False)

    st.download_button(
        label="Download CSV",
        data=df.to_csv().encode('utf-8'),
        file_name='output.csv',
        mime='text/csv'
    )

    


    # print(response)

    # print(type(response))

    # response = json.loads(response)

    # df = pd.DataFrame(response)
    # st.write(df)


