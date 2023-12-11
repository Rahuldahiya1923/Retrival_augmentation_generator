#!/usr/bin/env python
# coding: utf-8

# In[1]:


# vector DB
import os
from getpass import getpass
import kdbai_client as kdbai
import time


# In[2]:


# langchain packages
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import KDBAI
from langchain import HuggingFaceHub
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain


# In[3]:


os.environ["OPENAI_API_KEY"] = (
    os.environ["OPENAI_API_KEY"]
    if "OPENAI_API_KEY" in os.environ
    else getpass("OpenAI API Key: ")
)


# In[4]:


os.environ["HUGGINGFACEHUB_API_TOKEN"] = (
    os.environ["HUGGINGFACEHUB_API_TOKEN"]
    if "HUGGINGFACEHUB_API_TOKEN" in os.environ
    else getpass("Hugging Face API Token: ")
)


# In[5]:


# Load the documents we want to prompt an LLM about
doc = TextLoader("data/state_of_the_union.txt").load()


# In[6]:


# Chunk the documents into 500 character chunks using langchain's text splitter "RucursiveCharacterTextSplitter"
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)


# In[7]:


# split_documents produces a list of all the chunks created, printing out first chunk for example
pages = [p.page_content for p in text_splitter.split_documents(doc)]


# In[8]:


pages[0]


# In[9]:


embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")


# In[10]:


KDBAI_ENDPOINT = (
    os.environ["KDBAI_ENDPOINT"]
    if "KDBAI_ENDPOINT" in os.environ
    else input("KDB.AI endpoint: ")
)
KDBAI_API_KEY = (
    os.environ["KDBAI_API_KEY"]
    if "KDBAI_API_KEY" in os.environ
    else getpass("KDB.AI API key: ")
)


# In[11]:


session = kdbai.Session(api_key=KDBAI_API_KEY, endpoint=KDBAI_ENDPOINT)


# In[ ]:


session = kdbai.Session(endpoint='http://localhost:8082')


# ### Define Vector DB Table Schema

# In[12]:


rag_schema = {
    "columns": [
        {"name": "id", "pytype": "str"},
        {"name": "text", "pytype": "bytes"},
        {
            "name": "embeddings",
            "pytype": "float32",
            "vectorIndex": {"dims": 1536, "metric": "L2", "type": "flat"},
        },
    ]
}


# In[13]:


# First ensure the table does not already exist
try:
    session.table("rag_langchain").drop()
    time.sleep(5)
except kdbai.KDBAIException:
    pass


# In[14]:


table = session.create_table("rag_langchain", rag_schema)


# In[15]:


# use KDBAI as vector store
vecdb_kdbai = KDBAI(table, embeddings)
vecdb_kdbai.add_texts(texts=pages)


# Now we have the vector embeddings stored in KDB.AI we are ready to query.

# In[16]:


query = "what are the nations strengths?"


# In[17]:


# query_sim holds results of the similarity search, the closest related chunks to the query.
query_sim = vecdb_kdbai.similarity_search(query)


# In[18]:


query_sim


# In[19]:


# select two llm models (OpenAI text-davinci-003, HuggingFaceHub google/flan-t5-xxl(designed for short answers))
llm_openai = OpenAI(model="text-davinci-003", max_tokens=512)
llm_flan = HuggingFaceHub(
    repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512}
)


# In[20]:


# create the chain for each model using langchain load_qa_chain
chain_openAI = load_qa_chain(llm_openai, chain_type="stuff")
chain_HuggingFaceHub = load_qa_chain(llm_flan, chain_type="stuff")


# In[21]:


# Show the most related chunks to the query
query_sim


# In[22]:


# OpenAI - run the chain on the query and the related chunks from the documentation
chain_openAI.run(input_documents=query_sim, question=query)


# In[23]:


# HugginFace - run the chain on the query and the related chunks from the documentation
chain_HuggingFaceHub.run(input_documents=query_sim, question=query)


# We can see the response from OpenAI is longer and more detailed and seems to have done a better job summarizing the nation's strengths from the document provided.

# In[24]:


K = 10


# In[25]:


qabot = RetrievalQA.from_chain_type(
    chain_type="stuff",
    llm=ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.0),
    retriever=vecdb_kdbai.as_retriever(search_kwargs=dict(k=K)),
    return_source_documents=True,
)


# In[26]:


print(query)
print("-----")
print(qabot(dict(query=query))["result"])


# In[27]:


def query_qabot(qabot, query: str):
    print(new_query)
    print("---")
    return qabot(dict(query=new_query))["result"]


# In[28]:


new_query = "what are the things this country needs to protect?"
query_qabot(qabot, new_query)


# In[29]:


table.drop()

