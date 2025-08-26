import pandas as pd
import streamlit as st
import os
import asyncio

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec


# Load environment variables for API keys
GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
PINECONE_API_KEY = os.environ['PINECONE_API_KEY']


@st.cache_resource
def validate_faq(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean a FAQ DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with only 'Question' and 'Answer' columns.
    """
    # Ensure column names are strings
    df.columns = df.columns.astype('str')

    q = False
    a = False

    for col in df.columns:
        # Standardize "Question" column naming
        if col.lower().strip() in ['question', 'pertanyaan', 'q', 'questions']:
            df = df.rename(columns={col:'Question'})
            q = True
            continue

        # Standardize "Answer" column naming
        if col.lower().strip() in ['answer','jawaban', 'a', 'answers']:
            df = df.rename(columns={col:'Answer'})
            a = True
            continue

        # Drop any other columns
        if col.lower().strip() not in ['question', 'answer']:
            df = df.drop(columns=col)
    
    # Raise error if required columns ("Question" and "Answer") are not available
    if not (q and a):
        raise KeyError('Missing required columns "Question" and "Answer"')
    
    # Drop any rows with missing values
    df = df.dropna(how='any')

    # Drop any rows with duplicated questions
    df = df.drop_duplicates(subset='Question', keep='last')

    # Ensure all values are strings
    df = df.astype('str')

    return df


@st.cache_resource
def load_faq(filepath: str) -> pd.DataFrame:
    """
    Load FAQ data from an Excel file.

    Args:
        filepath (str): Path to the Excel file.

    Returns:
        pd.DataFrame: Processed FAQ DataFrame.
    """

    # Read Excel file and change some values as NaN
    df = pd.read_excel(filepath, na_values=[' ', '-', 'missing'])
    
    # Clean and validate the DataFrame
    df = validate_faq(df)
    
    # Create a QnA column
    df['QnA'] = df['Question'] + " " + df['Answer']

    return df

@st.cache_resource
def insert_to_vectorstore(df: pd.DataFrame, index_name: str):
    """
    Insert FAQ data into a Pinecone vectorstore.

    Args:
        df (pd.DataFrame): FAQ DataFrame.
        index_name (str): Name of Pinecone index.

    Returns:
        PineconeVectorStore: Vectorstore instance with inserted FAQ embeddings.
    """

    # Ensure async event loop is set for GoogleGenerativeAIEmbeddings
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Initialize embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001",
                                              google_api_key=GOOGLE_API_KEY)
    
     # Convert dataframe index and QnA to list
    ids = df.index.astype('str').tolist()
    texts = df['QnA'].tolist()

    # Connect to Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Create Pinecone index if it doesnt exist
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    # Insert texts to Pinecone vectorstore
    vectorstore = PineconeVectorStore.from_texts(texts=texts,
                                                 ids=ids,
                                                 index_name=index_name,
                                                 embedding=embeddings)
    
    return vectorstore