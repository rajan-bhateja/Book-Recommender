from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

books = pd.read_csv('books_dataset_cleaned.csv')
print(books['tagged_description'])