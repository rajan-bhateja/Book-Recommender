from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain_chroma import Chroma
import pandas as pd

books = pd.read_csv(r'books_dataset_cleaned.csv')
print(books['tagged_description'])