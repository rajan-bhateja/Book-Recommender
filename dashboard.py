import pandas as pd
import numpy as np

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from dotenv import load_dotenv

import streamlit as st

import os

load_dotenv()

books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
placeholder_url = "https://publications.iarc.fr/uploads/media/default/0001/02/thumb_1306_default_publication.jpeg"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),     # condition: no large thumbnails
    placeholder_url,                     # condition True
    books["large_thumbnail"]             # condition False
)

persist_directory = ".\chroma_db_books"
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

if os.path.exists(persist_directory):
    db_books = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
else:
    raw_documents = TextLoader("cleaned_description.txt", encoding="utf-8").load()
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)
    db_books = Chroma.from_documents(
        documents,
        embedding_model,
        persist_directory=persist_directory
    )

def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16
) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(final_top_k)

    if category!= "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs

def recommend_books(
        query: str,
        category: str = None,
        tone: str = None
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])} and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]


# Streamlit UI
st.set_page_config(page_title="Semantic Book Recommendation System", layout="wide")
st.title("Semantic Book Recommendation System")
st.caption("Powered by Langchain, Google Embeddings, Hugging Face, and ChromaDB")
st.write("Get book recommendations based on description, preferred categories and desired tones.")

# inputs
user_query = st.text_input("Please enter book description:", placeholder="E.g.: books about the Roman Empire")
col1, col2 = st.columns(2)
with col1:
    selected_category = st.selectbox("Select a Category:", options=categories, index=0)
with col2:
    selected_tone = st.selectbox("Select a Tone:", options=tones, index=0)

submit = st.button("Find Recommendations")

if submit:
    if not user_query.strip():
        st.warning("Please enter book description to continue")
    else:
        with st.spinner("Finding recommendations..."):
            results = recommend_books(user_query, selected_category, selected_tone)

            st.subheader("Recommendations:")

            if results:
                cols = st.columns(6)
                for i, (img_url, caption) in enumerate(results):
                    with cols[i%6]:
                        st.image(img_url, use_container_width=True, caption=caption)
            else:
                st.info("No recommendations found.")