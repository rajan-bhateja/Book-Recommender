# 📚 Semantic Book Recommendation System

This project is an **AI-powered book recommender** that uses **semantic similarity** to suggest books matching a user’s query, preferred category, and desired emotional tone.

It leverages:
- **LangChain** for text embeddings and semantic search
- **Google Generative AI Embeddings**
- **Chroma** as a local vector database
- **Streamlit** for an interactive web dashboard

---

## 🚀 Features

✅ **Semantic search** — recommend books based on meaning, not just keywords  
✅ **Emotion filtering** — rank books by tones like Happy, Sad, Suspenseful, etc.  
✅ **Category selection** — narrow down results by genre  
✅ **Fast performance** — vector embeddings stored locally for instant reuse  
✅ **Beautiful web UI** — easy to run and share

---

## 🗂️ Project Structure

```plaintext
├── dashboard.py            # Main Streamlit app
├── books_with_emotions.csv # Book data with precomputed emotion scores
├── cleaned_description.txt # Text corpus for embeddings
├── chroma_db_books/        # Local vector database (auto-created)
├── .env                    # API keys and environment variables
