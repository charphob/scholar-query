# :books: ScholarQuery

Welcome to **ScholarQuery** – a powerful semantic search and retrieval system built using state-of-the-art technologies for exploring large volumes of text and extracting meaningful insights. This repository contains all the resources and notebooks used to build the system, including the Streamlit app, various notebooks for data preprocessing, and exploration.

## :rocket: Overview

ScholarQuery is designed to help researchers easily search, rank, and extract relevant text from large datasets of complex and rich text. It uses embeddings, vector search, and machine learning models to provide accurate results based on the meaning of the query, not just keyword matches.

Key features include:
- **Semantic Search**: Find results based on meaning, not just keywords.
- **Summarization**: Get summaries and translations of relevant text.
- **RAG (Retrieval-Augmented Generation)**: Leverage advanced LLM capabilities to generate responses.
- **Topic Filtering**: Filter results by topics generated using embedding clustering.

## :wrench: Tech Stack

This project leverages a wide range of tools and frameworks to achieve its goals. Here's an overview of the tech stack:
- **Weaviate**: A vector database and search engine.
- **Cohere**: Embedding and LLM generation models to process and analyze the text.
- **Anthropic**: Used to generate synthetic queries for evaluation.
- **Streamlit**: The app's frontend, providing an interactive interface for users to input queries and view results.
- **Annoy**: Spotify's Approximate Nearest Neighbors library for fast vector search. Used for evaluation and testing.

## :file_folder: Repository Structure

- **app.py**: The main Streamlit app file.
- **notebooks**: Contains Jupyter notebooks for data preprocessing, embeddings generation, clustering, and exploration.
  - **clustering.ipynb**: KMeans clustering and topic generation.
  - **data.ipynb**: Data preprocessing and cleaning.
  - **dbtest.ipynb**: Testing Weaviate database connection and features.
  - **eval.ipynb**: Evaluation of the search system using synthetic queries.
  - **kitab-data.ipynb**: Early stage data exploration and analysis.
  - **vdb.ipynb**: Vector database setup and testing.

## :computer: Streamlit App

The ScholarQuery app provides an interactive interface for users to explore the dataset using semantic search, summarization, and topic-based filtering.

:point_right: [ScholarQuery](https://scholarquery.streamlit.app)

## :link: Useful Links

- [Weaviate Documentation](https://weaviate.io/developers/weaviate)
- [Cohere Documentation](https://docs.cohere.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)