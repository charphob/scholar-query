import streamlit as st
import weaviate
import weaviate.classes as wvc
from weaviate.classes.query import Rerank, MetadataQuery, Filter
from weaviate.classes.aggregate import GroupByAggregate

import os
from dotenv import load_dotenv

load_dotenv()

# Load environment variables for Weaviate and Cohere API keys
wcd_url = st.secrets['weaviate']['weaviate_url']
wcd_api_key = st.secrets['weaviate']['weaviate_api']
cohere_api_key = st.secrets['cohere']['cohere_prod']

# Connect to Weaviate
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=wcd_url,
    auth_credentials=wvc.init.Auth.api_key(wcd_api_key),
    headers={"X-Cohere-Api-Key": cohere_api_key},
    skip_init_checks=True,
)

# Use the sidebar for controls
st.sidebar.title(":books: ScholarQuery")
st.sidebar.subheader("Find meaningful insights with GenAI Semantic Search")

# Retrieve available collections from Weaviate
response = client.collections.list_all(simple=False)
classes = sorted(list(response.keys()))

# Input fields and controls in the sidebar
selected_collection = st.sidebar.selectbox(
    "Which book selection?",
    options=classes,
    placeholder="Select a book collection to search in",
    help="Select one of the available book collections to search in.",
)

book_collection = client.collections.get(selected_collection)

book_response = book_collection.aggregate.over_all(
    group_by=GroupByAggregate(
        prop="book",
    )
)

list_of_books = [group.grouped_by.value for group in book_response.groups]

selected_book = st.sidebar.selectbox(
    "Which book?",
    options=list_of_books,
    placeholder="Select a book to search in",
    help="You can search over a specific book. Search over all books is not supported.",
)

topic_response = book_collection.aggregate.over_all(
    group_by=GroupByAggregate(
        prop="topic",
    )
)

list_of_topics = [group.grouped_by.value for group in topic_response.groups]

selected_topics = st.sidebar.multiselect(
    "Which topics?",
    options=list_of_topics,
    help="ScholarQuery offers LLM generated topic clusters for the books. You can select one or more topics to search in.",
)

selected_filters = []

if selected_book:
    selected_filters.append(Filter.by_property("book").equal(selected_book))

if selected_topics:
    selected_filters.append(Filter.by_property("topic").contains_any(selected_topics))

if selected_filters:
    final_filters = Filter.all_of(selected_filters)
else:
    final_filters = None

search_mode = st.sidebar.radio(
    "Select search mode:",
    captions=[
        "Provides Semantic Search using Cosine similarity and ANN (HNSW)",
        "Provides summarized translation and explanation of relevance to the query",
        "Generates a summary of the search results based on the query",
    ],
    options=["Semantic Search", "Explained Search", "Summary Generation Search"],
    help="Select the search mode to use for the query.",
)

top_k = st.sidebar.slider(
    "Select number of ranked results",
    min_value=1,
    max_value=10,
    value=5,
    help="Select the number of top results to return for the query. Limited to maximum of 10 results.",
)

query = st.sidebar.text_input(
    "Enter your query",
    help="Enter the query to semantically search for in the selected collection. Query can be a natural language question or any other text.",
)

rerank_choice = st.sidebar.toggle("Use Rerank!", False)
rerank = None

if rerank_choice:
    rerank_query = st.sidebar.text_input(
        "Enter rerank query",
        help="A reranker generates a new ranking of the search results based on the relevance to the rerank query. The relevance is determined by the reranking LLM",
        placeholder="Enter the query to rerank the results based on the relevance to the query.",
    )

    if rerank_query:
        rerank = Rerank(
            prop="text",
            query=rerank_query,
        )

# Search button in the sidebar
search_clicked = st.sidebar.button("Search")


def render_progress_bar(container, value, inverse=False):
    l, c, r = container.columns([0.1, 0.8, 0.1])
    l.write(":no_entry_sign:")
    c.progress(int(value))
    r.write(":dart:")


def display_result(container, result, i, rerank_choice):

    text = result.properties["text"]
    topic = result.properties["topic"]
    book = result.properties["book"]
    author = result.properties["author"]
    page = int(result.properties["page"][1:])
    volume = int(result.properties["volume"][1:])
    distance = result.metadata.distance
    rerank_score = result.metadata.rerank_score

    relevance = (1 - (distance / 2)) * 100  # Adjust for 0 to 2 range

    reranked_relevance = rerank_score * 100 if rerank_score is not None else None

    st.subheader(f"Result {i + 1}")
    left, right = container.columns([0.3, 0.7], gap="large")
    left.write(f"**Relevance:** {round(relevance, 1)}%")
    render_progress_bar(left, relevance, inverse=True)

    if rerank_choice and rerank_score:
        left.write(f"**Reranked Relevance:** {round(reranked_relevance, 1)}%")
        render_progress_bar(left, reranked_relevance, inverse=False)

    left.write(f"**Topic:** {topic}")
    left.write(f"**Book:** {book}")
    left.write(f"**Author:** {author}")
    left.write(f"**Page:** {page}")
    left.write(f"**Volume:** {volume}")

    right.write(f"**Text:** {text}")


# Helper function to handle and display results based on search mode
def run_search():
    if search_mode == "Explained Search":
        prompt = '''Given following text: {text}, perform these tasks:
        
        - Provide summarized translation to English, it needs to be a direct summarized translation.
        - Explain why it is relevant to the query.
        - Answer back in English.
        - Strictly follow the format below:
        
        **Summarized Translation**:
        **Relevance**:
        '''
        response = book_collection.generate.near_text(
            query=query,
            limit=top_k,
            single_prompt=prompt,
            filters=final_filters,
            rerank=rerank,
            return_metadata=MetadataQuery(distance=True, score=True),
        )
        st.header("Semantic Search Results")

        # Display results (original + explanation)
        if response.objects:
            for i, result in enumerate(response.objects):
                display_result(st,result, i, rerank_choice)
                st.write(f"{result.generated}")
        else:
            st.write("No results found.")

    elif search_mode == "Summary Generation Search":
        task_prompt = '''Provide a short summary in English based on the query results. The summary should represent the content of combined results. Address the query even if it is not a proper question.
        '''
        response = book_collection.generate.near_text(
            query=query,
            limit=top_k,
            grouped_task=task_prompt,
            grouped_properties=["text"],
            filters=final_filters,
            rerank=rerank,
            return_metadata=MetadataQuery(distance=True, score=True),
        )

        st.header("Generated Summary")
        # Display the grouped summary first
        if response.generated:
            st.write(f"{response.generated}")
        
        st.subheader("Semantic Search Results")
        # Then show individual results
        if response.objects:
            for i, result in enumerate(response.objects):
                display_result(st,result, i, rerank_choice)
        else:
            st.write("No results found.")

    else:  # Semantic Search
        response = book_collection.query.near_text(
            query=query,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True, score=True),
            rerank=rerank,
            filters=final_filters,
        )
        st.header("Semantic Search Results")
        # Display results (standard semantic search)
        if response.objects:
            for i, result in enumerate(response.objects):
                display_result(st,result, i, rerank_choice)
        else:
            st.write("No results found.")


# Execute search on button click
if search_clicked:
    run_search()