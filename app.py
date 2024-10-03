import streamlit as st
import weaviate
import weaviate.classes as wvc
from weaviate.classes.query import Rerank, MetadataQuery, Filter
from weaviate.classes.aggregate import GroupByAggregate

wcd_url = st.secrets["weaviate"]["weaviate_url"]
wcd_api_key = st.secrets["weaviate"]["weaviate_api"]
cohere_api_key = st.secrets["cohere"]["cohere_prod"]

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=wcd_url,
    auth_credentials=wvc.init.Auth.api_key(wcd_api_key),
    headers={"X-Cohere-Api-Key": cohere_api_key},
    skip_init_checks=True,
)

st.sidebar.title(":books: ScholarQuery")
st.sidebar.subheader("🔍 Uncover Meaningful Insights with GenAI Semantic Search")

response = client.collections.list_all(simple=False)
classes = sorted(list(response.keys()))

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

search_clicked = st.sidebar.button("Search")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        ":mag_right: Search",
        ":desktop_computer: Tech",
        ":book: Concepts",
        ":information_source: About",
    ]
)


def render_progress_bar(container, value):
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
    render_progress_bar(left, relevance)

    if rerank_choice and rerank_score:
        left.write(f"**Reranked Relevance:** {round(reranked_relevance, 1)}%")
        render_progress_bar(left, reranked_relevance)

    left.write(f"**Topic:** {topic}")
    left.write(f"**Book:** {book}")
    left.write(f"**Author:** {author}")
    left.write(f"**Page:** {page}")
    left.write(f"**Volume:** {volume}")

    right.write(f"**Text:** {text}")


def run_search():
    if search_mode == "Explained Search":
        prompt = """Given following text: {text}, perform these tasks:
        
        - Provide summarized translation to English, it needs to be a direct summarized translation.
        - Explain why it is relevant to the query.
        - Answer back in English.
        - Strictly follow the format below:
        
        **Summarized Translation**:
        **Relevance**:
        """
        response = book_collection.generate.near_text(
            query=query,
            limit=top_k,
            single_prompt=prompt,
            filters=final_filters,
            rerank=rerank,
            return_metadata=MetadataQuery(distance=True, score=True),
        )
        st.header("Semantic Search Results")

        if response.objects:
            for i, result in enumerate(response.objects):
                display_result(st, result, i, rerank_choice)
                st.write(f"{result.generated}")
        else:
            st.write("No results found.")

    elif search_mode == "Summary Generation Search":
        task_prompt = """Provide a short summary in English based on the query results. The summary should represent the content of combined results. Address the query even if it is not a proper question.
        """
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
        if response.generated:
            st.write(f"{response.generated}")

        st.subheader("Semantic Search Results")
        if response.objects:
            for i, result in enumerate(response.objects):
                display_result(st, result, i, rerank_choice)
        else:
            st.write("No results found.")

    else:
        response = book_collection.query.near_text(
            query=query,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True, score=True),
            rerank=rerank,
            filters=final_filters,
        )
        st.header("Semantic Search Results")
        if response.objects:
            for i, result in enumerate(response.objects):
                display_result(st, result, i, rerank_choice)
        else:
            st.write("No results found.")


with tab1:
    if not search_clicked:
        st.title("📚 Welcome to ScholarQuery!")
        st.subheader("🔍 Uncover Meaningful Insights with GenAI Semantic Search")
        st.markdown(
            """
                📚**ScholarQuery** allows you to semantically search through large volumes of texts, providing you with **accurate** and **relevant** results. Our goal is to enable researchers to search through complex textual corpora with ease.
                
                Here are a few things you can do:
                
                - **Semantic Search**: Find content relevant to your query.
                - **Explained Search**: Get a summarized translation and explanation of why the result matches your query.
                - **Summary Generation Search**: Generate summaries based on your query and results.
                
                **How to start:**
                1. Select a **book** and **topic** from the options on the left.
                2. Enter your **query** and choose your preferred **search mode**.
                3. Select the number of **top results** to return.
                4. Optionally, enable **Rerank** and provide a rerank query.
                5. Click the **Search**.

                Let's get started! 🎯
                """
        )
    else:
        run_search()

with tab2:
    st.title(":desktop_computer: Tech")
    st.markdown(
        """
            ScholarQuery is powered by cutting-edge technologies and frameworks that allow seamless semantic search and efficient processing of large textual datasets. Below is a detailed breakdown of the technologies in use:
                            
            - **Weaviate**: A cloud-native, open-source, real-time vector database. [Learn more](https://weaviate.io/).
            - **Cohere**: Powerful GenAI Chat and Embedding models. [Learn more](https://cohere.com/).
            - **Streamlit**: An open-source app framework for Machine Learning and Data Science projects. [Learn more](https://streamlit.io/).

        """
    )

with tab3:
    st.title(":book: Concepts")
    st.markdown(
        """
            ScholarQuery leverages several advanced concepts to power its semantic search and retrieval capabilities. Below are key concepts that are foundational to understanding how ScholarQuery works:

            - **Embeddings**: Embeddings are a way to represent the meaning of text as a list of numbers. Using a simple comparison function, we can then calculate a similarity score for two embeddings to figure out whether two texts are talking about similar things. Common use-cases for embeddings include semantic search, clustering, and classification. [Learn more](https://docs.cohere.com/v2/docs/embeddings).
            - **Semantic Search**: Semantic search is a data retrieval method where results are determined based on their contextual and semantic similarity to the query. [Learn more](https://cohere.com/llmu/what-is-semantic-search).
            - **Rerankers**: Rerankers are models that generate a new ranking of search results based on the relevance to a rerank query. The relevance is determined by the reranking LLM. [Learn more](https://cohere.com/llmu/reranking).
            - **Synthetic Topic Generation**: We provide the topics provided by clustering the embeddings space and using the top 10 cluster representatives to generate synthetic topics.
            - **Explained Search**: We use Retrieval Augmented Generation (RAG) to generate a summarized translation and explanation of relevance to the query using Weaviate's Generative Search. [Learn more](https://weaviate.io/developers/weaviate/search/generative).
            - **Summary Generation Search**: We use Weaviate's Grouped Task Search to generate a summary of the combined search results based on the query. [Learn more](https://weaviate.io/developers/weaviate/search/generative#grouped-task-search).
            - **Prompt Engineering**: We use prompt engineering for many different tasks, from generating synthetic queries for retrieval evaluation, to generating synthetic topics for clustering. [Learn more](https://cohere.com/llmu/constructing-prompts). 
        """
    )

with tab4:
    st.title(":information_source: About")
    st.markdown(
        """
            - ScholarQuery is a project developed by [Amin Tehaucha](https://www.linkedin.com/in/amintehaucha/) as part of the final project for the B.Sc. in Information Systems (Data Science).

            - This project was developed under the supervision and guidance of [Dr. Loai Abdallah](https://www.linkedin.com/in/loaia/).

            - This project is demonstrated by using the [OpenITI](https://openiti.org/) collection of digitized Islamic, Arabic and Persian texts.

            - The project GitHub repository, which contains all material for the final project, can be found [here](https://github.com/charphob/scholar-query).

        """
)
