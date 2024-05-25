from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from text_utils import split_documents, flatten_texts
from sentence_transformers import SentenceTransformer


def setup_database(embedding_model):
    pc = Pinecone(api_key="29d7d23a-569c-473e-bc4f-c5541f6d2b99")

    # Check if the index exists
    existing_indices = pc.list_indexes().names()
    index_name = "vectordb"
    if index_name not in existing_indices:
        # Create a new index
        pc.create_index(
            name=index_name,
            dimension=384,  # Adjust dimension as needed
            metric="cosine",  # Adjust metric as needed
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        index = pc.Index(index_name)
        # Upload data from PDFs
        upload_data_from_pdfs(embedding_model, index)
    else:
        # Use the existing index
        index = pc.Index(index_name)

    return index


def batch_data(data, batch_size=100):
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


def upload_data_from_pdfs(embedding_model, index):
    loader = DirectoryLoader(
        "<PDF Directory path>",  # Give pdfs file path
        glob="*.pdf",
        loader_cls=PyPDFLoader,
    )

    documents = loader.load()
    print("PDF files loaded")
    texts = split_documents(documents)
    flattened_texts = flatten_texts(texts)
    print("Started Embedding....")
    # Assuming embeddings are computed here
    embeddings = embedding_model.encode(flattened_texts)
    print("Embedding Done....")
    embedding_data = [
        {
            "id": f"doc_{i}",
            "values": embedding.tolist(),  # Convert numpy array to list
            "metadata": {"text": text},
        }
        for i, (text, embedding) in enumerate(zip(flattened_texts, embeddings))
    ]
    print("Data uploading the vectordb....")
    batch_size = 100  # Adjust batch size as needed
    for batch in batch_data(embedding_data, batch_size=batch_size):
        index.upsert(vectors=batch)

    print("Data upserted successfully!")
