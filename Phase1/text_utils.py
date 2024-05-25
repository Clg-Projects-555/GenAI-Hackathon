from langchain.text_splitter import RecursiveCharacterTextSplitter


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)


def flatten_texts(texts):
    return [doc.page_content for doc in texts]
