from langchain.text_splitter import RecursiveCharacterTextSplitter


def create_documents(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=100,
        length_function=len,
    )
    return text_splitter.create_documents([text])
