import asyncio

from bs4 import BeautifulSoup
from langchain.document_loaders import RecursiveUrlLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.database import async_session, engine
from app.langchain_ext.indexes.index import index_async
from app.langchain_ext.indexes.record_manager import SQLRecordManagerAsync
from app.langchain_ext.vectorstores.pgvector import PGVectorAsync

settings = get_settings()


def get_langchain_docs() -> list[Document]:
    urls = [
        # "https://api.python.langchain.com/en/latest/api_reference.html#module-langchain",
        "https://python.langchain.com/docs/get_started",
        # "https://python.langchain.com/docs/use_cases",
        # "https://python.langchain.com/docs/integrations",
        # "https://python.langchain.com/docs/modules",
        # "https://python.langchain.com/docs/guides",
        # "https://python.langchain.com/docs/ecosystem",
        # "https://python.langchain.com/docs/additional_resources",
        # "https://python.langchain.com/docs/community",
    ]

    documents: list[Document] = []

    for url in urls:
        loader = RecursiveUrlLoader(
            url=url,
            max_depth=8,
            extractor=lambda x: BeautifulSoup(x, "lxml").text,
            prevent_outside=True,
        )

        temp_docs = loader.load()
        temp_docs = [doc for i, doc in enumerate(temp_docs) if doc not in temp_docs[:i]]
        documents.extend(temp_docs)

    html2text = Html2TextTransformer()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)

    docs_transformed = html2text.transform_documents(documents)
    docs_transformed = text_splitter.split_documents(docs_transformed)

    for doc in docs_transformed:
        if "source" not in doc.metadata:
            doc.metadata["source"] = ""
            print("No source found")
        if "title" not in doc.metadata:
            doc.metadata["title"] = ""
            print("No title found")

    return docs_transformed


async def main():
    docs = get_langchain_docs()

    record_manager = SQLRecordManagerAsync(
        namespace="langchain_docs",
        engine=engine,
    )

    await record_manager.create_schema()

    embeddings = OpenAIEmbeddings(  # type: ignore
        openai_api_key=settings.OPENAI_API_KEY,
    )

    async with async_session() as session:
        await PGVectorAsync.create(
            session=session,
            embedding_function=embeddings,
            collection_name="langchain_docs",
        )

        vectorstore = await PGVectorAsync.afrom_existing_index(
            session=session,
            embedding=embeddings,
            collection_name="langchain_docs",
        )

        result = await index_async(
            docs_source=docs,
            record_manager=record_manager,
            vector_store=vectorstore,
            cleanup="full",
            source_id_key="source",
        )

    print(result)


if __name__ == "__main__":
    asyncio.run(main())
