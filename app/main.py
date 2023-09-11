import uuid
from contextlib import asynccontextmanager

import openai
from fastapi import Depends, FastAPI
from fastapi.responses import StreamingResponse
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import Document
from langchain.schema.runnable import RunnableMap
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.database import async_session, engine, get_session, initialize_database
from app.langchain_ext.indexes.index import index_async
from app.langchain_ext.indexes.record_manager import SQLRecordManagerAsync
from app.langchain_ext.memory.chat_memory import ConversationBufferMemoryAsync
from app.langchain_ext.memory.chat_message_history import (
    PostgresChatMessageHistoryAsync,
)
from app.langchain_ext.vectorstores.pgvector import PGVectorAsync


@asynccontextmanager
async def lifespan(app: FastAPI):
    await initialize_database()

    record_manager = SQLRecordManagerAsync(
        namespace="testing",
        engine=engine,
    )

    await record_manager.create_schema()

    embeddings = OpenAIEmbeddings(
        openai_api_key=settings.OPENAI_API_KEY,
        client=openai,
    )

    async with async_session() as session:
        vectorstore = await PGVectorAsync.afrom_existing_index(
            session=session,
            embedding=embeddings,
        )

        docs = [
            Document(
                page_content="The dog is happy.",
                metadata={"status": "happy", "isExample": False, "source_id": "1"},
            ),
            Document(
                page_content="The cat is sad.",
                metadata={"status": "sad", "isExample": False, "source_id": "2"},
            ),
            Document(
                page_content="I'm really sad.",
                metadata={"status": "sad", "isExample": True, "source_id": "3"},
            ),
        ]

        result = await index_async(
            docs_source=docs,
            record_manager=record_manager,
            vector_store=vectorstore,
            cleanup="incremental",
            source_id_key="source_id",
        )

        print(result)

    yield


settings = get_settings()

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    """A request to create a new conversation."""

    conversation_id: uuid.UUID
    message: str


@app.post("/chat")
async def chat(
    request: ChatRequest,
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_session),
):
    chat_memory = await PostgresChatMessageHistoryAsync.create(
        conversation_id=request.conversation_id, session=session
    )

    memory = ConversationBufferMemoryAsync(
        chat_memory=chat_memory,
        return_messages=True,
        memory_key="chat_history",
    )

    model = ChatOpenAI(temperature=0.9, openai_api_key=settings.OPENAI_API_KEY)
    prompt = ChatPromptTemplate.from_messages(  # type: ignore
        messages=[
            (
                "system",
                "You are a cat. You always answer questions with 'meow' and in a cat-like way. Let's have fun.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    chain = (
        RunnableMap(
            {
                "input": lambda x: x["input"],
                "memory": memory.load_memory_variables,
            }
        )
        | {
            "input": lambda x: x["input"],
            "chat_history": lambda x: x["memory"]["chat_history"],
        }
        | prompt
        | model
    )

    async def generate_response():
        input = {"input": request.message}

        response = ""
        async for token in chain.astream(input=input):
            yield token.content
            response += token.content
        await memory.save_context(input, {"output": response})

    return StreamingResponse(generate_response(), media_type="text/plain")


@app.post("/vector")
async def vector(
    settings: Settings = Depends(get_settings),
    session: AsyncSession = Depends(get_session),
):
    embeddings = OpenAIEmbeddings(
        openai_api_key=settings.OPENAI_API_KEY,
        client=openai,
    )

    vectorstore = await PGVectorAsync.afrom_existing_index(
        session=session,
        embedding=embeddings,
    )

    return await vectorstore.asimilarity_search(
        query="I like happy cats.",
        filter={
            "status": {"in": ["sad"]},
            "isExample": True,
        },
    )
