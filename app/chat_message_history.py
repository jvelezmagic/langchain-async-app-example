from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import List

from langchain.schema import messages_from_dict  # type: ignore
from langchain.schema.messages import AIMessage, BaseMessage, HumanMessage
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Conversation, Message


class BaseChatMessageHistoryAsync(ABC):
    """Abstract base class for storing chat message history."""

    @property
    @abstractmethod
    async def messages(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve the messages from the store."""
        raise NotImplementedError()

    async def add_user_message(self, message: str) -> None:
        """Convenience method for adding a human message string to the store.

        Args:
            message: The string contents of a human message.
        """
        await self.add_message(HumanMessage(content=message))

    async def add_ai_message(self, message: str) -> None:
        """Convenience method for adding an AI message string to the store.

        Args:
            message: The string contents of an AI message.
        """
        await self.add_message(AIMessage(content=message))

    @abstractmethod
    async def add_message(self, message: BaseMessage) -> None:
        """Add a Message object to the store.

        Args:
            message: A BaseMessage object to store.
        """
        raise NotImplementedError()

    @abstractmethod
    async def clear(self) -> None:
        """Clear session memory from the store."""
        raise NotImplementedError()


class PostgresChatMessageHistoryAsync(BaseChatMessageHistoryAsync):
    """Chat message history stored in a Postgres database."""

    def __init__(
        self,
        conversation_id: uuid.UUID,
        session: AsyncSession,
    ):
        self.conversation_id = conversation_id
        self.session = session

    @classmethod
    async def create(
        cls,
        conversation_id: uuid.UUID,
        session: AsyncSession,
    ) -> PostgresChatMessageHistoryAsync:
        """Create a new instance of the class.

        Args:
            conversation_id: The ID of the conversation to retrieve.
            session: The SQLAlchemy session to use for database interactions.

        Returns:
            A new instance of the class.
        """

        # Create conversation if it doesn't exist
        query = select(Conversation).where(Conversation.id == conversation_id)
        result = await session.execute(query)
        if not result.scalars().first():
            new_conversation = Conversation(id=conversation_id)
            session.add(new_conversation)
            await session.commit()

        return cls(conversation_id=conversation_id, session=session)

    @property
    async def messages(self) -> List[BaseMessage]:
        """Retrieve the messages from PostgreSQL"""
        query = select(Message).where(Message.conversation_id == self.conversation_id)
        result = await self.session.execute(query)
        items = [
            {
                "type": item.role,
                "data": {
                    "content": item.content,
                    "additional_kwargs": item.additional_kwargs,
                    "example": False,
                },
            }
            for item in result.scalars().all()
        ]
        messages = messages_from_dict(items)
        return messages

    async def add_user_message(self, message: str) -> None:
        """Convenience method for adding a human message string to the store."""
        return await super().add_user_message(message)

    async def add_ai_message(self, message: str) -> None:
        """Convenience method for adding an AI message string to the store."""
        return await super().add_ai_message(message)

    async def add_message(self, message: BaseMessage) -> None:
        """Append the message to the record in PostgreSQL"""
        new_message = Message(
            conversation_id=self.conversation_id,
            role=message.type,
            content=message.content,
            additional_kwargs=message.additional_kwargs,  # type: ignore
        )
        self.session.add(new_message)
        await self.session.commit()

    async def clear(self) -> None:
        """Clear session memory from PostgreSQL"""
        query = delete(Message).where(Message.conversation_id == self.conversation_id)
        await self.session.execute(query)
        await self.session.commit()
