import datetime
import uuid
from typing import Any, Dict

from pgvector.sqlalchemy import Vector  # type: ignore
from sqlalchemy import JSON, UUID, DateTime, ForeignKey, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Conversation(Base):
    __tablename__ = "conversations"
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.uuid_generate_v4(),
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime,
        default=datetime.datetime.utcnow,
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime,
        default=datetime.datetime.utcnow,
        nullable=False,
        onupdate=datetime.datetime.utcnow,
        server_default=func.now(),
    )

    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
    )


class Message(Base):
    __tablename__ = "messages"
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.uuid_generate_v4(),
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(
        nullable=False,
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    additional_kwargs: Mapped[Dict[Any, Any]] = mapped_column(
        JSON,
        nullable=True,
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime,
        default=datetime.datetime.utcnow,
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime,
        default=datetime.datetime.utcnow,
        nullable=False,
        onupdate=datetime.datetime.utcnow,
        server_default=func.now(),
    )

    conversation = relationship("Conversation", back_populates="messages")


class CollectionStore(Base):
    __tablename__ = "collections"
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.uuid_generate_v4(),
    )
    name: Mapped[str] = mapped_column(
        nullable=False,
    )
    cmetadata: Mapped[Dict[Any, Any]] = mapped_column(
        JSON,
        nullable=True,
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime,
        default=datetime.datetime.utcnow,
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime,
        default=datetime.datetime.utcnow,
        nullable=False,
        onupdate=datetime.datetime.utcnow,
        server_default=func.now(),
    )

    embeddings = relationship(
        "EmbeddingStore",
        back_populates="collection",
        cascade="all, delete-orphan",
    )


class EmbeddingStore(Base):
    __tablename__ = "embeddings"
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.uuid_generate_v4(),
    )
    collection_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("collections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    embedding: Mapped[Vector] = mapped_column(
        Vector,
        nullable=False,
    )
    document: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    cmetadata: Mapped[Dict[Any, Any]] = mapped_column(
        JSON,
        nullable=True,
    )
    custom_id: Mapped[str] = mapped_column(
        nullable=True,
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime,
        default=datetime.datetime.utcnow,
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime,
        default=datetime.datetime.utcnow,
        nullable=False,
        onupdate=datetime.datetime.utcnow,
        server_default=func.now(),
    )

    collection = relationship("CollectionStore", back_populates="embeddings")
