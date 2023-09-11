from abc import ABC
from typing import Any, List, Optional

from langchain.vectorstores.base import VectorStore as LangchainVectorStore


class VectorStore(LangchainVectorStore, ABC):
    """Interface for vector store"""

    def delete(
        self,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError("delete method must be implemented")

    async def adelete(
        self,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError("adelete method must be implemented")
