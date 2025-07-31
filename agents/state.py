from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from operator import add as add_messages

class StateAgent(TypedDict):
    message: Annotated[Sequence[BaseMessage], add_messages]
    task: str  # 'generate', 'explain', or 'fallback'
    classification: str
