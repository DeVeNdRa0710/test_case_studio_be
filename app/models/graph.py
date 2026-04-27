from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class NodeLabel(str, Enum):
    MODULE = "Module"
    ENTITY = "Entity"
    API = "API"
    UI_SCREEN = "UIScreen"


class RelType(str, Enum):
    CREATES = "CREATES"
    TRIGGERS = "TRIGGERS"
    GENERATES = "GENERATES"
    UPDATES = "UPDATES"
    DEPENDS_ON = "DEPENDS_ON"
    CALLS_API = "CALLS_API"


class GraphNode(BaseModel):
    label: NodeLabel
    name: str
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphRelationship(BaseModel):
    type: RelType
    from_label: NodeLabel
    from_name: str
    to_label: NodeLabel
    to_name: str
    condition: str | None = None
    action: str | None = None
    trigger_point: str | None = None
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphExtraction(BaseModel):
    nodes: list[GraphNode] = Field(default_factory=list)
    relationships: list[GraphRelationship] = Field(default_factory=list)
