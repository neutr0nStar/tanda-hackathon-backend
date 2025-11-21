import os
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

from dotenv import load_dotenv
from langchain.messages import AIMessage, HumanMessage
from langchain_anthropic import ChatAnthropic

load_dotenv()


DEFAULT_MODEL = "claude-sonnet-4-5"


def create_llm(api_key: Optional[str] = None, model: str = DEFAULT_MODEL) -> ChatAnthropic:
    """Instantiate the Anthropic chat model with a helpful error if the key is missing."""

    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("Missing ANTHROPIC_API_KEY in environment or .env")

    return ChatAnthropic(model=model, api_key=key)  # type: ignore[arg-type]


MessageRole = Literal["user", "assistant"]


@dataclass
class Message:
    role: MessageRole
    content: str


@dataclass
class ChatNode:
    node_id: str
    parent_id: Optional[str]
    messages: List[Message] = field(default_factory=list)


class ConversationGraph:
    """In-memory conversation graph that supports branching and merging of chat history."""

    def __init__(self) -> None:
        self.nodes: Dict[str, ChatNode] = {}

    def create_root(self, node_id: str = "ROOT") -> str:
        """Create the root node for a conversation tree."""

        node = ChatNode(node_id=node_id, parent_id=None)
        self.nodes[node_id] = node
        return node_id

    def branch_from(self, node_id: str, new_id: Optional[str] = None) -> str:
        """Fork a node into a new branch preserving messages to date."""

        original = self._require_node(node_id)
        branch_id = new_id or f"BRANCH-{uuid.uuid4().hex[:8]}"

        new_node = ChatNode(
            node_id=branch_id,
            parent_id=node_id,
            messages=list(original.messages),  # freeze state at branch time
        )
        self.nodes[branch_id] = new_node
        return branch_id

    def delete_node(self, node_id: str) -> None:
        """Remove a node from the graph."""

        self._require_node(node_id)
        del self.nodes[node_id]

    def get_node(self, node_id: str) -> ChatNode:
        return self._require_node(node_id)

    def add_message(self, node_id: str, role: MessageRole, content: str) -> None:
        self._require_node(node_id).messages.append(Message(role, content))

    def merge_nodes(self, id1: str, id2: str) -> str:
        """Concatenate two nodes' histories into a new merged node and drop the second."""

        node1 = self._require_node(id1)
        node2 = self._require_node(id2)

        merged_msgs = node1.messages + node2.messages  # raw concatenation
        merged_id = f"MERGED-{uuid.uuid4().hex[:8]}"

        new_node = ChatNode(
            node_id=merged_id,
            parent_id=None,
            messages=merged_msgs,
        )

        self.nodes[merged_id] = new_node
        self.delete_node(id2)
        return merged_id

    def show_full_graph(self) -> None:
        """Pretty-print the nodes ordered by depth and id."""

        depths = self._compute_depths()
        ordered = sorted(self.nodes.values(), key=lambda n: (depths[n.node_id], n.node_id))

        print("\n=== Conversation Graph ===\n")
        for node in ordered:
            depth = depths[node.node_id]
            indent = "  " * depth
            print(f"{indent}- Node ID: {node.node_id}")
            print(f"{indent}  Parent: {node.parent_id}")
            print(f"{indent}  Messages:")

            if not node.messages:
                print(f"{indent}    (empty)")
            else:
                for m in node.messages:
                    print(f"{indent}    [{m.role}] {m.content}")

            print()
        print("==========================\n")

    def _compute_depths(self) -> Dict[str, int]:
        depths: Dict[str, int] = {}
        for node_id in self.nodes:
            depth = 0
            cursor = self.nodes[node_id]
            while cursor.parent_id is not None:
                depth += 1
                cursor = self.nodes[cursor.parent_id]
            depths[node_id] = depth
        return depths

    def _require_node(self, node_id: str) -> ChatNode:
        if node_id not in self.nodes:
            raise KeyError(f"Node '{node_id}' does not exist")
        return self.nodes[node_id]


def build_context(graph: ConversationGraph, node_id: str) -> List[HumanMessage | AIMessage]:
    """Translate stored messages into LangChain message objects."""

    node = graph.get_node(node_id)
    msgs: List[HumanMessage | AIMessage] = []
    for m in node.messages:
        if m.role == "user":
            msgs.append(HumanMessage(content=m.content))
        else:
            msgs.append(AIMessage(content=m.content))
    return msgs


def chat(llm: ChatAnthropic, graph: ConversationGraph, node_id: str, user_input: str) -> str:
    """Send a user message to a node, persist the exchange, and return the reply."""

    graph.add_message(node_id, "user", user_input)
    response = llm.invoke(build_context(graph, node_id))
    graph.add_message(node_id, "assistant", response.content)  # type: ignore[arg-type]
    return response.content  # type: ignore[arg-type]


def print_graph(graph: ConversationGraph) -> None:
    """Helper to bracket the graph dump with dividers for readability."""

    divider = "#" * 30
    print(divider)
    graph.show_full_graph()
    print(divider)


def run_demo() -> None:
    """Execute a sample conversation showing branching, merging, and graph inspection."""

    llm = create_llm()
    graph = ConversationGraph()
    root = graph.create_root()

    print(chat(llm, graph, root, "Explain quantum computing simply."))
    print_graph(graph)

    b1 = graph.branch_from(root, "BRANCH-1")
    print(chat(llm, graph, b1, "Tell me about QAOA."))
    print_graph(graph)

    b2 = graph.branch_from(root, "BRANCH-2")
    print(chat(llm, graph, b2, "Explain Grover's algorithm."))
    print_graph(graph)

    print(chat(llm, graph, b2, "Explain Shor's algorithm."))
    print_graph(graph)

    merged = graph.merge_nodes(root, b2)
    print("Merged node id:", merged)
    print_graph(graph)

    print(chat(llm, graph, merged, "Continue from here with a combined explanation."))
    print_graph(graph)


if __name__ == "__main__":
    run_demo()
