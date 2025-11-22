import os
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

from dotenv import load_dotenv
from langchain.chat_models.base import BaseChatModel # type: ignore
from langchain.messages import AIMessage, HumanMessage
from langchain_anthropic import ChatAnthropic

load_dotenv()


USE_FAKE_LLM = False
DEFAULT_MODEL = "claude-sonnet-4-5"


class DummyChatAnthropic:
    """Lightweight stand-in that mimics ChatAnthropic's invoke interface for testing."""

    def __init__(self, reply: str = "<<AI Res>>") -> None:
        self.reply = reply

    def invoke(self, *_: object, **__: object) -> AIMessage:
        return AIMessage(content=self.reply)


def create_llm(api_key: Optional[str] = None, model: str = DEFAULT_MODEL) -> BaseChatModel:
    """Instantiate the Anthropic chat model with a helpful error if the key is missing."""

    if USE_FAKE_LLM:
        return DummyChatAnthropic()  # type: ignore[return-value]

    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("Missing ANTHROPIC_API_KEY in environment or .env")

    return ChatAnthropic(model=model, api_key=key)  # type: ignore[arg-type]


MessageRole = Literal["user", "assistant"]


@dataclass
class Message:
    role: MessageRole
    content: str


MessageLike = Message | tuple[MessageRole, str]


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

    def branch_from(
        self,
        node_id: str,
        new_id: Optional[str] = None,
        *,
        carry_messages: bool = True,
        additional_messages: Optional[List[MessageLike]] = None,
    ) -> str:
        """Fork a node into a new branch with optional history and extra context.

        carry_messages controls whether existing messages are copied. additional_messages
        allows seeding the branch with extra user/assistant turns at creation.
        """

        original = self._require_node(node_id)
        branch_id = new_id or f"BRANCH-{uuid.uuid4().hex[:8]}"

        messages: List[Message] = list(original.messages) if carry_messages else []
        if additional_messages:
            for msg in additional_messages:
                if isinstance(msg, Message):
                    messages.append(msg)
                else:
                    role, content = msg
                    messages.append(Message(role=role, content=content))

        new_node = ChatNode(node_id=branch_id, parent_id=node_id, messages=messages)
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

    def merge_nodes(self, target_id: str, source_id: str) -> str:
        """Merge source into target in place, reparenting source's children.

        Rules:
        - If histories are identical, target remains unchanged.
        - If one is a prefix of the other, the longer tail is kept.
        - If histories are disjoint (e.g., branch created without context), source is appended.
        - Otherwise, divergence after a shared prefix raises an error.
        - If source extends target, the extra source messages are appended to target.
          If target is already longer or equal and histories match, target is unchanged.
        - Children of the source node are reparented to the target before the source is deleted.
        """

        target = self._require_node(target_id)
        source = self._require_node(source_id)

        merged_messages = self._merge_histories(target.messages, source.messages)
        target.messages = merged_messages

        # Reparent children of source to target
        for node in self.nodes.values():
            if node.parent_id == source_id:
                node.parent_id = target_id

        if source_id != target_id:
            self.delete_node(source_id)

        return target_id

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

    @staticmethod
    def _merge_histories(target_msgs: List[Message], source_msgs: List[Message]) -> List[Message]:
        """Merge two histories with prefix detection and permissive append for fresh branches."""

        prefix_len = 0
        for m1, m2 in zip(target_msgs, source_msgs):
            if m1 == m2:
                prefix_len += 1
            else:
                break

        if prefix_len == len(target_msgs) and prefix_len == len(source_msgs):
            return list(target_msgs)  # identical

        if prefix_len == len(target_msgs):
            # target is prefix; append source remainder
            return list(target_msgs) + source_msgs[prefix_len:]

        if prefix_len == len(source_msgs):
            # source is prefix; keep target as-is
            return list(target_msgs)

        if prefix_len == 0:
            # Completely disjoint (e.g., branch created without context); append source to target.
            return list(target_msgs) + list(source_msgs)

        raise ValueError("Cannot merge: histories diverge after shared prefix")


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


def chat(llm: BaseChatModel, graph: ConversationGraph, node_id: str, user_input: str) -> str:
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

    b2a = graph.branch_from(b2, "BRANCH-2A")
    print(chat(llm, graph, b2a, "Summarize Grover vs. amplitude amplification variants."))
    print_graph(graph)

    b3 = graph.branch_from(
        b1,
        "BRANCH-3",
        carry_messages=False,
        additional_messages=[
            ("user", "Start a clean fork: briefly compare QAOA vs VQE."),
            ("assistant", "Sure, I'll outline similarities and differences."),
        ],
    )
    print(chat(llm, graph, b3, "Include a note on current research directions."))
    print_graph(graph)

    merged = graph.merge_nodes(root, b2)
    print("Merged node id:", merged)
    print_graph(graph)

    print(chat(llm, graph, merged, "Continue from here with a combined explanation."))
    print_graph(graph)


if __name__ == "__main__":
    run_demo()
