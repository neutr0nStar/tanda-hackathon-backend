import os
from dataclasses import asdict
from typing import Dict, List

from flask import Flask, flash, redirect, render_template_string, request, url_for

from main import (
    ConversationGraph,
    Message,
    MessageRole,
    chat,
    create_llm,
)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret")

graph = ConversationGraph()
root_id = graph.create_root()
llm = create_llm()


def _build_children_index(graph: ConversationGraph) -> Dict[str, List[str]]:
    children: Dict[str, List[str]] = {}
    for node in graph.nodes.values():
        if node.parent_id is None:
            continue
        children.setdefault(node.parent_id, []).append(node.node_id)
    return children


def _render_tree(graph: ConversationGraph) -> str:
    children = _build_children_index(graph)

    def walk(node_id: str, indent: str = "") -> List[str]:
        lines = [f"{indent}- {node_id}"]
        for child_id in sorted(children.get(node_id, [])):
            lines.extend(walk(child_id, indent + "  "))
        return lines

    roots = [node_id for node_id, node in graph.nodes.items() if node.parent_id is None]
    tree_lines: List[str] = []
    for rid in sorted(roots):
        tree_lines.extend(walk(rid))
    return "\n".join(tree_lines) if tree_lines else "(empty graph)"


def _compute_depths(graph: ConversationGraph) -> Dict[str, int]:
    depths: Dict[str, int] = {}
    for node_id in graph.nodes:
        depth = 0
        cursor = graph.nodes[node_id]
        while cursor.parent_id is not None:
            depth += 1
            cursor = graph.nodes[cursor.parent_id]
        depths[node_id] = depth
    return depths


def _render_svg(graph: ConversationGraph, highlight: str) -> str:
    """Build a lightweight layered tree SVG."""

    depths = _compute_depths(graph)
    layers: Dict[int, List[str]] = {}
    for node_id, depth in depths.items():
        layers.setdefault(depth, []).append(node_id)

    # deterministic ordering
    for nodes in layers.values():
        nodes.sort()

    x_spacing, y_spacing, radius = 160, 120, 48
    coords: Dict[str, tuple[int, int]] = {}

    for depth in sorted(layers):
        for idx, node_id in enumerate(layers[depth]):
            x = 80 + depth * x_spacing
            y = 60 + idx * y_spacing
            coords[node_id] = (x, y)

    width = (max(depths.values()) + 1) * x_spacing + 80 if depths else 400
    height = max((y for _, y in coords.values()), default=0) + 80

    lines = []
    lines.append(f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
                 'xmlns="http://www.w3.org/2000/svg">')
    lines.append('<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="6" refY="3" orient="auto">'
                 '<path d="M0,0 L0,6 L9,3 z" fill="#888" /></marker></defs>')

    # edges
    for node in graph.nodes.values():
        if node.parent_id is None or node.node_id not in coords or node.parent_id not in coords:
            continue
        x1, y1 = coords[node.parent_id]
        x2, y2 = coords[node.node_id]
        lines.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            'stroke="#bbb" stroke-width="2" marker-end="url(#arrow)" />'
        )

    # nodes
    for node_id, (x, y) in coords.items():
        active = node_id == highlight
        fill = "#06c" if active else "#fff"
        stroke = "#06c" if active else "#333"
        lines.append(
            f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="2" />'
        )
        lines.append(
            f'<text x="{x}" y="{y+5}" text-anchor="middle" font-size="10" '
            f'font-family="Arial" fill="{ "#fff" if active else "#000"}">{node_id}</text>'
        )

    lines.append("</svg>")
    return "\n".join(lines)


def _serialize_messages(node_id: str) -> List[Dict[str, str]]:
    node = graph.get_node(node_id)
    return [asdict(m) for m in node.messages]


@app.route("/", methods=["GET", "POST"])
def index():
    selected_node = request.args.get("node") or root_id

    if request.method == "POST":
        action = request.form.get("action")
        try:
            if action == "chat":
                node_id = request.form.get("chat_node", root_id)
                user_input = (request.form.get("user_input") or "").strip()
                if not user_input:
                    flash("Please enter a message to chat.", "warning")
                else:
                    response = chat(llm, graph, node_id, user_input)
                    flash(f"Assistant: {response}", "info")
                    selected_node = node_id

            elif action == "branch":
                parent_id = request.form.get("branch_parent", root_id)
                new_id = request.form.get("branch_name") or None
                carry = bool(request.form.get("carry_messages"))
                seed_text = (request.form.get("branch_seed") or "").strip()
                additional = None
                if seed_text:
                    additional = [("user", seed_text)]  # type: ignore[list-item]
                branch_id = graph.branch_from(
                    parent_id,
                    new_id,
                    carry_messages=carry,
                    additional_messages=additional, # type: ignore
                )
                flash(f"Created branch {branch_id} from {parent_id}.", "success")
                selected_node = branch_id

            elif action == "merge":
                target = request.form.get("merge_target", root_id)
                source = request.form.get("merge_source")
                if not source:
                    flash("Select a source node to merge.", "warning")
                else:
                    merged_id = graph.merge_nodes(target, source)
                    flash(f"Merged {source} into {target} (result: {merged_id}).", "success")
                    selected_node = merged_id

            else:
                flash("Unknown action.", "warning")
        except Exception as exc:  # broad for user feedback
            flash(f"Error: {exc}", "danger")

        return redirect(url_for("index", node=selected_node))

    available_nodes = sorted(graph.nodes.keys())
    if selected_node not in graph.nodes:
        selected_node = root_id

    messages = _serialize_messages(selected_node)
    tree = _render_tree(graph)
    svg = _render_svg(graph, selected_node)

    return render_template_string(
        TEMPLATE,
        nodes=available_nodes,
        selected_node=selected_node,
        messages=messages,
        tree=tree,
        svg=svg,
    )


TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Conversation Graph UI</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 1.5rem; display: flex; gap: 1.5rem; }
    .column { flex: 1; min-width: 320px; }
    form { margin-bottom: 1rem; padding: 1rem; border: 1px solid #ddd; border-radius: 8px; }
    textarea, select, input[type=text] { width: 100%; box-sizing: border-box; }
    pre { background: #f6f6f6; padding: 1rem; border-radius: 8px; overflow-x: auto; }
    .messages { list-style: none; padding-left: 0; }
    .messages li { margin-bottom: 0.5rem; }
    .role-user { color: #0a7; }
    .role-assistant { color: #06c; }
    .flash { padding: 0.5rem 0.75rem; margin-bottom: 0.5rem; border-radius: 4px; }
    .flash.info { background: #e8f3ff; }
    .flash.success { background: #e8f7e8; }
    .flash.warning { background: #fff7e0; }
    .flash.danger { background: #ffe8e8; }
  </style>
</head>
<body>
  <div class="column">
    <form method="get">
      <h3>Switch Node</h3>
      <label>Active node</label>
      <select name="node">
        {% for node in nodes %}
          <option value="{{ node }}" {% if node == selected_node %}selected{% endif %}>{{ node }}</option>
        {% endfor %}
      </select>
      <button type="submit">Switch</button>
    </form>

    {% with msgs = get_flashed_messages(with_categories=true) %}
      {% if msgs %}
        {% for category, message in msgs %}
          <div class="flash {{ category }}">{{ message }}</div>
        {% endfor %}
      {% endif %}
    {% endwith %}

    <form method="post">
      <h3>Chat</h3>
      <input type="hidden" name="action" value="chat">
      <label>Node</label>
      <select name="chat_node">
        {% for node in nodes %}
          <option value="{{ node }}" {% if node == selected_node %}selected{% endif %}>{{ node }}</option>
        {% endfor %}
      </select>
      <label>Message</label>
      <textarea name="user_input" rows="3" placeholder="Ask something..."></textarea>
      <button type="submit">Send</button>
    </form>

    <form method="post">
      <h3>Branch</h3>
      <input type="hidden" name="action" value="branch">
      <label>Parent node</label>
      <select name="branch_parent">
        {% for node in nodes %}
          <option value="{{ node }}" {% if node == selected_node %}selected{% endif %}>{{ node }}</option>
        {% endfor %}
      </select>
      <label>New branch id (optional)</label>
      <input type="text" name="branch_name" placeholder="BRANCH-xyz">
      <label><input type="checkbox" name="carry_messages" checked> Carry existing messages</label>
      <label>Seed message (optional, user)</label>
      <textarea name="branch_seed" rows="2" placeholder="Seed the branch with user context"></textarea>
      <button type="submit">Create branch</button>
    </form>

    <form method="post">
      <h3>Merge</h3>
      <input type="hidden" name="action" value="merge">
      <label>Target (kept)</label>
      <select name="merge_target">
        {% for node in nodes %}
          <option value="{{ node }}" {% if node == selected_node %}selected{% endif %}>{{ node }}</option>
        {% endfor %}
      </select>
      <label>Source (merged into target)</label>
      <select name="merge_source">
        <option value="">-- choose --</option>
        {% for node in nodes %}
          {% if node != selected_node %}
          <option value="{{ node }}">{{ node }}</option>
          {% endif %}
        {% endfor %}
      </select>
      <button type="submit">Merge</button>
      <p><small>Merge allowed only if histories share a prefix; otherwise an error is shown.</small></p>
    </form>
  </div>

  <div class="column">
    <h3>Selected Node: {{ selected_node }}</h3>
    <ul class="messages">
      {% if messages %}
        {% for m in messages %}
          <li class="role-{{ m.role }}"><strong>[{{ m.role }}]</strong> {{ m.content }}</li>
        {% endfor %}
      {% else %}
        <li>(no messages)</li>
      {% endif %}
    </ul>

    <h3>Conversation Graph</h3>
    <pre>{{ tree }}</pre>
    <div style="margin-top:1rem; border:1px solid #ddd; border-radius:8px; padding:0.5rem; background:#fafafa;">
      {{ svg|safe }}
    </div>
  </div>
</body>
</html>
"""


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
