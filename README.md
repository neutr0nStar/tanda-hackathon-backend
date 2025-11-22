## Conversation Graph Backend

Flask app + LangChain wrapper with branching conversations and a simple web UI.

### Run the app
```bash
export ANTHROPIC_API_KEY=...
# optional: USE_FAKE_LLM=1 to stub responses, SYSTEM_PROMPT="Respond concisely."
python server.py
```
Open `http://localhost:5000` for the UI.

---

## API
Base URL: `http://localhost:5000`

All responses are JSON; errors return `{"error": "<message>"}` with appropriate status codes.

### List nodes
`GET /api/nodes`
- Returns all nodes with parent, children, and message counts.

### Get a node
`GET /api/nodes/<node_id>`
- Returns node metadata and full message list.

### Chat
`POST /api/chat`
```json
{ "node_id": "ROOT", "message": "Hello" }
```
- Appends the user/assistant turns and returns the assistant reply.

### Branch
`POST /api/branch`
```json
{
  "parent_id": "ROOT",
  "new_id": "BRANCH-1",
  "carry_messages": true,
  "seed_message": "Optional seed as user"
}
```
- Creates a new branch from `parent_id`; if `new_id` is omitted, auto-generated.

### Merge
`POST /api/merge`
```json
{ "target_id": "ROOT", "source_id": "BRANCH-1" }
```
- Merges `source_id` into `target_id` (prefix/disjoint-aware).

### Summarize & Branch
`POST /api/summarize_branch`
```json
{ "source_id": "BRANCH-1", "new_id": "SUMM-1" }
```
- Creates a new branch containing only a concise summary of the source branch (default id `SUMM-<parent-suffix>`).

### Switch LLM backend
`POST /api/llm_mode`
```json
{ "mode": "real" }   // or "fake"
```
- Switches between Anthropic and the local dummy responder (`<<AI Res>>`).
