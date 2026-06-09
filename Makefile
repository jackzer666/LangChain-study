.PHONY: api

api:
	uv run uvicorn agent_rag_tool_project.app2:app --reload --host 0.0.0.0 --port 8010
