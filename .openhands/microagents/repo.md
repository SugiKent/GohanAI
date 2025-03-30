# GohanAI Project Documentation

## Project Structure
```
GohanAI/
├── app/
│   ├── agents/               # Contains various agent implementations
│   │   ├── planner.py
│   │   ├── reporter.py
│   │   ├── researcher.py
│   │   ├── supervisor.py
│   │   ├── coder.py
│   │   ├── coordinator.py
│   ├── graph/                # Workflow/graph related code
│   │   ├── workflow.py
│   ├── prompts/              # Prompt templates
│   ├── utils/                # Utility functions
│   ├── main.py               # FastAPI entry point
├── docs/                     # Documentation
├── tests/                    # Test files
```

## Key Technologies
- FastAPI (main.py)
- Langchain (workflow.py)
- OpenAI integration (workflow.py)

## Entry Points
- Main API: `app/main.py` (FastAPI)
- Workflow definition: `app/graph/workflow.py`

## Agents
1. Planner
2. Reporter
3. Researcher
4. Supervisor
5. Coder
6. Coordinator
