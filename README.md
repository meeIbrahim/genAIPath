# Gen AI Roadmap

## Setup

Install dependencies:
```
uv sync
uv run python -m spacy download en_core_web_sm
```

Before running the app, ensure:
- Ollama is running locally with the `qwen3-embedding:0.6b` model (`ollama pull qwen3-embedding:0.6b`)
- `GROQ_API_KEY` is exported in your environment
