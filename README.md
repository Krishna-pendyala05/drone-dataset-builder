# drone-dataset-builder
An AI-augmented ETL pipeline designed to crawl and extract high-fidelity technical specifications for drones. This project utilizes GPT-5.2 Codex and the o3 reasoning model to handle dynamic web content and resolve inconsistent data fields in complex hardware datasets.

## Architecture
```
Seeds (URLs)
    └── agents.cli:build_dataset
          ├── fetch_markdown (Playwright deep wait)
          ├── deterministic parser (tables/lists) -> optional LLM mapper -> Pydantic schemas
          ├── extract_with_self_healing (brand/model retry)
          └── JSONL writer (+progress logging)
```

## Quickstart (semantic crawler)
1. Install dependencies:
   ```bash
   pip install playwright pydantic markdownify
   playwright install chromium
   ```
2. Run the sample crawler to see the flow end-to-end:
   ```bash
   python -c "from agents.crawler import run_sample; print(run_sample())"
   ```
3. Execute the offline parsing flow with the local fixtures and stub parsers:
   ```bash
   python -m pytest
   # or
   make test
   ```
4. Integrate with your own parsing agent by passing a `parser(prompt, url) -> JSON string` callable into `extract_with_self_healing`.

## CLI dataset generator
Generate JSONL from a list of seed URLs (one per line):
```bash
python -m agents.cli --seeds seeds.txt --output drones.jsonl
# Optional: provide an LLM mapper to enrich deterministic fields
python -m agents.cli --seeds seeds.txt --output drones.jsonl --llm-mapper mymodule:mapper_fn
```
