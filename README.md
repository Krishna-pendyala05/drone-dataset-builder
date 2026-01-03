# drone-dataset-builder
An AI-augmented ETL pipeline designed to crawl and extract high-fidelity technical specifications for drones. This project utilizes GPT-5.2 Codex and the o3 reasoning model to handle dynamic web content and resolve inconsistent data fields in complex hardware datasets.

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
3. Integrate with your own parsing agent by passing a `parser(prompt, url) -> JSON string` callable into `extract_with_self_healing`.

### Testing on dynamic sites (DJI, Autel Robotics)
Pass a `debug_agent(prompt, url) -> [selectors]` callable into `extract_with_self_healing`. If critical fields (brand/model) remain empty, the crawler asks:
```
Debug this page structure—identify the CSS selector for the technical specs table and update the Playwright wait logic.
```
The selectors returned will be used for a retry with deeper waits on Shadow DOM tables.

## Targeted test plan (boundary coverage)
Use live product pages to validate the crawler, parser, and debug loop:

1) **DJI Mavic 3 Pro (consumer/camera)**
   - Expected challenges: Nested tabs and expandable sections for specs.
   - Goal: Confirm `max_flight_time`, `sensor_type`, and `brand/model` populate without invoking the debug_agent.
2) **DJI Avata 2 (FPV)**
   - Expected challenges: Specs split across marketing and accessories sections.
   - Goal: Validate category `fpv`, `video_tx_power_mw` (if available), and ensure brand/model resolves; observe if debug selectors are requested.
3) **DJI Matrice 350 RTK (enterprise)**
   - Expected challenges: Spec tables may live in Shadow DOM with role-based selectors.
   - Goal: Trigger deep-wait success; if fields are empty, verify the debug prompt returns selectors such as `[role='table']` or `#specifications`.
4) **Autel EVO II Pro (camera/enterprise crossover)**
   - Expected challenges: Mixed marketing copy and modal spec popups.
   - Goal: Ensure Markdown capture includes modal content; check `sensor_type` and `payload_capacity` mapping accuracy.
5) **Parrot Anafi USA (enterprise)**
   - Expected challenges: PDF/spec download links instead of inline tables.
   - Goal: Observe how missing inline specs are handled; expect `debug_agent` to propose selectors or signal need for PDF ingestion in future iterations.
