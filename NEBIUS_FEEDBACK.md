# Developer Feedback: Nebius Token Factory & NVIDIA AI Tools

*Author: LexiRAG Engineering Team | NVIDIA × Nebius Global AI Hackathon*

---

### Executive Submission Paragraph (Ready for Submission Form)

> "Integrating NVIDIA Nemotron models via Nebius Token Factory delivered a unified, production-grade inference backbone for legal RAG. The standout technical capability is Nemotron's deep chain-of-thought reasoning, which excels at multi-statute Indian legal synthesis and structured statutory interpretation. However, an essential architectural discovery every engineer must navigate is how Nebius returns reasoning output: Nemotron models deliver their generated rationale and conclusions inside the `reasoning_content` payload attribute rather than the standard OpenAI `content` field. Standard client parsers expecting `message.content` fail silently with empty strings. Implementing a multi-tier defensive extractor (`reasoning_content` -> `model_extra` -> `content`) was essential to unlock Nemotron's full capabilities. Furthermore, having `BAAI/bge-m3` dense embeddings served directly on the same Token Factory `/v1/embeddings` endpoint streamlined our entire RAG architecture, eliminating the complexity of running external embedding clusters."

---

### Section 1: Empirically Verified Integration Findings

#### 1. The `reasoning_content` Response Structure (Critical Developer Experience Finding)
* **Verified Behavior**: On Nebius Token Factory, NVIDIA Nemotron models (`nvidia/nemotron-3-nano-30b-a3b` and `nvidia/nemotron-3-super-120b-a12b`) populate `choice.message.reasoning_content` with their thinking trace and legal synthesis, while `choice.message.content` is returned empty or `None`.
* **Impact on Python SDKs**: In Pydantic v2 (OpenAI Python SDK `>=1.0.0`), undeclared attributes on `ChatCompletionMessage` are placed into `model_extra["reasoning_content"]` or require dynamic attribute inspection. Unaware integrations will encounter HTTP 200 OK responses with empty string payloads.
* **Actionable Recommendations for Nebius**:
  1. **Documentation Callout**: Add an explicit note in the Nebius Token Factory Model Library for Nemotron reasoning models: *"OpenAI SDK integration: Response payload is located in `choice.message.reasoning_content`."*
  2. **Optional API Passthrough Parameter**: Support an optional request flag in `extra_body` (e.g., `{"mirror_reasoning_to_content": true}`) allowing seamless drop-in compatibility with legacy frameworks that only bind to `message.content`.

#### 2. Single-Endpoint RAG Co-location (BGE-M3 Embeddings)
* **Verified Behavior**: Having `BAAI/bge-m3` served natively at `https://api.tokenfactory.nebius.com/v1/embeddings` allows both dense vector generation and LLM reasoning to run through a single authenticated base URL and API key.
* **Impact**: Eliminates the operational overhead and cold starts of managing self-hosted embedding containers or separate cloud providers, ensuring the entire RAG pipeline runs strictly on Nebius infrastructure.

#### 3. Standard HTTP Error Adherence & Resilience
* **Verified Behavior**: Nebius Token Factory returns standard HTTP 429 status codes with actionable error payloads when rate limits are approached. This enables standard exponential backoff policies (`tenacity`) to retry transparently without connection resets.

---

### Section 2: Qualitative Model Capabilities & Benchmarking Roadmap

#### Qualitative Architecture Observations
* **Nemotron-3-Nano-30b**: Effective as an ultra-low-latency classification and routing node. It reliably categorizes queries into straightforward statutory lookups vs multi-statute conflicts.
* **Nemotron-3-Super-120b**: Demonstrates strong adherence to complex statutory prompts, citing specific Sections, Sub-sections, and judicial ratios without conversational filler.

#### Benchmarking Commitment (Phase 10)
To maintain the highest standard of scientific integrity, LexiRAG does not publish estimated or unsupported numerical performance claims. The following metrics are slated for rigorous evaluation in **Phase 10 (Testing, Benchmarking & Reliability)** using automated test harnesses:
1. **Time-to-First-Token (TTFT) and End-to-End P50/P95/P99 Latency** for both Nano-30b and Super-120b.
2. **Retrieval Precision & Recall** across Qdrant Cloud vector search using BGE-M3 dense embeddings.
3. **Statutory Grounding Accuracy & Hallucination Rate** measured against a curated test suite of 50 Indian statutory questions (Companies Act, GST, IBC, and Income Tax Act).
