# The Complete AI Engineering Course
### From Zero to Production AI Systems — Volume 1

---

> **Who this is for:** Software developers and project managers who are new to AI Engineering and want to design and build production-grade AI systems.

> **Assumed knowledge:** APIs, databases, HTTP, basic programming, version control, deployment concepts.

> **No assumed knowledge:** Machine learning, neural networks, maths, statistics, or AI APIs.

> **What you will build by the end:** A complete, production-ready AI system with agents, RAG pipelines, multi-modal capabilities, observability, security controls, and cost management — deployed on real infrastructure.

---

## Course Structure

### MODULE 1 — FOUNDATIONS
*Build the mental model before writing production code.*

| # | Chapter | Status | Key Skills |
|---|---------|--------|-----------|
| 01 | [What is AI Engineering? The Complete Landscape](./chapters/chapter-01-what-is-ai-engineering.md) | ✅ | Vocabulary, first API calls, full-stack chat app |
| 02 | [How LLMs Work: The Brain Behind AI](./chapters/chapter-02-how-llms-work.md) | ✅ | Tokenisation, context windows, temperature, hallucination |
| 03 | [Setting Up Your AI Development Environment](./chapters/chapter-03-dev-environment.md) | ✅ | VS Code, Cursor, Claude Code CLI, Antigravity CLI, uv, Ollama, LM Studio, Docker |

**Module 1 Learning Goal:** Understand what LLMs are, how they work internally, and operate a professional AI development environment.

---

### MODULE 2 — WORKING WITH AI APIS
*Master the tools that connect your code to AI models.*

| # | Chapter | Status | Key Skills |
|---|---------|--------|-----------|
| 04 | [AI APIs, SDKs & Streaming](./chapters/chapter-04-ai-apis-sdks.md) | ✅ | Multi-provider API calls, streaming, batching, error handling, retries |
| 05 | [Prompt Engineering](./chapters/chapter-05-prompt-engineering.md) | ✅ | System prompts, few-shot, chain-of-thought, prompt templates, evaluation |
| 06 | [Structured Outputs & Function Calling](./chapters/chapter-06-structured-outputs.md) | ✅ | JSON output, schema enforcement, function definitions, tool mechanics |

> **Note on Chapters 6 and 10:** Chapter 6 covers the mechanics of structured outputs and function calling — how to define tools and get structured JSON from a model. Chapter 10 builds on this foundation to create *agents* — AI systems that choose when and how to use those tools autonomously.

**Module 2 Learning Goal:** Call any AI API reliably, engineer prompts systematically, and get structured data back from AI models.

---

### MODULE 3 — BUILDING AI APPLICATIONS
*Combine AI with data to build real products.*

| # | Chapter | Status | Key Skills |
|---|---------|--------|-----------|
| 07 | [Embeddings: Teaching AI to Understand Meaning](./chapters/chapter-07-embeddings.md) | ✅ | Vector representations, semantic search, similarity, embedding models |
| 08 | [Vector Databases: AI Memory at Scale](./chapters/chapter-08-vector-databases.md) | ✅ | Pinecone, Weaviate, ChromaDB, pgvector, indexing, querying |
| 09 | [RAG: Retrieval Augmented Generation](./chapters/chapter-09-rag.md) | ✅ | Document ingestion, chunking, retrieval, generation, citation |
| 10 | [AI Agents: Autonomous AI Systems](./chapters/chapter-10-ai-agents.md) | ✅ | ReAct pattern, tool use, memory, planning, agent loops |

**Module 3 Learning Goal:** Build AI applications that work with your own data — not just the model's training data.

---

### MODULE 4 — ADVANCED AI SYSTEMS
*Build complex, multi-model, multi-modal systems.*

| # | Chapter | Status | Key Skills |
|---|---------|--------|-----------|
| 11 | [Multi-Agent Systems & Orchestration](./chapters/chapter-11-multi-agent.md) | ✅ | Orchestrator/worker patterns, coordination, state management |
| 12 | [Local AI: Ollama, LM Studio & Self-Hosted Models](./chapters/chapter-12-local-ai.md) | ✅ | Running models locally, model selection, hardware requirements, Modelfile |
| 13 | [Fine-Tuning & Model Customization](./chapters/chapter-13-fine-tuning.md) | ✅ | When to fine-tune, dataset preparation, training, evaluation |
| 14 | [Multi-Modal AI: Images, Audio & Video](./chapters/chapter-14-multimodal.md) | ✅ | Vision, speech-to-text, image generation, multi-modal pipelines |

**Module 4 Learning Goal:** Build systems that coordinate multiple AI models, run models locally, and process non-text data.

---

### MODULE 5 — PRODUCTION AI ENGINEERING
*Deploy, operate, and maintain AI systems at scale.*

| # | Chapter | Status | Key Skills |
|---|---------|--------|-----------|
| 15 | [Production Architecture: Building AI at Scale](./chapters/chapter-15-production-architecture.md) | ✅ | Rate limiting, queuing, caching, fallbacks, load balancing |
| 16 | [Testing & Evaluating AI Systems](./chapters/chapter-16-testing-evaluation.md) | ✅ | Unit testing, integration testing, golden sets, LLM-as-judge, evals |
| 17 | [AI Observability: Monitoring & Tracing](./chapters/chapter-17-observability.md) | ✅ | LangSmith, Helicone, OpenTelemetry, logging, alerting |
| 18 | [AI Security: Prompt Injection, Safety & Red-Teaming](./chapters/chapter-18-security.md) | 🔜 | Threat modelling, prompt injection, content filtering, red-teaming |
| 19 | [Cost Engineering: Running AI Without Going Broke](./chapters/chapter-19-cost-engineering.md) | ⬜ | Token budgets, caching strategy, model routing, batch processing |
| 20 | [Capstone: Build a Production AI System End-to-End](./chapters/chapter-20-capstone.md) | ⬜ | Integrates all 19 chapters into one complete production system |

**Module 5 Learning Goal:** Deploy AI systems that are observable, secure, cost-controlled, and reliable enough for production traffic.

---

## Chapter Dependency Map

```
Ch 01 (What is AI Engineering)
  └── Ch 02 (How LLMs Work)
       └── Ch 03 (Dev Environment)
            └── Ch 04 (AI APIs & SDKs)
                 ├── Ch 05 (Prompt Engineering)
                 │    └── Ch 06 (Structured Outputs)
                 │         └── Ch 10 (AI Agents)
                 │              └── Ch 11 (Multi-Agent)
                 └── Ch 07 (Embeddings)
                      └── Ch 08 (Vector Databases)
                           └── Ch 09 (RAG)
                                └── Ch 10 (AI Agents)

Ch 03 (Dev Environment)
  └── Ch 12 (Local AI)

Ch 04–12 (Any completed chapter)
  └── Ch 13 (Fine-Tuning)
  └── Ch 14 (Multi-Modal)
  └── Ch 15–19 (Production chapters)

Ch 15–19 (All production chapters)
  └── Ch 20 (Capstone)
```

---

## Tools & Technologies Covered

| Category | Tools |
|----------|-------|
| **AI Providers (Cloud)** | Anthropic (Claude Haiku/Sonnet/Opus), OpenAI (GPT-4o/mini), Google AI (Gemini) |
| **AI Tools (Local)** | Ollama, LM Studio |
| **Dev Environments** | VS Code, Cursor, Claude Code CLI, Gemini CLI |
| **AI Assistants** | GitHub Copilot |
| **Languages** | Python (uv), TypeScript/JavaScript (Node.js) |
| **Frontend** | React, React Native |
| **Infrastructure** | Docker, Docker Compose, Kubernetes (basics) |
| **Vector DBs** | Pinecone, Weaviate, ChromaDB, pgvector |
| **Observability** | LangSmith, Helicone, OpenTelemetry |
| **Databases** | PostgreSQL, MongoDB, Redis, SQLite |
| **Cloud** | AWS, GCP, Vercel |

---

## How to Use This Course

1. **Complete chapters in order.** The dependency map above shows why — later chapters assume earlier ones.
2. **Read the full chapter before touching any code.** Diagrams and explanations prime your mental model.
3. **Run every code example.** Reading code is not the same as running code.
4. **Complete the exercises** before moving on — they verify your understanding.
5. **Build the mini project** (2–3 hours) — this converts reading into doing.
6. **Build the production project** when you finish each module — this tests whether you can apply the concepts independently.
7. **Use Claude Code, Cursor, or GitHub Copilot** to help you when stuck — the AI tools themselves are part of the curriculum.

---

## Progress Tracker

| Status | Count |
|--------|-------|
| ✅ Complete | 17 |
| 🔜 Next | 1 |
| ⬜ Not started | 2 |

---

*Start here → [Chapter 1: What is AI Engineering?](./chapters/chapter-01-what-is-ai-engineering.md)*

*See future volumes → [ROADMAP.md](./ROADMAP.md)*
