# AI Engineering Handbook

### From Zero to Production AI Systems — Volume 1

A complete, practical AI Engineering course for software developers and technical project managers. Not another conceptual overview — a handbook that produces engineers capable of designing, building, debugging, deploying, and operating production AI systems.

---

## What This Is

This handbook is structured as a 20-chapter bootcamp. Each chapter is a full book chapter — 60–90 minutes of reading, 3–5 hours of hands-on work, working code in Python and Node.js, architecture diagrams, production issue walkthroughs, mini projects, and production projects.

The goal is not a reader who can explain AI. The goal is a reader who can deliver working AI systems.

---

## Who It's For

| Reader | Background | What They Get |
|--------|-----------|--------------|
| **Software Developer** | Any stack, zero AI knowledge | Full AI Engineering capability from scratch |
| **Technical PM** | Manages technical teams | Enough depth to lead, scope, and de-risk AI projects |
| **Backend Engineer** | Python / Node.js experience | Production-grade patterns for every AI system type |

**Assumed knowledge:** APIs, HTTP, basic programming, git, deployment concepts.

**No assumed knowledge:** Machine learning, neural networks, mathematics, statistics, AI APIs.

---

## Progress

**11 of 20 chapters complete** — Chapter 11 (Multi-Agent Systems) now complete.

| Module | Chapters | Status |
|--------|----------|--------|
| 1 — Foundations | Ch 01–03 | ✅ Complete |
| 2 — Working with AI APIs | Ch 04–06 | ✅ Complete |
| 3 — Building AI Applications | Ch 07–10 | ✅ Complete |
| 4 — Advanced AI Systems | Ch 11–14 | 🔜 Ch 12 next |
| 5 — Production AI Engineering | Ch 15–20 | ⬜ Not started |

---

## Course Structure

### Module 1 — Foundations

| # | Chapter | Status |
|---|---------|--------|
| 01 | [What is AI Engineering? The Complete Landscape](./chapters/chapter-01-what-is-ai-engineering.md) | ✅ Complete |
| 02 | [How LLMs Work: The Brain Behind AI](./chapters/chapter-02-how-llms-work.md) | ✅ Complete |
| 03 | [Setting Up Your AI Development Environment](./chapters/chapter-03-dev-environment.md) | ✅ Complete |

### Module 2 — Working with AI APIs

| # | Chapter | Status |
|---|---------|--------|
| 04 | [AI APIs, SDKs & Streaming](./chapters/chapter-04-ai-apis-sdks.md) | ✅ Complete |
| 05 | [Prompt Engineering](./chapters/chapter-05-prompt-engineering.md) | ✅ Complete |
| 06 | [Structured Outputs & Function Calling](./chapters/chapter-06-structured-outputs.md) | ✅ Complete |

### Module 3 — Building AI Applications

| # | Chapter | Status |
|---|---------|--------|
| 07 | [Embeddings: Teaching AI to Understand Meaning](./chapters/chapter-07-embeddings.md) | ✅ Complete |
| 08 | [Vector Databases: AI Memory at Scale](./chapters/chapter-08-vector-databases.md) | ✅ Complete |
| 09 | [RAG: Retrieval Augmented Generation](./chapters/chapter-09-rag.md) | ✅ Complete |
| 10 | [AI Agents: Autonomous AI Systems](./chapters/chapter-10-ai-agents.md) | ✅ Complete |

### Module 4 — Advanced AI Systems

| # | Chapter | Status |
|---|---------|--------|
| 11 | [Multi-Agent Systems & Orchestration](./chapters/chapter-11-multi-agent.md) | ✅ Complete |
| 12 | [Local AI: Ollama, LM Studio & Self-Hosted Models](./chapters/chapter-12-local-ai.md) | ⬜ |
| 13 | [Fine-Tuning & Model Customization](./chapters/chapter-13-fine-tuning.md) | ⬜ |
| 14 | [Multi-Modal AI: Images, Audio & Video](./chapters/chapter-14-multimodal.md) | ⬜ |

### Module 5 — Production AI Engineering

| # | Chapter | Status |
|---|---------|--------|
| 15 | [Production Architecture: Building AI at Scale](./chapters/chapter-15-production-architecture.md) | ⬜ |
| 16 | [Testing & Evaluating AI Systems](./chapters/chapter-16-testing-evaluation.md) | ⬜ |
| 17 | [AI Observability: Monitoring & Tracing](./chapters/chapter-17-observability.md) | ⬜ |
| 18 | [AI Security: Prompt Injection, Safety & Red-Teaming](./chapters/chapter-18-security.md) | ⬜ |
| 19 | [Cost Engineering: Running AI Without Going Broke](./chapters/chapter-19-cost-engineering.md) | ⬜ |
| 20 | [Capstone: Build a Production AI System End-to-End](./chapters/chapter-20-capstone.md) | ⬜ |

---

## What Each Chapter Contains

Every chapter includes all of the following:

- **Learning objectives** — what you will be able to do after reading
- **Why this topic exists** — the engineering problem it solves
- **Real-world analogy** — something a developer immediately recognises
- **Core concepts** — every term defined: technical + plain English + analogy
- **Architecture diagrams** — Mermaid diagrams before any code
- **Three implementation levels** — learning example → production example → enterprise example
- **Technology comparisons** — never "use this tool" without comparing alternatives
- **Decision frameworks** — how to choose, not just what to choose
- **Production issues** — realistic failures with symptoms, root cause, diagnosis, fix, and prevention
- **Best practices** — numbered, actionable, with code
- **Security considerations** — threats specific to the topic
- **Cost breakdown** — free vs paid, development vs production
- **Common mistakes** — wrong/right code pairs
- **Debugging guide** — diagnostic flowchart + error reference table
- **Exercises** — 5 practical exercises with time estimates
- **Quiz** — 10 questions with full answers
- **Mini project** — 2–3 hours, builds something useful
- **Production project** — 1–2 days, realistic system
- **Key takeaways, glossary, and further reading**

---

## Tools and Technologies Covered

| Category | Tools |
|----------|-------|
| **AI Providers** | Anthropic Claude, OpenAI GPT-4o, Google Gemini |
| **Local AI** | Ollama, LM Studio |
| **Dev Environments** | VS Code, Cursor, Claude Code CLI, Antigravity CLI |
| **AI Assistants** | GitHub Copilot, Continue.dev, Gemini Code Assist |
| **Python** | uv, anthropic SDK, openai SDK, FastAPI |
| **Node.js / TypeScript** | @anthropic-ai/sdk, Express, nvm |
| **Frontend** | React, React Native |
| **Infrastructure** | Docker, Docker Compose, Kubernetes |
| **Vector Databases** | ChromaDB, Pinecone, Weaviate, pgvector, Qdrant |
| **Observability** | LangSmith, Helicone, OpenTelemetry |
| **Databases** | PostgreSQL, MongoDB, Redis |
| **Cloud** | AWS, GCP, Vercel |

---

## How to Use This Handbook

1. **Read chapters in order** — each chapter builds on the previous ones
2. **Run every code example** — reading code is not learning code
3. **Complete the exercises** before moving to the next chapter
4. **Build the mini project** — this converts reading into doing
5. **Build the production project** at the end of each module

You do not need to buy any tools to start. Chapter 3 explains how to get a complete professional AI development environment — including free local models via Ollama — running in under 2 hours.

---

## Prerequisites

- A computer running macOS, Linux, or Windows 11 (8 GB RAM minimum, 16 GB recommended for local models)
- Basic programming knowledge (any language)
- Familiarity with APIs and HTTP
- Git and a terminal

Start here: [Chapter 1 — What is AI Engineering?](./chapters/chapter-01-what-is-ai-engineering.md)

---

## Future Volumes

This is Volume 1 of a planned 12-volume series. Future volumes cover:

- **Volume 2:** MCP Engineering
- **Volume 3:** AI Agent Engineering
- **Volume 4:** n8n AI Workflow Automation
- **Volume 5:** RAG Deep Dive
- **Volume 6:** Vector Database Engineering
- **Volume 7:** Coding Agents
- **Volume 8:** DevOps AI
- **Volume 9:** Technical PM AI
- **Volume 10:** Enterprise AI Systems
- **Volume 11:** AI Architecture Patterns
- **Volume 12:** Real Production Case Studies

See [ROADMAP.md](./ROADMAP.md) for the full plan.

---

## Contributing

This handbook is actively being written. Content quality and practical accuracy take priority over publication speed. If you find a technical error, an outdated API reference, or a better way to explain a concept, open an issue.

---

*Started June 2026. Volume 1 target: 20 chapters.*
