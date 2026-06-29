# AI Engineering Course — Roadmap

## Volume Planning

This document plans future volumes beyond Volume 1. Do not generate content for these volumes until Volume 1 is complete. This is a planning reference only.

---

## Volume 1 — AI Engineering Foundations (Current)

**Status:** In progress (2 of 20 chapters complete)

**Objective:** A software developer with zero AI knowledge graduates capable of independently designing, building, debugging, deploying, and maintaining production AI systems.

**Chapters:** 20 chapters across 5 modules. See `COURSE_INDEX.md`.

---

## Volume 2 — MCP Engineering

**Objective:** Master the Model Context Protocol — the emerging standard for connecting AI models to tools, data sources, and external systems.

**Planned Chapters:**
1. What is MCP? The Protocol That Connects AI to Everything
2. MCP Architecture: Hosts, Clients, and Servers
3. Building Your First MCP Server in Python
4. Building Your First MCP Server in Node.js
5. MCP Tools: Exposing Functions to AI
6. MCP Resources: Exposing Data to AI
7. MCP Prompts: Reusable Prompt Templates
8. Connecting Claude Code to Custom MCP Servers
9. MCP Authentication and Security
10. Production MCP: Hosting, Scaling, and Monitoring
11. MCP Registry and Discovery
12. Building a Full MCP Ecosystem (Capstone)

**Prerequisites:** Volume 1 complete, particularly Chapters 10 (AI Agents) and 11 (Multi-Agent Systems).

---

## Volume 3 — AI Agent Engineering

**Objective:** Build sophisticated, production-grade AI agents that autonomously accomplish complex real-world tasks.

**Planned Chapters:**
1. Advanced Agent Patterns: Beyond ReAct
2. Agent Memory Systems: Short-Term, Long-Term, Episodic
3. Agent Planning: MCTS, Tree-of-Thought, and Beyond
4. Tool Design for Agents: What Makes a Good Tool
5. Agent Reliability: Retries, Fallbacks, and Human-in-the-Loop
6. Agent Evaluation: How to Know If Your Agent Is Good
7. Coding Agents: Autonomous Code Generation and Execution
8. Research Agents: Autonomous Web Research Systems
9. Data Agents: Autonomous Data Analysis and Reporting
10. Agent Orchestration Frameworks: LangGraph, CrewAI, AutoGen
11. Multi-Agent Collaboration Patterns
12. Production Agent Systems: Deployment, Monitoring, Cost (Capstone)

**Prerequisites:** Volume 1 complete, with emphasis on Chapters 10 and 11.

---

## Volume 4 — n8n AI Engineering

**Objective:** Build production AI workflows using n8n — the open-source workflow automation platform.

**Planned Chapters:**
1. What is n8n? Workflow Automation for AI Engineers
2. n8n Architecture: Nodes, Connections, and Executions
3. Setting Up n8n: Self-Hosted vs Cloud
4. Connecting AI Models to n8n (Claude, GPT-4, Gemini)
5. Building AI-Powered Notification Systems
6. Building AI Document Processing Pipelines
7. Building AI Customer Support Automation
8. Building AI Data Enrichment Workflows
9. n8n + RAG: Dynamic Knowledge Base Workflows
10. n8n + Agents: Autonomous Workflow Execution
11. n8n Security, Error Handling, and Monitoring
12. Enterprise AI Automation (Capstone)

**Prerequisites:** Volume 1 complete. Volume 3 (Agent Engineering) recommended.

---

## Volume 5 — RAG Deep Dive

**Objective:** Master every aspect of Retrieval Augmented Generation — from basic document ingestion to advanced hybrid search and multi-modal RAG.

**Planned Chapters:**
1. RAG Architecture Review: From Simple to Advanced
2. Document Ingestion: PDFs, HTML, DOCX, Code, and More
3. Chunking Strategies: Fixed, Semantic, Recursive, and Hybrid
4. Embedding Models: Choosing and Benchmarking
5. Sparse Search: BM25, TF-IDF, and Keyword Matching
6. Dense Search: Vector Similarity and ANN Algorithms
7. Hybrid Search: Combining Sparse and Dense Retrieval
8. Re-ranking: Cross-Encoders and LLM Re-ranking
9. Advanced Retrieval: HyDE, Multi-Query, and Step-Back
10. Contextual Compression and Result Filtering
11. Multi-Modal RAG: Images, Tables, and Charts
12. Agentic RAG: When the Agent Controls Retrieval (Capstone)

**Prerequisites:** Volume 1 Chapters 7 (Embeddings), 8 (Vector Databases), and 9 (RAG).

---

## Volume 6 — Vector Database Engineering

**Objective:** Deep expertise in vector database design, implementation, and production operation.

**Planned Chapters:**
1. Vector Database Internals: How HNSW Works
2. Pinecone: Production-Grade Managed Vector Search
3. Weaviate: Schema-Based Vector Search
4. Qdrant: High-Performance Self-Hosted Vector Search
5. pgvector: Adding Vector Search to PostgreSQL
6. ChromaDB: Lightweight Local and Cloud Vector Search
7. Indexing Strategies: HNSW vs IVF vs Flat
8. Filtering and Hybrid Search at Scale
9. Vector Database Performance Tuning
10. Multi-Tenancy in Vector Databases
11. Vector Database Backup, Migration, and Disaster Recovery
12. Choosing the Right Vector Database for Your Use Case

**Prerequisites:** Volume 1 Chapter 8 (Vector Databases). Volume 5 (RAG Deep Dive) recommended.

---

## Volume 7 — Coding Agents

**Objective:** Build autonomous coding agents capable of generating, testing, debugging, and deploying code.

**Planned Chapters:**
1. Coding Agents: The Architecture of an Autonomous Developer
2. Code Generation: Models, Prompts, and Quality
3. Code Execution: Safe Sandboxing with Docker
4. Test-Driven AI Development: Agents That Write Tests First
5. Debugging Agents: Autonomous Error Detection and Fixing
6. Code Review Agents: Automated Quality Gates
7. Refactoring Agents: Autonomous Codebase Improvement
8. Documentation Agents: Automated Code Documentation
9. Repository Agents: Understanding and Navigating Codebases
10. CI/CD Integration: Agents in the Build Pipeline
11. Security Scanning Agents: Automated Vulnerability Detection
12. Full Autonomous Development Loop (Capstone)

**Prerequisites:** Volume 1 complete. Volume 3 (Agent Engineering) strongly recommended.

---

## Volume 8 — DevOps AI

**Objective:** Apply AI Engineering to DevOps — build AI-powered infrastructure automation, incident response, and deployment systems.

**Planned Chapters:**
1. AI in the DevOps Lifecycle
2. AI-Powered Log Analysis and Anomaly Detection
3. Intelligent Alerting: Reducing Alert Fatigue with AI
4. AI-Powered Incident Response and Runbooks
5. Infrastructure as Code with AI Assistance
6. Kubernetes with AI: Intelligent Scaling and Scheduling
7. AI-Powered Security Scanning in CI/CD
8. Predictive Performance Monitoring
9. AI-Powered Cost Optimisation for Cloud Infrastructure
10. Autonomous Deployment Pipelines
11. AI Chaos Engineering: Intelligent Fault Injection
12. AI-Native DevOps Platform (Capstone)

**Prerequisites:** Volume 1 complete. Practical DevOps and Kubernetes experience recommended.

---

## Volume 9 — Technical Project Manager AI

**Objective:** For technical PMs who want to lead AI Engineering projects effectively — understanding requirements, scope, risk, estimation, and delivery.

**Planned Chapters:**
1. Managing AI Engineering Projects: What Makes Them Different
2. AI Project Discovery: Understanding Requirements and Scope
3. AI System Design for Non-Engineers: What to Know
4. Estimating AI Engineering Work
5. Managing AI Development Teams
6. AI Risk Management: Technical, Legal, and Ethical
7. AI Product Management: Metrics and Success Criteria
8. Stakeholder Communication for AI Projects
9. AI Project Delivery: Milestones and Quality Gates
10. Managing AI Vendors and API Providers
11. AI Governance and Compliance
12. Leading a Production AI System Launch (Capstone)

**Prerequisites:** Volume 1 Chapters 1–3 sufficient for this volume.

---

## Volume 10 — Enterprise AI Systems

**Objective:** Design and deploy AI systems at enterprise scale — multi-region, multi-tenant, compliant with enterprise security and governance requirements.

**Planned Chapters:**
1. Enterprise AI Architecture Patterns
2. Multi-Tenant AI Systems: Isolation, Billing, and Scaling
3. Enterprise Identity and Access Control for AI
4. Data Governance and Privacy for AI Systems
5. Enterprise AI Compliance: SOC 2, GDPR, HIPAA
6. Enterprise AI Security: Zero-Trust Architecture
7. On-Premises AI: Air-Gapped and Private Deployments
8. Enterprise AI Observability at Scale
9. Enterprise AI Cost Management and Chargeback
10. AI System Integration: ERP, CRM, and Enterprise Software
11. Enterprise AI Change Management and Adoption
12. Enterprise AI Reference Architecture (Capstone)

**Prerequisites:** Volume 1 complete. Enterprise software engineering experience recommended.

---

## Volume 11 — AI Architecture Patterns

**Objective:** A pattern library for AI Engineers — 40+ reusable architecture patterns for building AI systems.

**Planned Chapters:**
1. AI Architecture Patterns: An Introduction
2. Input Processing Patterns: Routing, Validation, and Transformation
3. Prompt Patterns: Templates, Chaining, and Composition
4. Output Patterns: Parsing, Validation, and Formatting
5. Memory Patterns: Short-Term, Long-Term, and Shared
6. Retrieval Patterns: RAG Variants and Hybrid Approaches
7. Agent Patterns: ReAct, Plan-and-Execute, and Beyond
8. Multi-Model Patterns: Routing, Ensemble, and Debate
9. Caching Patterns: Semantic, Exact, and Hierarchical
10. Reliability Patterns: Retry, Fallback, and Circuit Breaker
11. Observability Patterns: Tracing, Metrics, and Logging
12. Security Patterns: Defence-in-Depth for AI Systems

**Prerequisites:** Volume 1 complete. Recommended after any 2 other volumes.

---

## Volume 12 — Real Production Case Studies

**Objective:** Deep technical analysis of how real companies built, deployed, and scaled AI systems.

**Planned Case Studies:**
1. Building a Customer Support AI That Replaced Tier-1 Support
2. RAG at Scale: A Document Intelligence Platform for 10M Docs
3. Autonomous Coding Agent in a Production Engineering Team
4. Real-Time Fraud Detection with AI
5. AI-Powered Personalisation Engine for an E-Commerce Platform
6. Multi-Language AI Support System Across 30 Countries
7. AI Legal Document Review: Replacing Junior Associates
8. Healthcare AI: Diagnosis Assistance Under HIPAA
9. AI-Powered DevOps: Cutting Incident Response Time by 80%
10. Enterprise Chatbot: 50,000 Internal Users
11. AI Content Moderation at Social Media Scale
12. Post-Mortems: AI Projects That Failed and Why

**Prerequisites:** Volume 1 complete. Real-world engineering experience beneficial.

---

## Version Control for This Roadmap

This roadmap is intentionally high-level. Chapter titles, order, and scope will be refined as each volume is written. The primary constraint is that Volume 1 must be completely finished before any other volume begins.

Last updated: 2026-06-29
