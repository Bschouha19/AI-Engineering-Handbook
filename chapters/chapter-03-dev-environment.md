# Chapter 3: Setting Up Your AI Development Environment

---

> *"Give me six hours to chop down a tree, and I will spend the first four sharpening the axe."*
> — Attributed to Abraham Lincoln

---

## Learning Objectives

By the end of this chapter you will be able to:

- Explain what makes an AI development environment different from a standard developer setup
- Install and configure the three primary AI-powered code editors: VS Code, Cursor, and Claude Code CLI
- Choose the right AI coding assistant for your workflow using a principled decision framework
- Set up Python with uv and Node.js with nvm for isolated, reproducible AI project environments
- Store and load API keys safely using environment variables and `.env` files — and never commit them to git
- Run a complete local AI development stack using Docker Compose (Ollama + ChromaDB + Redis)
- Use Ollama and LM Studio to run AI models locally — free, private, and offline
- Build a reusable AI project template that any project can start from

---

## Prerequisites

- **Required:** Chapter 1 — What is AI Engineering (vocabulary, first API calls)
- **Required:** Chapter 2 — How LLMs Work (tokenisation, context windows)
- **Required:** A computer running macOS, Linux, or Windows 11 with at least 8 GB RAM (16 GB recommended for local models)

---

## Estimated Reading Time

**60 – 90 minutes**

---

## Estimated Hands-on Time

**3 – 5 hours** (installing and configuring everything, completing exercises)

---

## Table of Contents

1. [Why This Topic Exists](#1-why-this-topic-exists)
2. [The Real-World Analogy](#2-the-real-world-analogy)
3. [Core Concepts](#3-core-concepts)
4. [Architecture Diagrams](#4-architecture-diagrams)
5. [Flow Diagrams](#5-flow-diagrams)
6. [Beginner Implementation — The Minimal Setup](#6-beginner-implementation)
7. [Intermediate Implementation — The Professional Setup](#7-intermediate-implementation)
8. [Advanced Implementation — The Team Setup](#8-advanced-implementation)
9. [Production Architecture — Secrets at Scale](#9-production-architecture)
10. [Tool Comparisons & Decision Frameworks](#10-tool-comparisons)
11. [Best Practices](#11-best-practices)
12. [Security Considerations](#12-security-considerations)
13. [Cost Considerations](#13-cost-considerations)
14. [Common Mistakes](#14-common-mistakes)
15. [Debugging Guide](#15-debugging-guide)
16. [Performance Optimisation](#16-performance-optimisation)
17. [Exercises](#17-exercises)
18. [Quiz](#18-quiz)
19. [Mini Project](#19-mini-project)
20. [Production Project](#20-production-project)
21. [Key Takeaways](#21-key-takeaways)
22. [Chapter Summary](#22-chapter-summary)
23. [Resources](#23-resources)
24. [Glossary Terms Introduced](#24-glossary-terms-introduced)
25. [See Also](#25-see-also)
26. [Preparation for Chapter 4](#26-preparation-for-chapter-4)

---

## 1. Why This Topic Exists

A standard developer workspace — your usual editor, terminal, and package manager — is not wrong for AI Engineering. It is simply incomplete.

AI Engineering introduces requirements that most developer environments were not designed to handle:

**1. You write code that calls AI models.** Those models are non-deterministic. When your output is different every run, you need tools that help you observe, compare, and evaluate outputs — not just check if the program crashed.

**2. You need multiple API providers configured simultaneously.** Claude for reasoning, GPT-4o for comparison testing, Gemini for multimodal, Ollama for free local testing during development. Each requires credentials, each has a different SDK.

**3. You write prompts as well as code.** The best AI development tools understand that a system prompt is as important as a function definition. They highlight it, version it, and help you improve it.

**4. You run local AI models during development.** To iterate quickly without burning API credits, you need a local model runner that starts in seconds and serves a compatible API.

**5. Your projects have more secrets than a typical app.** Three to five API keys per project is common. A leaked key can cost hundreds of dollars in minutes. Secret management must be automatic and enforced from day one.

**6. You build against APIs that change.** Model versions deprecate. SDKs release breaking changes. Your environment needs to help you catch these quickly, not hide them.

Getting the environment right once takes 3–5 hours. Getting it wrong costs far more than that across every project you work on.

---

## 2. The Real-World Analogy

### The Surgeon's Operating Room

A general doctor's consultation room has a desk, a computer, and basic equipment. That is sufficient for consultations. But a surgeon walking into a consultation room to perform surgery would be missing everything they need: the operating table, the lights, the instrument trays, the monitoring equipment, the anaesthesia machine.

The surgeon's operating room is not more complex for the sake of complexity. Every piece of equipment exists because surgery has requirements that a consultation does not.

An AI Engineer's development environment works the same way. Your general developer setup (VS Code, npm, Python, git) is the consultation room. Excellent for what it does. But AI engineering adds surgery-level requirements: a local model runner, a vector database for testing RAG, a multi-provider API manager, an AI-powered editor that understands AI code, and secret management that is enforced rather than optional.

### The Workshop Rule

Professional tradespeople have a second rule that applies here: **set up your workshop before starting the job.** A carpenter who has to stop and find tools in the middle of cutting timber loses flow, makes mistakes, and takes twice as long.

Set up your AI development environment completely before starting a project. The 3–5 hours you invest today saves 2–3 hours per project, every project.

---

## 3. Core Concepts

### IDE (Integrated Development Environment)

**Technical definition:** A software application that provides a complete environment for writing, testing, and debugging code — including an editor, file browser, terminal, debugger, and extension system.

**Simple definition:** Your main workspace. The application you spend most of your development time inside.

**Analogy:** Your desk. Everything you need to work is arranged on it or within reach. The quality of your desk setup directly affects how productive you are.

**In AI Engineering:** The IDE must support AI-powered code completion that understands AI APIs, have a built-in terminal for running models and experiments, and ideally have a chat panel where you can ask questions about your own codebase.

---

### AI Code Assistant

**Technical definition:** An AI-powered extension or application that integrates with your IDE to provide code suggestions, completions, chat, and autonomous code editing capabilities.

**Simple definition:** An AI that helps you write code while you are writing it — completing lines, suggesting fixes, explaining unfamiliar APIs, and generating boilerplate.

**Analogy:** A very experienced pair programmer who sits next to you and has memorised every API documentation ever written. They do not take over; they help when you want help.

**The key distinction:** There are two types:
- **Completion-based assistants** (GitHub Copilot, Tabnine): They predict what you are about to type and complete it, like a very smart autocomplete.
- **Agent-based assistants** (Cursor Agent, Claude Code): They can read your entire codebase, understand your intent, and make multi-file changes autonomously.

---

### Package Manager

**Technical definition:** A tool that installs, updates, and manages software libraries (packages) that your project depends on, along with their transitive dependencies.

**Simple definition:** The thing you use to install libraries. `pip install anthropic` and `npm install @anthropic-ai/sdk` both go through package managers.

**Why it matters for AI Engineering:** AI projects accumulate dependencies quickly — the AI SDK, the vector database client, the tokeniser, the web framework, the HTTP client. A good package manager keeps these isolated (so project A and project B do not conflict), reproducible (so your colleague's machine matches yours exactly), and fast (so you are not waiting 10 minutes to install packages).

---

### Virtual Environment

**Technical definition:** An isolated Python environment that contains its own Python interpreter and package installation, separate from the system Python and other projects.

**Simple definition:** A clean box for each project's Python packages. Project A has anthropic v1.0, project B has anthropic v2.0 — they do not conflict because each is in its own box.

**Analogy:** Separate toolboxes for different jobs. The kitchen toolbox has cooking tools. The garage toolbox has automotive tools. Keeping them separate means you always know what's in each box and nothing gets contaminated.

---

### Environment Variables

**Technical definition:** Key-value pairs stored in the operating system (or in a `.env` file loaded at startup) that provide configuration values to a running process without hardcoding them in source code.

**Simple definition:** Configuration that lives outside your code. API keys, database URLs, feature flags — anything that differs between environments (development, staging, production) or that should not be visible in source control.

**Why they exist:** You cannot commit an API key to a public git repository without someone finding it and billing you for their AI usage. Environment variables keep secrets out of code.

**The `.env` file:** A plain text file in your project root that contains environment variables for local development. It is listed in `.gitignore` so it is never committed to git. Your colleagues each have their own copy with their own keys.

---

### Containerisation

**Technical definition:** Packaging an application and all its dependencies into a container (a self-contained unit) that runs identically on any machine regardless of the host operating system or installed software.

**Simple definition:** "It works on my machine" solved. A container includes your application, its exact runtime version, all its libraries, and its configuration — wrapped into a single portable unit.

**Analogy:** Shipping containers. Before standardised shipping containers, every cargo ship had to be loaded and unloaded differently for each port. With containers, the format is standardised — any crane at any port can handle it. Docker does the same for software.

**Docker Compose:** A tool for defining and running multi-container applications. For AI development, this means running your app, Ollama (local model), ChromaDB (vector database), and Redis (cache) as a single coordinated stack with one command.

---

### Local Model Runner

**Technical definition:** Software that downloads, manages, and serves open-source AI models on your local hardware, exposing an API that your code can call exactly as it would call a cloud provider's API.

**Simple definition:** Run AI models on your own computer — free, private, offline, no API keys required.

**Why it exists:** During development you iterate constantly. Running every test through a cloud API costs money and requires an internet connection. A local model runner lets you develop and test for free, then switch to a cloud model for production quality.

**The two options:**
- **Ollama:** Terminal-based, scriptable, Docker-compatible. Best for developers.
- **LM Studio:** GUI-based, beginner-friendly, built-in model browser. Best for exploring models visually.

---

## 4. Architecture Diagrams

### 4.1 The Complete AI Development Stack

```mermaid
graph TB
    subgraph "Your Machine"
        subgraph "Editor Layer"
            VS[VS Code<br/>General-purpose editor]
            CUR[Cursor<br/>AI-native editor]
        end

        subgraph "AI Assistant Layer"
            COP[GitHub Copilot<br/>Completions + Chat]
            CC[Claude Code CLI<br/>Terminal agent]
            CON[Continue.dev<br/>Open-source assistant]
            AG[Antigravity CLI<br/>Google terminal agent]
        end

        subgraph "Runtime Layer"
            UV[uv<br/>Python + packages]
            NVM[nvm<br/>Node.js versions]
            GIT[git<br/>Version control]
        end

        subgraph "Local AI Layer (Docker Compose)"
            OL[Ollama<br/>Local model server<br/>:11434]
            LMS[LM Studio<br/>GUI model runner<br/>:1234]
            CHROMA[ChromaDB<br/>Vector database<br/>:8000]
            REDIS[Redis<br/>Cache + rate limit<br/>:6379]
        end

        subgraph "Secrets Layer"
            ENV[.env file<br/>Development keys]
            GI[.gitignore<br/>Never commit .env]
        end
    end

    subgraph "Cloud APIs (when needed)"
        ANT[Anthropic<br/>Claude]
        OAI[OpenAI<br/>GPT-4o]
        GOO[Google AI<br/>Gemini]
    end

    CUR --> CC
    UV --> OL
    UV --> ANT
    UV --> OAI
    UV --> GOO
    ENV --> UV
```

### 4.2 Development Topology — How Tools Connect

```mermaid
graph LR
    DEV[Developer]

    DEV -->|writes code in| EDITOR[Cursor / VS Code]
    DEV -->|runs commands in| TERM[Terminal]

    EDITOR -->|AI completions| COP[GitHub Copilot]
    EDITOR -->|agent tasks| CUR_AI[Cursor Agent]
    TERM -->|agent tasks| CC_CLI[Claude Code CLI]
    TERM -->|local models| OL_API[Ollama API :11434]
    TERM -->|Docker stack| DC[Docker Compose]

    DC -->|starts| OL_C[Ollama container]
    DC -->|starts| CHROMA_C[ChromaDB container]
    DC -->|starts| REDIS_C[Redis container]

    EDITOR -->|reads from| ENV[.env file]
    ENV -->|never goes to| GIT[git repository]
```

---

## 5. Flow Diagrams

### 5.1 Development Workflow With AI Tools

```mermaid
flowchart TD
    START[Start new AI feature] --> LOC{Use local model<br/>during dev?}
    LOC -->|Yes| OLL[ollama run llama3.2<br/>Free, fast, private]
    LOC -->|No| CLOUD[Use cloud API<br/>with .env key]

    OLL --> CODE[Write code in Cursor/VS Code]
    CLOUD --> CODE

    CODE --> COP_H{Want AI help?}
    COP_H -->|Completion| TAB[Tab-complete with<br/>Copilot/Cursor]
    COP_H -->|Complex task| AGT[Cursor Agent or<br/>Claude Code CLI]

    TAB --> TEST[Run/test code]
    AGT --> TEST

    TEST --> PASS{Works?}
    PASS -->|No| DEBUG[Debug with AI chat<br/>in editor]
    PASS -->|Yes| SWITCH{Switch to<br/>production model?}

    DEBUG --> CODE
    SWITCH -->|Yes| PROD_API[Replace Ollama URL<br/>with cloud API in .env]
    SWITCH -->|No| DONE[Feature complete]
    PROD_API --> DONE
```

### 5.2 Docker Compose Local AI Stack

```mermaid
flowchart LR
    CMD["docker compose up -d"]
    CMD --> OL[Ollama\n:11434\nLocal LLM server]
    CMD --> CH[ChromaDB\n:8000\nVector database]
    CMD --> RE[Redis\n:6379\nCache + queues]

    APP[Your Python/Node app] -->|AI calls| OL
    APP -->|Vector search| CH
    APP -->|Caching| RE

    OL -->|pull models| ML[Model library\nllama3.2, qwen3:8b...]
```

---

## 6. Beginner Implementation

### The Minimal Setup — Get Running in 30 Minutes

This setup is enough to start building AI applications. It uses free tools and one API key.

```
What you will have after this section:
✓ VS Code with GitHub Copilot
✓ Python 3.12 with pip
✓ Node.js 22 LTS
✓ One Anthropic API key loaded safely
✓ A first working API call verified
```

#### Step 1 — Install VS Code

Download from code.visualstudio.com. During installation, check "Add to PATH" on Windows so you can open VS Code from the terminal with `code .`.

#### Step 2 — Install GitHub Copilot in VS Code

1. Open VS Code
2. Press `Cmd+Shift+X` (macOS) or `Ctrl+Shift+X` (Windows/Linux) to open the Extensions panel
3. Search "GitHub Copilot" and install the **GitHub Copilot** extension (by GitHub)
4. Sign in with your GitHub account
5. The free tier (Hobby plan) gives you limited completions and chat — enough to start

> **Note:** GitHub Copilot pricing changed to usage-based billing in June 2026. The free tier includes basic completions. Pro is $10/month and includes $10 in monthly AI Credits. Check [github.com/features/copilot/plans](https://github.com/features/copilot/plans) for current pricing before choosing a plan.

#### Step 3 — Install Python

The recommended approach in 2026 is to install Python through **uv**, which manages Python versions alongside packages. But for the minimal setup, installing Python directly is fine.

**macOS:**
```bash
# Using Homebrew (install Homebrew first at brew.sh if needed)
brew install python@3.12

# Verify
python3 --version   # Should print Python 3.12.x
```

**Windows:**
Download the installer from python.org/downloads. During installation, check "Add Python to PATH."

```bash
# Verify in Command Prompt
python --version
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update && sudo apt install python3.12 python3.12-venv python3-pip -y
python3 --version
```

#### Step 4 — Install Node.js

Download the LTS version (22.x) from nodejs.org/en/download.

```bash
# Verify
node --version    # Should print v22.x.x
npm --version     # Should print 10.x.x
```

#### Step 5 — Get Your Anthropic API Key

1. Go to console.anthropic.com
2. Create an account (free)
3. Navigate to API Keys → Create Key
4. Copy the key immediately — it is only shown once

> **Free tier note:** Anthropic does not offer a permanently free API tier. You will need to add a credit card. New accounts receive a small credit for getting started. Running Chapter exercises with Claude Haiku costs fractions of a cent.

**Free alternatives:**
- Google Gemini at aistudio.google.com offers a free tier with generous daily limits — excellent for learning
- Ollama (Section 7) is completely free and runs locally

#### Step 6 — Create Your First Project

```bash
# Create project directory
mkdir my-first-ai-app
cd my-first-ai-app

# Create Python virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows

# Install the Anthropic SDK
pip install anthropic python-dotenv
```

#### Step 7 — Create Your `.env` File

```bash
# Create the file (do NOT put spaces around the = sign)
touch .env
```

Open `.env` in VS Code and add your key:

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

Immediately create `.gitignore`:

```bash
# .gitignore
.env
.venv/
__pycache__/
*.pyc
.DS_Store
node_modules/
```

> **Critical:** Add `.env` to `.gitignore` BEFORE your first `git commit`. If you commit `.env` even once, the key is in your git history forever. Deleting the file does not remove it from history.

#### Step 8 — Write Your First API Call

```python
# first_call.py
# Load API key from .env file
from dotenv import load_dotenv
import anthropic

load_dotenv()  # Reads .env file and sets environment variables

client = anthropic.Anthropic()  # Automatically reads ANTHROPIC_API_KEY from environment

response = client.messages.create(
    model="claude-haiku-4-5-20251001",  # Cheapest model — good for testing
    max_tokens=200,
    messages=[
        {"role": "user", "content": "Say hello and tell me your model name."}
    ]
)

print(response.content[0].text)
print(f"\nTokens used: {response.usage.input_tokens} input, {response.usage.output_tokens} output")
```

```bash
python first_call.py
```

If you see a response, your minimal setup is working.

#### The Same in Node.js

```bash
npm init -y
npm install @anthropic-ai/sdk dotenv
```

```javascript
// first-call.mjs
import Anthropic from "@anthropic-ai/sdk";
import "dotenv/config";

const client = new Anthropic();  // Reads ANTHROPIC_API_KEY from .env

const response = await client.messages.create({
  model: "claude-haiku-4-5-20251001",
  max_tokens: 200,
  messages: [{ role: "user", content: "Say hello and tell me your model name." }],
});

console.log(response.content[0].text);
console.log(`Tokens: ${response.usage.input_tokens} in, ${response.usage.output_tokens} out`);
```

```bash
node first-call.mjs
```

---

## 7. Intermediate Implementation

### The Professional Setup — The Right Tools for Serious Work

This is the setup that professional AI Engineers use daily. It adds proper package management, AI-powered editors, local model running, and a solid project structure.

```
What you will have after this section:
✓ Cursor (AI-native editor) configured
✓ Claude Code CLI installed
✓ Antigravity CLI installed (Google's terminal AI agent)
✓ Python managed by uv (replaces pip + venv + pyenv)
✓ Node.js managed by nvm (replaces manual installs)
✓ Ollama running local models
✓ LM Studio for visual model exploration
✓ A complete AI project template
```

#### Install uv — The Modern Python Package Manager

uv is a Python package manager written in Rust. It is 10–100× faster than pip and replaces pip, venv, and pyenv in a single tool. It is now the standard for AI Engineering projects.

```bash
# macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# macOS with Homebrew
brew install uv

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Verify
uv --version
```

**The key uv commands you will use every day:**

```bash
# Create a new project (replaces: mkdir + python -m venv + pip init)
uv init my-ai-project
cd my-ai-project

# Add a package (replaces: pip install)
uv add anthropic
uv add python-dotenv

# Add a development dependency (not included in production installs)
uv add --dev pytest ruff

# Run a script in the project environment (no manual activation needed)
uv run python main.py

# Install a specific Python version
uv python install 3.12

# Sync environment with pyproject.toml (replaces: pip install -r requirements.txt)
uv sync

# Self-update
uv self update
```

**What uv creates for you:**

```
my-ai-project/
├── pyproject.toml      ← Project metadata and dependencies (replaces requirements.txt)
├── uv.lock             ← Exact locked versions for reproducibility
├── .python-version     ← Python version pin for this project
├── .venv/              ← Virtual environment (auto-created, never commit)
└── main.py             ← Starting point
```

> **Why uv instead of pip?** Three reasons: speed (10–100× faster installs), correctness (resolves conflicts that pip misses), and integration (manages Python versions too). Once you use uv you will not go back to pip for project work.

#### Install nvm — Node.js Version Manager

nvm lets you switch between Node.js versions per project. Essential when different AI projects require different Node versions.

```bash
# macOS and Linux
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash

# Restart your terminal, then:
nvm install 22          # Install Node.js 22 LTS
nvm use 22              # Use it for this session
nvm alias default 22    # Make it the default

# Verify
node --version   # v22.x.x
```

**Windows:** Use nvm-windows from github.com/coreybutler/nvm-windows (different project, same concept).

#### Install Cursor — The AI-Native Editor

Cursor is a fork of VS Code with AI deeply integrated. It understands your entire codebase, not just the current file. This makes it far more useful for AI Engineering, where you need the AI to understand how your prompt templates relate to your API calls.

Download from cursor.com.

After installing, open Cursor and configure it:

```bash
# Open a project from the terminal
cursor .

# Or open from Cursor: File → Open Folder
```

**First-time Cursor configuration:**

1. Sign in (free Hobby plan or Pro $20/mo)
2. Open Settings (`Cmd/Ctrl + ,`), search for "models"
3. Enable the models you want: Claude claude-sonnet-4-6, GPT-4o, Gemini (uses your own API keys or Cursor's built-in credits depending on your plan)
4. Add your `.env` keys so Cursor's terminal inherits them
5. Try: `Cmd+K` (inline edit), `Cmd+L` (chat), `Cmd+Shift+I` (agent mode)

**Cursor-specific settings for AI Engineering projects** (add to `.cursorrules` in your project root):

```
You are an AI Engineer working on a production AI system.

When writing Python code:
- Use uv for package management
- Use the anthropic library for Claude calls
- Always load environment variables with python-dotenv
- Include proper error handling and logging
- Type-hint all function signatures

When writing Node.js code:
- Use ES modules (import/export, not require)
- Use @anthropic-ai/sdk for Claude calls
- Load .env with import "dotenv/config" at the top

This project uses:
- Python 3.12
- Node.js 22
- Claude claude-haiku-4-5-20251001 for fast/cheap calls
- Claude claude-sonnet-4-6 for quality-critical tasks
```

#### Install Claude Code CLI

Claude Code is Anthropic's terminal-based AI agent. Unlike Cursor (which works inside an editor), Claude Code runs from your terminal and can work on your files, run commands, search the web, and execute multi-step tasks autonomously.

```bash
# Install globally (requires Node.js 18+)
npm install -g @anthropic-ai/claude-code

# IMPORTANT: Never use sudo with npm install
# If you get permission errors, fix them by using nvm instead

# Verify
claude --version

# Start Claude Code in your project
cd my-ai-project
claude
```

**Basic Claude Code usage:**

```bash
# Start a session
claude

# Common tasks you can ask:
# "Add error handling to all the API calls in this project"
# "Write tests for the functions in main.py"
# "Find all places where I'm not handling rate limit errors"
# "Create a README for this project"
# "Explain what this codebase does"

# Non-interactive mode (for scripts/CI)
claude -p "What does main.py do?" --output-format json
```

> **Note:** Claude Code requires an Anthropic API key or a Claude.ai Pro/Max subscription. The CLI uses your `ANTHROPIC_API_KEY` environment variable, or you can log in interactively with `claude auth login`.

#### Install Antigravity CLI — Google's Terminal AI Agent

> **Important update (June 2026):** Google's Gemini CLI was retired for individual users on June 18, 2026, and replaced by Antigravity CLI — a Go-based terminal agent. If you previously used `gemini` in your terminal, you will need to migrate.
>
> Enterprise users with Gemini Code Assist Standard/Enterprise licenses are unaffected. For individuals, Antigravity CLI is the current tool. The free tier offers approximately 20 requests per day. Check [antigravity.google](https://antigravity.google) for current limits.

```bash
# macOS and Linux
curl -fsSL https://antigravity.google/cli/install.sh | bash

# Windows (PowerShell)
irm https://antigravity.google/cli/install.ps1 | iex

# Verify
agy --version

# Start a session
agy

# The command is 'agy' (Antigravity), not 'gemini'
```

**Antigravity CLI keeps the most useful features from Gemini CLI:** agent skills, hooks, subagents, and extensions (now called plugins). It is faster and supports asynchronous background workflows.

**For Gemini Code Assist in VS Code:** The IDE extension is a separate product and remains free for individuals at its current tier. Search "Gemini Code Assist" in the VS Code extensions panel.

#### Install Ollama — Free Local AI Models

Ollama lets you run open-source AI models on your own hardware. During development, use Ollama instead of cloud APIs — it is completely free, works offline, and has zero latency for small tasks.

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Docker (works on all platforms)
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

# Verify Ollama is running
ollama list          # Show installed models
curl http://localhost:11434/api/tags   # API check
```

**Download and run models:**

```bash
# Pull a model (download it to your machine)
ollama pull llama3.2           # Meta's Llama 3.2 3B — fast, good for basic tasks (~2 GB)
ollama pull qwen3:8b           # Alibaba Qwen3 8B — excellent reasoning (~5 GB)
ollama pull nomic-embed-text   # Embedding model for RAG (~274 MB)

# Run a model interactively
ollama run llama3.2

# Check what's running
ollama ps

# Remove a model (free disk space)
ollama rm llama3.2
```

**Call Ollama from Python (same API as cloud models):**

```python
# ollama_example.py
# Ollama exposes an OpenAI-compatible API at http://localhost:11434
from openai import OpenAI

# Use the OpenAI SDK pointed at Ollama — no API key required
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Any string works — Ollama doesn't validate keys
)

response = client.chat.completions.create(
    model="llama3.2",      # Must be a model you've pulled
    max_tokens=200,
    messages=[{"role": "user", "content": "Write a one-sentence summary of what RAG is."}]
)

print(response.choices[0].message.content)
```

**Call Ollama from Node.js:**

```javascript
// ollama-example.mjs
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:11434/v1",
  apiKey: "ollama",
});

const response = await client.chat.completions.create({
  model: "llama3.2",
  max_tokens: 200,
  messages: [{ role: "user", content: "What is a context window in one sentence?" }],
});

console.log(response.choices[0].message.content);
```

> **Pro tip:** Ollama uses the same OpenAI-compatible API that most AI applications support. By pointing your app at `http://localhost:11434/v1` instead of `https://api.openai.com/v1`, you get free development without changing your code structure.

#### Install LM Studio — Visual Model Management

LM Studio is a free desktop application for running AI models locally with a graphical interface. Unlike Ollama (which is terminal-based), LM Studio is point-and-click — you browse models, download them, and chat with them without writing any code.

Download from lmstudio.ai (free, no account required).

**What LM Studio offers:**
- Built-in model browser connected to Hugging Face (8,000+ models)
- One-click download with RAM/VRAM requirements shown before download
- Chat interface for manual testing
- **Developer mode:** Starts a local server on port 1234 with an OpenAI-compatible API

```python
# Use LM Studio's server from Python (same pattern as Ollama)
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:1234/v1",  # LM Studio's server
    api_key="lm-studio"
)

response = client.chat.completions.create(
    model="whatever-model-is-loaded",  # Use whatever you have loaded in LM Studio
    messages=[{"role": "user", "content": "Hello"}]
)
print(response.choices[0].message.content)
```

#### Build Your AI Project Template

This is the folder structure every new AI project should start from. Having this ready means every new project gets the full professional setup in 30 seconds.

```
ai-project-template/
├── .env.example            ← Template for env vars (commit this)
├── .env                    ← Your actual keys (never commit)
├── .gitignore              ← Excludes .env, .venv, __pycache__
├── .cursorrules            ← Cursor AI instructions for this project
├── pyproject.toml          ← Python project config (via uv init)
├── uv.lock                 ← Locked dependency versions
├── package.json            ← Node.js dependencies
├── docker-compose.yml      ← Local AI dev stack
├── Makefile                ← Common commands (make dev, make test)
├── README.md               ← Setup instructions
├── src/
│   ├── __init__.py
│   ├── main.py             ← Application entry point
│   ├── config.py           ← Configuration loading
│   ├── ai/
│   │   ├── __init__.py
│   │   ├── client.py       ← AI client initialisation
│   │   └── prompts.py      ← Prompt templates
│   └── utils/
│       ├── __init__.py
│       └── tokens.py       ← Token counting utilities
└── tests/
    ├── __init__.py
    └── test_ai.py          ← AI-specific tests
```

**`.env.example` — the template that IS committed to git:**

```bash
# .env.example
# Copy this to .env and fill in your own values
# NEVER commit .env

# AI Provider Keys (get at least one)
ANTHROPIC_API_KEY=         # console.anthropic.com
OPENAI_API_KEY=            # platform.openai.com
GOOGLE_AI_API_KEY=         # aistudio.google.com

# Local AI (no keys needed)
OLLAMA_BASE_URL=http://localhost:11434/v1
LM_STUDIO_BASE_URL=http://localhost:1234/v1

# App settings
LOG_LEVEL=DEBUG
MAX_TOKENS=2048
DEFAULT_MODEL=claude-haiku-4-5-20251001
```

**`config.py` — loads all config from environment:**

```python
# src/config.py
from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    # AI Provider Keys
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    GOOGLE_AI_API_KEY = os.environ.get("GOOGLE_AI_API_KEY")

    # Local AI
    OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    LM_STUDIO_BASE_URL = os.environ.get("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")

    # App settings
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
    MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "2048"))
    DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "claude-haiku-4-5-20251001")

    def validate(self):
        """Fail fast if critical config is missing."""
        if not any([self.ANTHROPIC_API_KEY, self.OPENAI_API_KEY, self.GOOGLE_AI_API_KEY]):
            raise ValueError(
                "No AI provider key found. Set at least one of: "
                "ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_AI_API_KEY in your .env file"
            )

config = Config()
```

**`ai/client.py` — a single client factory for all providers:**

```python
# src/ai/client.py
import anthropic
import openai
from src.config import config


def get_anthropic_client() -> anthropic.Anthropic:
    if not config.ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not set in .env")
    return anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)


def get_openai_client() -> openai.OpenAI:
    if not config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set in .env")
    return openai.OpenAI(api_key=config.OPENAI_API_KEY)


def get_ollama_client() -> openai.OpenAI:
    """Ollama uses the OpenAI SDK pointed at localhost — no API key needed."""
    return openai.OpenAI(
        base_url=config.OLLAMA_BASE_URL,
        api_key="ollama"
    )


def get_local_client() -> openai.OpenAI:
    """Use Ollama if available, fall back to LM Studio."""
    return openai.OpenAI(
        base_url=config.OLLAMA_BASE_URL,
        api_key="local"
    )
```

**`Makefile` — common commands:**

```makefile
# Makefile
.PHONY: install dev test lint clean

install:
	uv sync

dev:
	docker compose up -d
	uv run python src/main.py

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check src/
	uv run ruff format src/

clean:
	docker compose down
	find . -type d -name __pycache__ -exec rm -rf {} +
```

---

## 8. Advanced Implementation

### The Team Setup — Docker Compose Local AI Stack

When working in a team, or when you want a consistent local environment that mirrors production, use Docker Compose to run your entire AI stack.

This gives every developer on the team — and every CI runner — the same exact environment.

#### Prerequisites

Install Docker Desktop from docker.com/products/docker-desktop (free for individual use).

```bash
# Verify
docker --version          # Docker 27+
docker compose version    # Docker Compose v2+
```

#### The Complete `docker-compose.yml`

```yaml
# docker-compose.yml
# Local AI development stack
# Run: docker compose up -d
# Stop: docker compose down

services:

  # ── Local LLM Server ──────────────────────────────────────────────────────
  ollama:
    image: ollama/ollama:latest
    container_name: ai_ollama
    volumes:
      - ollama_data:/root/.ollama    # Models persist across restarts
    ports:
      - "11434:11434"
    restart: unless-stopped
    # For GPU acceleration on Linux with NVIDIA:
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: all
    #           capabilities: [gpu]

  # ── Model Puller (runs once, then exits) ──────────────────────────────────
  ollama-init:
    image: ollama/ollama:latest
    container_name: ai_ollama_init
    depends_on:
      - ollama
    entrypoint: >
      /bin/sh -c "
        sleep 5 &&
        ollama pull llama3.2 &&
        ollama pull nomic-embed-text &&
        echo 'Models ready'
      "
    environment:
      - OLLAMA_HOST=ollama:11434
    volumes:
      - ollama_data:/root/.ollama

  # ── Vector Database ───────────────────────────────────────────────────────
  chromadb:
    image: chromadb/chroma:latest
    container_name: ai_chromadb
    ports:
      - "8000:8000"
    volumes:
      - chroma_data:/chroma/chroma
    environment:
      - ALLOW_RESET=true              # Useful during development
      - ANONYMIZED_TELEMETRY=false
    restart: unless-stopped

  # ── Redis (Cache + Rate Limiting + Queues) ────────────────────────────────
  redis:
    image: redis:7-alpine
    container_name: ai_redis
    ports:
      - "6379:6379"
    command: redis-server --save 60 1 --loglevel warning
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  ollama_data:
  chroma_data:
  redis_data:
```

**Running the stack:**

```bash
# Start everything in the background
docker compose up -d

# Check status
docker compose ps

# View logs for a specific service
docker compose logs -f ollama

# Pull additional models
docker compose exec ollama ollama pull qwen3:8b

# Stop the stack
docker compose down

# Stop and remove all data (fresh start)
docker compose down -v
```

**Updated `.env.example` for team use:**

```bash
# .env.example — commit this file
# Copy to .env and fill in your own values

# AI Provider Keys (at least one required for cloud features)
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
GOOGLE_AI_API_KEY=

# Local AI Stack (Docker Compose — no keys needed)
OLLAMA_BASE_URL=http://localhost:11434/v1
CHROMADB_HOST=http://localhost:8000
REDIS_URL=redis://localhost:6379

# App Configuration
DEFAULT_MODEL=claude-haiku-4-5-20251001
LOCAL_MODEL=llama3.2
EMBEDDING_MODEL=nomic-embed-text
LOG_LEVEL=DEBUG
MAX_TOKENS=2048
```

#### VS Code Dev Container

A Dev Container runs your entire development environment inside Docker. Any developer who clones the repo gets the exact same setup with one click — no manual tool installation.

Create `.devcontainer/devcontainer.json`:

```json
{
  "name": "AI Engineering Dev",
  "dockerComposeFile": "../docker-compose.yml",
  "service": "app",
  "workspaceFolder": "/workspace",
  "features": {
    "ghcr.io/devcontainers/features/python:1": {
      "version": "3.12"
    },
    "ghcr.io/devcontainers/features/node:1": {
      "version": "22"
    }
  },
  "postCreateCommand": "curl -LsSf https://astral.sh/uv/install.sh | sh && uv sync",
  "customizations": {
    "vscode": {
      "extensions": [
        "GitHub.copilot",
        "Continue.continue",
        "ms-python.python",
        "charliermarsh.ruff"
      ],
      "settings": {
        "python.defaultInterpreterPath": ".venv/bin/python"
      }
    }
  }
}
```

To use: open VS Code → `Cmd/Ctrl+Shift+P` → "Dev Containers: Reopen in Container".

---

## 9. Production Architecture

### Secrets at Scale — Beyond `.env` Files

The `.env` file pattern is excellent for local development. It is not acceptable for production. Here is why and what to use instead.

**Why `.env` fails in production:**
- Containers and Kubernetes pods cannot easily read `.env` files
- Multiple instances of your app (load balancing) need the same secrets synchronised
- You cannot audit who accessed a secret or rotate it without redeployment
- Secrets injected through CI/CD must be stored somewhere — and that somewhere needs to be secure

**Production secrets management options:**

| Solution | Best For | Cost | Complexity |
|----------|---------|------|------------|
| **AWS Secrets Manager** | AWS deployments | ~$0.40/secret/month + $0.05/10K API calls | Low |
| **GCP Secret Manager** | GCP deployments | First 6 secrets free, then $0.06/secret/month | Low |
| **Azure Key Vault** | Azure deployments | $0.03/10K operations | Low |
| **HashiCorp Vault** | Multi-cloud / self-hosted | Free OSS, paid enterprise | High |
| **Doppler** | Any cloud, developer-friendly | Free tier, $10/mo Pro | Low |
| **Vercel / Railway secrets** | If using those platforms | Included with hosting | Very low |

**Python — reading from AWS Secrets Manager:**

```python
# secrets.py — production pattern
import boto3
import json
from functools import lru_cache

@lru_cache(maxsize=None)  # Cache to avoid repeated API calls
def get_secret(secret_name: str) -> dict:
    """
    Fetch a secret from AWS Secrets Manager.
    In production, the app role has IAM permissions to call this.
    No credentials needed in the code — they come from the instance role.
    """
    client = boto3.client("secretsmanager", region_name="us-east-1")
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response["SecretString"])


def get_ai_keys() -> dict:
    """Load AI API keys from Secrets Manager in production."""
    return get_secret("prod/ai-engineering/api-keys")

# Usage
keys = get_ai_keys()
anthropic_client = anthropic.Anthropic(api_key=keys["ANTHROPIC_API_KEY"])
```

**The environment parity rule:**

```
Local Dev:    .env file (manual secrets, Docker Compose services)
Staging:      Secrets Manager + staging API keys + staging services
Production:   Secrets Manager + production API keys + production services
```

The only difference between environments should be the values of secrets and URLs — not the code, not the structure. If your local code reads `os.environ["ANTHROPIC_API_KEY"]`, your production code reads the same variable — it just comes from Secrets Manager instead of a file.

---

## 10. Tool Comparisons & Decision Frameworks

> **Note:** Pricing and features verified June 2026. AI tools evolve rapidly — confirm current details at each provider's website before making purchasing decisions.

### VS Code vs Cursor

Both are excellent. Your choice should be based on how much you want AI integration in your editor.

| Dimension | VS Code + Copilot | Cursor |
|-----------|-------------------|--------|
| **Base cost** | Free | Free (Hobby) |
| **AI tier cost** | Copilot: $10/mo (Pro) | Cursor: $20/mo (Pro) |
| **Codebase understanding** | File-level context | Full codebase indexed |
| **AI model choice** | Mostly GitHub-managed | Claude, GPT-4o, Gemini |
| **Agent capability** | Copilot agent mode | Full Cursor Agent |
| **MCP support** | Limited via extensions | Full MCP support (Pro+) |
| **Open source** | Yes (VS Code) | No (proprietary fork) |
| **Extension ecosystem** | Massive (40K+ extensions) | Inherits VS Code extensions |
| **Privacy mode** | GitHub manages | Available on Pro |
| **Learning curve** | Low (already know VS Code?) | Low (same as VS Code) |
| **Best for** | Teams already on VS Code, budget-conscious | AI-first development, agent workflows |

**Decision rule:**
- Already using VS Code and GitHub? → Add Copilot and stay in VS Code.
- Starting fresh or want the strongest AI coding experience? → Start with Cursor.
- Both are valid. Many engineers use Cursor for AI projects and VS Code for everything else.

### AI Terminal Agents

| Tool | Company | Command | Free Tier | Paid | Best For |
|------|---------|---------|-----------|------|---------|
| **Claude Code** | Anthropic | `claude` | Via API key | Via API key (per token) | Deep Anthropic integration, autonomous tasks |
| **Antigravity CLI** | Google | `agy` | ~20 req/day | Gemini Code Assist license | Google ecosystem, Go-based speed |
| **Cursor Agent** | Cursor | In-editor | Hobby plan | Pro $20/mo | In-editor agent tasks |
| **Codex CLI** | OpenAI | `codex` | Limited | Pro plan | OpenAI-native tasks |

### Python Package Managers

| Tool | Speed | Replaces | Key Command | Best For |
|------|-------|---------|-------------|---------|
| **uv** (recommended) | ⚡⚡⚡ 10-100× faster | pip + venv + pyenv | `uv add <pkg>` | All AI Engineering projects |
| **pip + venv** | ⚡ Baseline | — | `pip install <pkg>` | Quick scripts, legacy projects |
| **conda** | ⚡ (sometimes slow) | pip + venv + binary deps | `conda install <pkg>` | Data science with binary deps (CUDA, etc.) |
| **poetry** | ⚡⚡ | pip + venv | `poetry add <pkg>` | Mature alternative to uv |

**Decision rule:** Use uv for all new projects. Use conda only if you need CUDA or other binary scientific packages not available via pip. Use pip+venv only for quick one-off scripts or legacy projects.

### Local AI Runners — Ollama vs LM Studio

| Dimension | Ollama | LM Studio |
|-----------|--------|-----------|
| **Interface** | Terminal (CLI) | Desktop GUI |
| **Installation** | Brew / install script | Download .exe/.dmg |
| **Model management** | `ollama pull <model>` | Point-and-click browser |
| **API** | OpenAI-compatible on :11434 | OpenAI-compatible on :1234 |
| **Docker support** | Yes — official Docker image | No |
| **Headless/server** | Yes — designed for it | Difficult |
| **Scriptable** | Yes | No |
| **Model count** | 500+ | 8,000+ (via Hugging Face) |
| **Best for** | Developers, CI/CD, Docker | Visual model exploration, non-developers |
| **Cost** | Free | Free |

**Decision rule:** 
- Writing code and scripts → Ollama
- Exploring models visually, comparing outputs → LM Studio
- Both → Install both; they run on different ports and do not conflict

### AI Code Assistants in VS Code

| Assistant | Cost | Local model support | Best for |
|-----------|------|-------------------|---------|
| **GitHub Copilot** | Free / $10/mo Pro | No | Most mature, best completions |
| **Continue.dev** | Free (OSS) | Yes (Ollama, LM Studio) | Privacy-focused, local model integration |
| **Gemini Code Assist** | Free (individual tier) | No | Google ecosystem, Python/Web |
| **Tabnine** | Free / $12/mo Pro | Yes (local inference) | Maximum privacy, corporate environments |
| **Codeium** | Free | No | Good free alternative to Copilot |

**Decision rule:**
- Standard developer → GitHub Copilot (most polished)
- Privacy-first, want local models → Continue.dev + Ollama
- Budget zero, Google ecosystem → Gemini Code Assist
- Corporate, code cannot leave the building → Tabnine Enterprise

---

## 11. Best Practices

### 1. Use uv for Every New Python AI Project

```bash
# Never do this for project work:
pip install anthropic   # Installs globally, causes version conflicts

# Always do this:
uv init my-project
cd my-project
uv add anthropic        # Installs in isolation, locked in uv.lock
```

### 2. Commit `.env.example`, Never `.env`

```bash
# Every repository must have these two files:

# 1. .env.example — committed to git, shows what keys are needed (no values)
ANTHROPIC_API_KEY=
OPENAI_API_KEY=

# 2. .env — never committed, contains your actual values
ANTHROPIC_API_KEY=sk-ant-your-real-key
```

### 3. Fail Fast on Missing Configuration

```python
# config.py — validate at startup, not at runtime
import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise RuntimeError(
        "ANTHROPIC_API_KEY is not set. "
        "Copy .env.example to .env and add your key."
    )
```

### 4. Use Ollama During Development, Cloud APIs for Production

```python
# ai_client.py
import os
from openai import OpenAI

def get_client(force_cloud: bool = False) -> tuple[OpenAI, str]:
    """
    Return the right client and model based on the environment.
    Local during development, cloud in production.
    """
    env = os.environ.get("ENVIRONMENT", "development")

    if env == "production" or force_cloud:
        return (
            OpenAI(api_key=os.environ["OPENAI_API_KEY"]),
            "gpt-4o-mini"
        )
    else:
        # Free local model during development
        return (
            OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
            "llama3.2"
        )
```

### 5. Pin Node.js and Python Versions Per Project

```bash
# .nvmrc — Node.js version pin
echo "22" > .nvmrc
# Now running: nvm use  (without version) reads this file

# .python-version — uv reads this automatically
echo "3.12" > .python-version
```

### 6. Set Up Secret Scanning Before Your First Commit

```bash
# Install detect-secrets
uv add --dev detect-secrets

# Create a baseline (scans current code)
detect-secrets scan > .secrets.baseline

# Add a pre-commit hook that checks every commit
uv add --dev pre-commit

# .pre-commit-config.yaml
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: ['--baseline', '.secrets.baseline']
EOF

pre-commit install
```

Now `git commit` will fail if any secrets are detected.

### 7. Use a Makefile for Common Commands

```makefile
# Makefile
.PHONY: help dev stop test clean env-check

help:
	@echo "Available commands:"
	@echo "  make dev       Start Docker stack + app"
	@echo "  make stop      Stop Docker stack"
	@echo "  make test      Run tests"
	@echo "  make clean     Remove all containers and volumes"
	@echo "  make env-check Verify .env is set up"

env-check:
	@test -f .env || (echo "ERROR: .env not found. Copy .env.example to .env" && exit 1)
	@echo ".env found"

dev: env-check
	docker compose up -d
	uv run python src/main.py

stop:
	docker compose down

test:
	uv run pytest tests/ -v --tb=short

clean:
	docker compose down -v
	find . -type d -name __pycache__ -exec rm -rf {} +
```

### 8. Separate Config from Code Completely

```python
# WRONG: mixing config and logic
def call_ai(prompt):
    client = anthropic.Anthropic(api_key="sk-ant-hardcoded")  # Never do this
    return client.messages.create(model="claude-haiku-4-5-20251001", ...)

# RIGHT: config is loaded once at startup
from src.config import config
from src.ai.client import get_anthropic_client

def call_ai(prompt):
    client = get_anthropic_client()  # Gets key from config, which reads from .env
    return client.messages.create(model=config.DEFAULT_MODEL, ...)
```

### 9. Use Descriptive Model Constants, Not Raw Strings

```python
# models.py
class Models:
    """
    Centralised model constants.
    When models update, change here once — not scattered across the codebase.
    Always verify current model IDs at docs.anthropic.com/models.
    """
    # Anthropic
    CLAUDE_FAST = "claude-haiku-4-5-20251001"
    CLAUDE_BALANCED = "claude-sonnet-4-6"
    CLAUDE_BEST = "claude-opus-4-8"

    # OpenAI
    GPT_FAST = "gpt-4o-mini"
    GPT_BALANCED = "gpt-4o"

    # Local
    LOCAL_FAST = "llama3.2"
    LOCAL_BALANCED = "qwen3:8b"

# Usage
from src.models import Models

response = client.messages.create(
    model=Models.CLAUDE_FAST,  # Change in one place, updates everywhere
    ...
)
```

### 10. Verify the Full Stack Works Before Starting a Project

```bash
#!/bin/bash
# scripts/verify-setup.sh
# Run this after cloning a project to verify everything is working

set -e

echo "Checking environment setup..."

# Check .env exists
[ -f .env ] || (echo "ERROR: .env not found" && exit 1)

# Check Docker is running
docker info > /dev/null 2>&1 || (echo "ERROR: Docker not running" && exit 1)

# Check Ollama is accessible
curl -s http://localhost:11434/api/tags > /dev/null || (echo "ERROR: Ollama not running. Run: docker compose up -d" && exit 1)

# Check Python environment
uv run python -c "import anthropic; print('anthropic SDK OK')"

# Check API key is set
uv run python -c "
import os; from dotenv import load_dotenv; load_dotenv()
key = os.environ.get('ANTHROPIC_API_KEY')
if not key: print('WARNING: ANTHROPIC_API_KEY not set')
else: print(f'ANTHROPIC_API_KEY set (ends in ...{key[-4:]})')
"

echo "Setup verification complete."
```

---

## 12. Security Considerations

### The API Key Threat Model

AI API keys are high-value targets. A leaked Anthropic or OpenAI key can result in hundreds or thousands of dollars in charges before you notice. Keys are harvested by automated scanners that watch GitHub 24/7 — within minutes of a key being pushed, it can be used.

**The attack timeline when a key is committed:**

```
0s    Key pushed to public GitHub
~60s  Automated scanner detects it
~90s  Attacker begins using key to generate tokens
~4h   You notice unusual charges (if you have billing alerts)
~8h   Total charges: $500+ depending on model used
```

**Defences, in priority order:**

#### 1. Never commit secrets — enforce with pre-commit hooks

```bash
# Install detect-secrets (see Best Practices #6)
# This runs on every git commit and blocks if secrets are found
pre-commit install
```

#### 2. Set billing alerts immediately

Every AI provider has billing alerts. Set them to email you at $5, $25, and $100 of usage. This catches both leaks and unexpected usage spikes.

- Anthropic: console.anthropic.com → Settings → Billing
- OpenAI: platform.openai.com → Settings → Billing
- Google: aistudio.google.com → Usage limits

#### 3. Use read-only / scoped keys for production

Where possible, create separate API keys for development and production. Anthropic and OpenAI allow you to set spending limits per key.

#### 4. What to do if a key is compromised

```bash
# Step 1: Immediately revoke the key (before doing anything else)
# Go to the provider's console and delete/regenerate the key

# Step 2: Rotate everywhere it was used
# Update .env on all machines, update production secrets manager

# Step 3: Check usage logs
# Look for any unexpected API calls in the last 24 hours

# Step 4: Remove from git history (if it was committed)
git filter-repo --invert-paths --path .env   # Requires git-filter-repo
# This rewrites history — coordinate with your team if on a shared repo

# Step 5: Force-push to override remote history
git push origin --force --all
```

#### 5. Use environment-specific keys

```bash
# .env.development  — key with low spending limit, debug logging enabled
# .env.staging      — separate key, staging environment config
# .env.production   — managed by Secrets Manager, not a file
```

---

## 13. Cost Considerations

> **Prices verified June 2026. These tools evolve rapidly. Confirm current pricing at each provider's website.**

### Full Stack Cost Options

| Setup | Monthly Cost | What You Get |
|-------|-------------|-------------|
| **Free Setup** | $0 | VS Code + GitHub Copilot free + Continue.dev + Ollama + Anthropic/Google trial credits |
| **Individual Pro** | ~$30/mo | Cursor Pro ($20) + GitHub Copilot Pro ($10) |
| **Individual All-In** | ~$30–60/mo | Cursor Pro ($20) + Claude Code via API (~$10) + Copilot Pro ($10) |
| **Team (per developer)** | ~$59/user/mo | Cursor Teams ($40) + Copilot Business ($19) |

### Where the Money Is Not Being Spent

Note that the above figures are for **development tools only** — not for the AI API calls your application makes. API costs scale with usage. The development environment costs are mostly fixed.

For development/learning:
- Ollama is completely free — use it for as much iteration as you want
- Google Gemini free tier: generous daily limits for learning and small projects
- Claude Haiku and GPT-4o-mini: fractions of a cent per test call

For production: see Chapter 19 (Cost Engineering) for full production cost modelling.

### Free vs Paid Tool Comparison

| Tool | Free Tier | Paid Features Worth It? |
|------|-----------|------------------------|
| VS Code | Fully free | N/A |
| GitHub Copilot | Basic completions | Yes if you code daily — $10/mo pays for itself quickly |
| Cursor | Hobby plan (limited) | Pro $20/mo — worth it for AI-heavy work |
| Claude Code | Via API key | Charged per token — use Ollama to reduce cost |
| Antigravity CLI | ~20 requests/day | Enterprise license for teams |
| Ollama | Completely free | N/A |
| LM Studio | Completely free | N/A |
| Docker Desktop | Free for personal use | Paid for companies >250 employees |

---

## 14. Common Mistakes

### Mistake 1: Using `sudo` with `npm install`

```bash
# WRONG — creates permission problems and security risk
sudo npm install -g @anthropic-ai/claude-code

# RIGHT — fix permissions by using nvm
# nvm installs Node.js in your home directory, no sudo needed
nvm install 22
npm install -g @anthropic-ai/claude-code  # Works without sudo
```

### Mistake 2: Committing `.env` to Git

```bash
# WRONG — .env is missing from .gitignore
git add .
git commit -m "initial commit"   # .env just became public

# RIGHT — add .gitignore BEFORE your first commit
echo ".env" >> .gitignore
echo ".env.local" >> .gitignore
git add .gitignore
git commit -m "add gitignore"
```

If you have already committed `.env`: revoke and rotate all keys immediately before doing anything else. Then clean the git history.

### Mistake 3: Hardcoding API Keys in Code

```python
# WRONG — the key is now in every clone of this repo forever
client = anthropic.Anthropic(api_key="sk-ant-api03-actual-key-here")

# RIGHT
from dotenv import load_dotenv
import os

load_dotenv()
client = anthropic.Anthropic()  # Reads ANTHROPIC_API_KEY from environment automatically
```

### Mistake 4: Installing Python Packages Globally

```bash
# WRONG — contaminates your global Python, causes version conflicts across projects
pip install anthropic

# RIGHT — install inside the project's virtual environment
uv add anthropic           # uv handles the venv automatically
# OR
python -m venv .venv && source .venv/bin/activate && pip install anthropic
```

### Mistake 5: Trying to Run a Model That Is Too Large for Your RAM

```bash
# Checking before you download saves time
# A 7B model needs ~8 GB RAM
# A 13B model needs ~16 GB RAM
# A 70B model needs ~48 GB RAM (or quantised to ~8 GB with significant quality loss)

# LM Studio shows RAM requirements before download — use it to check first
# Ollama: check ollama.com/library for size information

# If your model is too slow:
ollama pull llama3.2     # 3B model — runs on 8 GB RAM, fast
# Instead of:
ollama pull llama3.3:70b  # 70B — needs 48 GB+ RAM
```

### Mistake 6: Not Reading `.env` at Startup

```python
# WRONG — load_dotenv() not called before reading the key
import os
import anthropic

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
# Returns None if .env was not loaded → API call fails with auth error

# RIGHT — load first, read second
from dotenv import load_dotenv
import os
import anthropic

load_dotenv()  # This must come before any os.environ.get() calls
client = anthropic.Anthropic()  # Now picks up the key from .env
```

### Mistake 7: Forgetting That Docker Compose Services Need the Same Env Variables

```python
# WRONG — .env works locally but not in Docker
client = OpenAI(base_url="http://localhost:11434/v1")
# Inside a Docker container, "localhost" is the container itself, not the host

# RIGHT — use Docker's service name as the host
# In docker-compose.yml, your app service connects to ollama by its service name:
client = OpenAI(base_url="http://ollama:11434/v1")  # "ollama" = Docker service name

# In your .env:
OLLAMA_BASE_URL=http://ollama:11434/v1    # When running in Docker
# OLLAMA_BASE_URL=http://localhost:11434/v1  # When running locally (comment this back in)
```

---

## 15. Debugging Guide

### Diagnostic Flowchart

```mermaid
flowchart TD
    ERROR[Something is not working] --> Q1{Which layer?}

    Q1 -->|API call fails| Q2{What error?}
    Q1 -->|Ollama/local| Q3{Ollama running?}
    Q1 -->|Docker| Q4{Docker running?}
    Q1 -->|Import error| Q5{Is package installed?}

    Q2 -->|401 Unauthorized| FIX1["Check .env is loaded<br/>Check API key is valid<br/>Did load_dotenv() run first?"]
    Q2 -->|Connection refused| FIX2["Check internet/VPN<br/>Check API URL is correct<br/>Check provider status page"]
    Q2 -->|429 Rate limit| FIX3["Add retry with backoff<br/>Use cheaper model<br/>Switch to Ollama temporarily"]

    Q3 -->|No| FIX4["ollama serve<br/>OR docker compose up -d"]
    Q3 -->|Yes, but slow| FIX5["Model too large for RAM<br/>Use smaller model<br/>Check ollama ps"]

    Q4 -->|No| FIX6["Start Docker Desktop<br/>Wait 30s for daemon to start"]
    Q4 -->|Container fails| FIX7["docker compose logs <service><br/>Check for port conflicts<br/>docker compose down -v && up -d"]

    Q5 -->|No| FIX8["uv add <package><br/>Check pyproject.toml"]
    Q5 -->|Yes but fails| FIX9["Wrong venv active?<br/>uv run python ... instead of python ..."]
```

### Common Error Messages and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `AuthenticationError: Invalid API key` | Key not loaded or wrong value | Check `load_dotenv()` is called first; print `os.environ.get("ANTHROPIC_API_KEY")[:8]` to verify |
| `ConnectionRefusedError: [Errno 61]` | Ollama not running | `ollama serve` or `docker compose up -d` |
| `ModuleNotFoundError: No module named 'anthropic'` | Package not in current venv | `uv add anthropic` or activate the right venv |
| `npm ERR! EACCES: permission denied` | Using npm with sudo | Remove global npm, reinstall via nvm |
| `Error: Cannot find module '@anthropic-ai/claude-code'` | Claude Code not installed | `npm install -g @anthropic-ai/claude-code` (without sudo) |
| `docker: command not found` | Docker Desktop not installed or not in PATH | Install Docker Desktop and restart terminal |
| `port 11434 is already in use` | Ollama is already running outside Docker | `ollama stop` or kill the process using port 11434 |
| `WARNING: .env not found` | python-dotenv cannot find `.env` | Check your working directory; `.env` must be in the directory where you run the script |

### Verifying Each Tool

```bash
# Full environment verification script
echo "=== Claude Code CLI ===" && claude --version
echo "=== Antigravity CLI ===" && agy --version
echo "=== uv ===" && uv --version
echo "=== Python ===" && uv run python --version
echo "=== Node.js ===" && node --version
echo "=== Docker ===" && docker --version
echo "=== Ollama ===" && ollama list
echo "=== ChromaDB ===" && curl -s http://localhost:8000/api/v1/heartbeat | python3 -m json.tool
echo "=== Redis ===" && redis-cli ping
```

---

## 16. Performance Optimisation

### IDE Performance

**VS Code / Cursor becomes slow when:**
- Too many extensions are active: disable extensions you do not use per workspace
- AI completions are triggering on every keystroke: configure a delay in Copilot/Cursor settings
- The language server is indexing a huge `node_modules` or `.venv` directory: add them to `.vscode/settings.json`:

```json
{
  "files.watcherExclude": {
    "**/.venv/**": true,
    "**/node_modules/**": true,
    "**/.git/**": true
  },
  "search.exclude": {
    "**/.venv": true,
    "**/node_modules": true
  }
}
```

### Ollama Performance

```bash
# Check GPU utilisation (macOS Apple Silicon)
ollama run llama3.2 "hello"
# Look for "using Metal" in the output — means GPU acceleration is active

# Check memory usage
ollama ps   # Shows which models are loaded and how much VRAM/RAM they use

# Unload a model when not in use (free RAM)
ollama stop llama3.2

# For maximum speed, use a quantised smaller model
ollama pull llama3.2:3b-instruct-q4_K_M   # 4-bit quantised — 2× faster, slightly lower quality
```

### Docker Compose Startup Speed

```bash
# First run (downloads images): 5-15 minutes
# Subsequent runs (images cached): 10-30 seconds

# Pre-pull images to speed up first-run on new machines:
docker compose pull

# Mount code as a volume instead of copying for faster iteration:
# docker-compose.yml
services:
  app:
    volumes:
      - .:/workspace:cached  # "cached" = macOS performance optimization
```

---

## 17. Exercises

### Exercise 1 — Complete the Minimal Setup (45 minutes)
1. Install VS Code and GitHub Copilot
2. Install Python 3.12 and create a virtual environment
3. Install Node.js 22
4. Set up your Anthropic API key in a `.env` file
5. Run both the Python and Node.js first-call examples from Section 6
6. Verify you can see token usage in the output
7. Open a new terminal window and confirm your key is NOT visible in `env` (it should only be in the `.env` file)

### Exercise 2 — Install the Professional Toolset (60 minutes)
1. Install uv and create a new project with `uv init`
2. Add `anthropic` and `python-dotenv` as dependencies using uv
3. Install nvm and Node.js 22 via nvm
4. Install Claude Code CLI and run `claude --version`
5. Install Antigravity CLI and run `agy --version`
6. Install Ollama and pull `llama3.2`
7. Run the Ollama Python example from Section 7 and confirm you get a response from a local model

### Exercise 3 — Build the Docker AI Stack (45 minutes)
1. Install Docker Desktop
2. Create the `docker-compose.yml` from Section 8
3. Run `docker compose up -d`
4. Verify all services are running: `docker compose ps`
5. Confirm Ollama is accessible: `curl http://localhost:11434/api/tags`
6. Confirm ChromaDB is running: `curl http://localhost:8000/api/v1/heartbeat`
7. Pull the `llama3.2` model inside the container: `docker compose exec ollama ollama pull llama3.2`

### Exercise 4 — Secret Scanning (30 minutes)
1. In a test repository (NOT one with real keys), install `detect-secrets` via uv
2. Create a fake API key in a test file (`FAKE_KEY=sk-ant-fakekeyfortest123456`)
3. Run `detect-secrets scan` and verify it detects the key
4. Set up `pre-commit` with the detect-secrets hook
5. Try to commit the file with the fake key — verify the commit is blocked
6. Understand what you would need to do if you accidentally committed a real key

### Exercise 5 — Provider Switching (45 minutes)
Write a Python script that:
1. Accepts a `--provider` flag: `anthropic`, `openai`, or `local`
2. Sends the same prompt to the chosen provider
3. Returns the response along with the model name and token usage
4. Falls back to the `local` provider (Ollama) if the cloud provider key is not set

This is the foundation of model routing — a pattern you will use in many production systems.

---

## 18. Quiz

**1.** What is the correct command to install Claude Code CLI?
- A) `pip install claude-code`
- B) `npm install -g @anthropic-ai/claude-code` (without sudo)
- C) `brew install claude-code`
- D) `curl -fsSL https://claude.ai/install.sh | sh`

**2.** What does `uv add anthropic` do differently from `pip install anthropic`?

**3.** You commit a `.env` file containing your Anthropic API key to a public GitHub repository. What should you do FIRST?

**4.** Why is `http://localhost:11434` used for Ollama? What is running on that port?

**5.** What is the difference between Ollama and LM Studio for running local models?

**6.** As of June 2026, what tool replaced Google's Gemini CLI for individual users? What is the install command?

**7.** Your Python script works when you run it directly but fails with `AuthenticationError` when run inside a Docker container. What is the most likely cause?

**8.** What is the `.env.example` file for, and why is it committed to git when `.env` is not?

**9.** You want your development team to get a fully working local AI stack (Ollama + ChromaDB + Redis) running with one command. What tool would you use, and what is that command?

**10.** A colleague says "just use conda, it handles everything." List two reasons why uv might be a better choice for a new AI Engineering project in 2026.

---

**Quiz Answers:**

1. **B.** `npm install -g @anthropic-ai/claude-code` without sudo. Never use sudo with npm — it creates permission and security problems. If sudo is needed, your Node.js installation is misconfigured; fix it with nvm.

2. `uv add` installs the package into an isolated project virtual environment, adds it to `pyproject.toml` with the correct version, and updates `uv.lock` for reproducibility. `pip install` installs globally (or in whatever venv you have activated), does not track the dependency in a project file, and is 10–100× slower.

3. **Revoke and regenerate the API key immediately** — before cleaning up git history, before anything else. Scanners find exposed keys within seconds. Once revoked, the old key is worthless. Then rotate everywhere it was used. Then clean the git history with `git filter-repo`.

4. Ollama runs an HTTP server on port 11434 that exposes an OpenAI-compatible API. Any tool that can call `http://localhost:11434/v1/chat/completions` can use local models through Ollama — no code changes needed compared to using a cloud API.

5. **Ollama** is terminal-based: you pull and run models via CLI commands, it runs as a server/daemon, it supports Docker, and it is scriptable. **LM Studio** is a GUI desktop application: you browse and download models visually, it shows RAM requirements before download, it is easier for non-developers to use. Both expose an OpenAI-compatible API. Both are free.

6. **Antigravity CLI** replaced Gemini CLI for individuals on June 18, 2026. Install command: `curl -fsSL https://antigravity.google/cli/install.sh | bash` (macOS/Linux). The command is `agy` (not `gemini`).

7. Inside a Docker container, `localhost` refers to the container itself — not the host machine. The Anthropic API URL (`api.anthropic.com`) should still work if you have internet access. But if you were calling Ollama at `http://localhost:11434`, that would fail because Ollama is on the host machine. The fix: use the Docker service name (`http://ollama:11434`) instead of `localhost`, or use `host.docker.internal` to reach the host machine.

8. `.env.example` shows every environment variable the project needs, with placeholder values but no real secrets. It is committed so any developer who clones the repo knows exactly what variables to fill in. `.env` contains the real values and is never committed because those values include API keys that would be exposed to anyone with repository access.

9. **Docker Compose**, with the command `docker compose up -d`. The `-d` flag runs it in detached mode (background). One `docker-compose.yml` file defines the entire stack and every developer runs the same command.

10. Two reasons to prefer uv over conda for AI Engineering: (1) **Speed** — uv installs packages 10–100× faster, which matters when you are frequently spinning up new projects or updating dependencies. (2) **Simplicity** — uv replaces pip, venv, and pyenv in a single tool with one syntax. Conda is more appropriate when you need binary scientific packages (CUDA libraries, specific NumPy builds) that are not available via pip. For pure AI Engineering work using the Anthropic/OpenAI SDKs, uv is simpler and faster.

---

## 19. Mini Project

### Build a Development Environment Verifier (2–3 hours)

Create a command-line tool called `ai-verify` that checks whether a developer's machine has a complete AI development environment and reports what is missing.

**What it must do:**

1. Check each tool is installed and print its version: uv, Node.js, Docker, Ollama
2. Check that `.env` exists and that at least one API key is set (without printing the key value)
3. Make a live test API call to each configured provider and confirm it returns a response
4. Make a live test call to Ollama's local API and confirm it is running
5. Display a pass/fail summary with specific instructions for fixing each failure

**Example output:**
```
AI Development Environment Check
═══════════════════════════════════════

Runtime Tools:
  ✅ uv           0.4.x
  ✅ Node.js       v22.x.x
  ✅ Docker        27.x.x
  ✅ Ollama        running (3 models installed)

API Keys:
  ✅ ANTHROPIC_API_KEY   set (ends in ...xyz)
  ❌ OPENAI_API_KEY      not set
  ✅ GOOGLE_AI_API_KEY   set (ends in ...abc)

Live API Tests:
  ✅ Anthropic    responded in 423ms
  ⏭  OpenAI       skipped (key not set)
  ✅ Google AI    responded in 312ms
  ✅ Ollama       responded in 89ms (llama3.2)

Summary: 1 issue found
  → OPENAI_API_KEY not set (optional — add to .env if needed)
```

**Acceptance Criteria:**
- [ ] Works on macOS and Linux (bonus: Windows)
- [ ] Written in Python using uv
- [ ] Checks all four tools (uv, Node.js, Docker, Ollama)
- [ ] Checks at least two API providers
- [ ] Makes actual API calls to verify keys work (not just presence checks)
- [ ] Never prints the full API key value — only the last 4 characters
- [ ] Clear instructions printed for any check that fails
- [ ] Exit code 0 if all checks pass, non-zero if any fail (useful for CI)

---

## 20. Production Project

### Build a Team Onboarding Environment (1–2 days)

Build a complete, reproducible AI development environment that a new team member can set up from zero to productive in under 30 minutes.

**Requirements:**

1. **Setup script** (`scripts/setup.sh`) that installs all tools, creates the project structure, and verifies everything is working

2. **Docker Compose stack** with Ollama, ChromaDB, Redis, and a simple app service

3. **Project template** following the structure from Section 7, with all the config files, Makefile, and source code structure

4. **Environment validation** — the setup script runs all checks and fails with clear instructions if anything is wrong

5. **README.md** that a developer with no prior knowledge of this project can follow to get set up in under 30 minutes

6. **A working demo application** — a simple AI chat endpoint (FastAPI or Express) that:
   - Uses Ollama locally
   - Reads config from environment variables
   - Has a health check endpoint showing which AI provider is in use
   - Logs each request with token usage

**Acceptance Criteria:**
- [ ] A developer following the README can get from zero to running `make dev` in under 30 minutes
- [ ] `docker compose up -d` starts the entire AI stack cleanly
- [ ] The demo chat endpoint responds to POST requests with AI-generated responses
- [ ] The health check endpoint returns the current model and provider
- [ ] The setup script catches and explains every common setup error
- [ ] No API keys are required to run the basic setup (uses Ollama for local models)
- [ ] The `.env.example` documents every required variable with an explanation

---

## 21. Key Takeaways

- **The right environment is a force multiplier.** 3–5 hours of setup saves hours on every project. Never skip it.
- **uv replaces pip, venv, and pyenv.** Use it for all new Python AI projects — it is faster and simpler.
- **Ollama is the development AI.** Free, private, offline, and compatible with the OpenAI SDK. Use it during iteration, switch to cloud APIs for production quality.
- **`.env` is never committed.** `.env.example` always is. Set up secret scanning before the first commit.
- **Cursor is VS Code with AI deeply integrated.** The same extensions work, the same shortcuts work, and AI understands your entire codebase instead of just the current file.
- **Claude Code and Antigravity CLI are terminal agents.** They understand your project and can make multi-file changes, write tests, and execute commands autonomously.
- **Gemini CLI was retired June 18, 2026.** The replacement is Antigravity CLI (`agy`). The Gemini Code Assist VS Code extension remains available.
- **Docker Compose is your local production environment.** One command starts the entire AI stack.
- **Never hardcode secrets.** `os.environ["KEY"]` in dev, Secrets Manager in production.
- **Fail fast on missing config.** Validate at startup with helpful error messages, not at runtime with cryptic failures.

---

## 22. Chapter Summary

| Topic | Key Takeaway |
|-------|-------------|
| VS Code vs Cursor | Both excellent; Cursor better for AI-first workflows, VS Code for teams already on it |
| uv | Replaces pip + venv + pyenv; 10–100× faster; use for all new projects |
| Claude Code CLI | `npm install -g @anthropic-ai/claude-code`; terminal AI agent for autonomous tasks |
| Antigravity CLI | Google's Gemini CLI replacement (June 2026); `agy` command; install via shell script |
| Ollama | Free local models; OpenAI-compatible API at :11434; Docker support |
| LM Studio | Free GUI model runner; OpenAI-compatible API at :1234; best for visual model exploration |
| `.env` file | Never commit; always have `.env.example`; set up secret scanning pre-commit |
| Docker Compose | Local AI stack (Ollama + ChromaDB + Redis) in one command |
| Production secrets | .env → AWS/GCP/Azure Secrets Manager; never a file in production |
| Project template | `.env.example`, `config.py`, `Makefile`, `docker-compose.yml` — create it once, reuse forever |

---

## 23. Resources

### GitHub Repositories

| Repository | Purpose |
|-----------|---------|
| [astral-sh/uv](https://github.com/astral-sh/uv) | uv source code, documentation, and issue tracker |
| [ollama/ollama](https://github.com/ollama/ollama) | Ollama source code and model library |
| [nvm-sh/nvm](https://github.com/nvm-sh/nvm) | Node Version Manager |
| [Yelp/detect-secrets](https://github.com/Yelp/detect-secrets) | Pre-commit secret detection |
| [anthropics/anthropic-cookbook](https://github.com/anthropics/anthropic-cookbook) | Working examples for all Claude patterns |
| [google-antigravity/antigravity-cli](https://github.com/google-antigravity/antigravity-cli) | Antigravity CLI source and documentation |

### Official Documentation

| Resource | URL | What It Covers |
|----------|-----|---------------|
| uv Documentation | docs.astral.sh/uv | Full uv reference |
| Claude Code Docs | code.claude.com/docs | Claude Code CLI setup and usage |
| Antigravity CLI Docs | antigravity.google/docs | Antigravity setup and commands |
| Ollama Documentation | docs.ollama.com | Model management, API reference |
| LM Studio | lmstudio.ai | Download and usage guide |
| Cursor Documentation | cursor.com/docs | Editor features and AI settings |

### YouTube Videos

| Video | Why Watch |
|-------|----------|
| "Ollama in 5 minutes" | Quickest way to understand what Ollama does |
| "uv: A Modern Python Package Manager" (Fireship) | Best intro to why uv matters |
| "Claude Code Tutorial" (Anthropic) | Official walkthrough of Claude Code CLI |
| "Docker Compose for Developers" (TechWorld with Nana) | Clear Docker Compose fundamentals |

### Books

| Book | Why Read |
|------|---------|
| **The Pragmatic Programmer** | Timeless on development environment discipline |
| **Docker Deep Dive** (Nigel Poulton) | Short, clear Docker fundamentals |

---

## 24. Glossary Terms Introduced

| Term | Definition |
|------|-----------|
| IDE | Integrated Development Environment — the editor plus all its tools |
| AI Code Assistant | An AI-powered extension that helps write, complete, and refactor code |
| Package Manager | Tool that installs and manages software dependencies (uv, npm, pip) |
| Virtual Environment | Isolated Python environment per project — prevents version conflicts |
| Environment Variable | Configuration value stored outside code, loaded at runtime |
| `.env` file | Plain text file containing environment variables for local development; never committed |
| `.env.example` | Template `.env` with blank values — committed to git as documentation |
| `.gitignore` | File listing paths git should never track or commit |
| Containerisation | Packaging an app and all its dependencies into a portable, reproducible unit |
| Docker | The most popular container runtime and tooling |
| Docker Compose | Tool for defining and running multi-container apps with one command |
| Local Model Runner | Software that runs open-source AI models on your own hardware (Ollama, LM Studio) |
| uv | Modern Python package and project manager written in Rust — replaces pip + venv + pyenv |
| nvm | Node Version Manager — installs and switches between Node.js versions |
| Claude Code CLI | Anthropic's terminal-based AI agent; `npm install -g @anthropic-ai/claude-code` |
| Antigravity CLI | Google's terminal AI agent, replaced Gemini CLI June 2026; command `agy` |
| Ollama | Terminal-based local model runner; OpenAI-compatible API on port 11434 |
| LM Studio | GUI-based local model runner; OpenAI-compatible API on port 1234 |
| `.cursorrules` | Per-project instructions file that tells Cursor how to assist in this codebase |
| Secrets Manager | Production service for storing API keys securely (AWS, GCP, Azure, Vault) |
| Dev Container | VS Code feature that runs your entire dev environment inside a Docker container |
| Secret Scanning | Pre-commit tool that blocks committing files containing API keys or passwords |

---

## 25. See Also

| Chapter | Why It's Related |
|---------|-----------------|
| [Chapter 1: What is AI Engineering?](./chapter-01-what-is-ai-engineering.md) | Introduced all tools mentioned here; the first API calls in Chapter 1 now have the proper environment behind them |
| [Chapter 2: How LLMs Work](./chapter-02-how-llms-work.md) | Token counting tools (tiktoken) introduced there run in this environment |
| [Chapter 4: AI APIs, SDKs & Streaming](./chapter-04-ai-apis-sdks.md) | The environment set up here is the foundation for production API patterns |
| [Chapter 8: Vector Databases](./chapter-08-vector-databases.md) | ChromaDB in the Docker Compose stack is introduced here and used there |
| [Chapter 9: RAG](./chapter-09-rag.md) | The full Docker stack (Ollama + ChromaDB) is the development environment for RAG |
| [Chapter 18: AI Security](./chapter-18-security.md) | Secret management, key rotation, and API key threat models covered there in depth |
| [Chapter 19: Cost Engineering](./chapter-19-cost-engineering.md) | Production secrets management and environment parity covered there in full |

---

## 26. Preparation for Chapter 4

Chapter 4 (AI APIs, SDKs & Streaming) builds directly on this environment. Before starting it, verify:

**Technical checklist:**
- [ ] At least one AI provider API key is set in your `.env` file and confirmed working
- [ ] Ollama is installed and `llama3.2` is pulled — `ollama run llama3.2 "hello"` returns a response
- [ ] uv is installed — `uv --version` works
- [ ] The Docker Compose stack starts without errors — `docker compose up -d && docker compose ps`
- [ ] Claude Code CLI is installed — `claude --version` works

**Conceptual check — answer without notes:**
- [ ] What is the difference between a local model (Ollama) and a cloud API (Anthropic)?
- [ ] Why should you never put an API key directly in your Python code?
- [ ] What does the `load_dotenv()` call do?
- [ ] When your app runs inside Docker, why is `http://localhost:11434` the wrong URL for Ollama?

**Optional challenge before Chapter 4:**
Build the `ai-verify` tool from the Mini Project (Section 19). By the time Chapter 4 asks you to make production-grade API calls, you will already have a tool that checks whether your environment is ready — and you will have practised error handling patterns that Chapter 4 builds on.

---

*Chapter 3 of 20 | The Complete AI Engineering Course*

*Previous: [Chapter 2: How LLMs Work](./chapter-02-how-llms-work.md)*
*Next: [Chapter 4: AI APIs, SDKs & Streaming](./chapter-04-ai-apis-sdks.md)*
