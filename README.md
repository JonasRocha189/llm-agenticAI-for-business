# AI Projects for Business

A collection of applied AI projects exploring how LLMs and agents can solve real business problems — from marketing content generation to customer service automation. Built as part of the "LLMs e Agentes de IA para Empresas e Negócios" course.

The goal is to end up with **8 projects**, each tackling a different business scenario (marketing, customer service, and others still to come), using tools like LangChain, LangGraph, Groq/OpenAI models, and Streamlit.

## How to Run

1. Clone the repository and move into the project folder.
2. Create and activate a virtual environment (Python >= 3.11.5).
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file inside the project folder you want to run (e.g. `01-Marketing/.env`) with the required API keys (e.g. `GROQ_API_KEY`, `OPENAI_API_KEY`).
5. Run the desired project's Streamlit app from its folder:
   ```bash
   streamlit run app.py
   ```
   Or open its `app.ipynb` notebook to explore the logic interactively.

> This repo also includes a `pyproject.toml` / `uv.lock`, so `uv sync` is an alternative to `pip install -r requirements.txt` if you use [uv](https://github.com/astral-sh/uv).

## Projects

### 01 - Marketing Content Generator
**Scenario:** A business needs to regularly produce SEO-optimized marketing copy (social posts, blog text, emails) tailored to different platforms, tones, and audiences, without relying on a copywriter for every piece of content.

**What it does:** A Streamlit app where the user picks a topic, platform (Instagram, Facebook, LinkedIn, Blog, Email), tone, length, target audience, SEO keywords, and whether to include a CTA and/or hashtags. An LLM then generates ready-to-publish copy based on those parameters.

**Technologies:** Python, Streamlit, LangChain, ChatGroq (`llama-3.3-70b-versatile`), python-dotenv.

### 02 - Customer Service RAG
**Scenario:** A company handles a high volume of customer support calls/messages with repetitive, similar inquiries. Instead of routing every question to a human agent, answers should be retrieved automatically from the company's own knowledge material.

**What it does:** A chatbot built with Retrieval-Augmented Generation (RAG) that answers customer questions grounded in company-provided documents, reducing repetitive support workload. *(Work in progress.)*

**Technologies:** Python, LangChain, RAG (retrieval + LLM generation), Streamlit.

### 03 - 08 - Coming Soon
The remaining 6 projects are still being planned and built. This section will be updated with a scenario, description, and tech stack for each as they're added.
