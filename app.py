"""
AI Research Agent - Streamlit web app
=======================================================
Wraps the same tool-calling Gemini agent (fetch_page / web_search /
final_answer) in a simple web UI so it can be deployed and shared as a
live link instead of run from the command line.

DEPLOY ON STREAMLIT COMMUNITY CLOUD (free)
-------------------------------------------
1. Push this file + requirements.txt to a public GitHub repo.
2. Go to https://share.streamlit.io -> "New app" -> pick the repo.
3. In "Advanced settings" -> "Secrets", add:
       GEMINI_API_KEY = "your-key-here"
4. Deploy. You'll get a public https://<something>.streamlit.app link.

RUN LOCALLY FIRST (recommended before deploying)
--------------------------------------------------
    pip install streamlit requests beautifulsoup4 google-genai
    export GEMINI_API_KEY="your-key-here"      # Windows PowerShell: $env:GEMINI_API_KEY="..."
    streamlit run app.py
"""

import os
import sys

import requests
import streamlit as st
from bs4 import BeautifulSoup

MAX_AGENT_STEPS = 6
MODEL = "gemini-3.5-flash-lite"  # swap here if you want to try a different model

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


# ---------------------------------------------------------------------------
# TOOLS - unchanged from the CLI version
# ---------------------------------------------------------------------------
def fetch_page(url: str) -> dict:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        desc_tag = soup.find("meta", attrs={"name": "description"})
        description = desc_tag["content"].strip() if desc_tag and desc_tag.get("content") else ""
        links = sorted({a.get_text(strip=True) for a in soup.find_all("a")
                         if a.get_text(strip=True) and 2 <= len(a.get_text(strip=True)) <= 30})
        visible_text = soup.get_text(separator=" ", strip=True)
        return {
            "status": "ok",
            "title": title,
            "meta_description": description,
            "nav_or_link_labels": links[:60],
            "visible_text_sample": visible_text[:2000],
            "note": ("Content looks very short - page may be JS-rendered or blocked; "
                      "consider web_search instead.") if len(visible_text) < 500 else "",
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": str(exc)}


def web_search(query: str) -> dict:
    try:
        resp = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers=HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for r in soup.select(".result")[:6]:
            title_tag = r.select_one(".result__a")
            snippet_tag = r.select_one(".result__snippet")
            if title_tag:
                results.append({
                    "title": title_tag.get_text(strip=True),
                    "snippet": snippet_tag.get_text(strip=True) if snippet_tag else "",
                })
        return {"status": "ok", "results": results}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": str(exc)}


TOOL_IMPLEMENTATIONS = {
    "fetch_page": fetch_page,
    "web_search": web_search,
}

TOOL_DECLARATIONS = [
    {
        "name": "fetch_page",
        "description": "Directly fetch and parse a webpage by URL. May fail or return thin "
                        "content on JS-heavy or bot-protected sites.",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
    {
        "name": "web_search",
        "description": "Search the web for information when direct page fetching fails or "
                        "doesn't give enough detail.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "final_answer",
        "description": "Call this once you have gathered enough information to answer the "
                        "task. This ends the research loop.",
        "parameters": {
            "type": "object",
            "properties": {
                "about": {"type": "string", "description": "2-3 sentence summary of what the site/company is"},
                "products": {"type": "array", "items": {"type": "string"}},
                "categories": {"type": "array", "items": {"type": "string"}},
                "reasoning_trace": {
                    "type": "string",
                    "description": "Briefly explain what tools you used and why, in your own words.",
                },
            },
            "required": ["about", "products", "categories", "reasoning_trace"],
        },
    },
]


# ---------------------------------------------------------------------------
# AGENT LOOP - unchanged logic, now yields progress for the UI
# ---------------------------------------------------------------------------
def run_agent(url: str, task: str, log_area):
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not found. Set it as an environment variable locally, "
            "or add it under Settings -> Secrets when deployed on Streamlit Cloud."
        )
    client = genai.Client(api_key=api_key)

    tool = types.Tool(function_declarations=TOOL_DECLARATIONS)
    config = types.GenerateContentConfig(tools=[tool])

    contents = [
        types.Content(
            role="user",
            parts=[types.Part(text=(
                f"Target URL: {url}\n"
                f"Task: {task}\n\n"
                "You have tools available. Decide yourself which to call, in what order. "
                "If fetch_page fails or returns thin/blocked content, try web_search instead "
                "of giving up. Call final_answer only once you're confident in your answer."
            ))],
        )
    ]

    for step in range(1, MAX_AGENT_STEPS + 1):
        response = client.models.generate_content(model=MODEL, contents=contents, config=config)
        candidate = response.candidates[0]
        contents.append(candidate.content)

        function_calls = [p.function_call for p in candidate.content.parts if p.function_call]
        if not function_calls:
            log_area.write(f"**Step {step}:** Agent responded with text but called no tool — stopping.")
            break

        function_response_parts = []
        for fc in function_calls:
            name = fc.name
            args = dict(fc.args)
            log_area.write(f"**Step {step}:** Agent called `{name}({args})`")

            if name == "final_answer":
                return args

            impl = TOOL_IMPLEMENTATIONS.get(name)
            result = impl(**args) if impl else {"status": "error", "error": "unknown tool"}
            function_response_parts.append(
                types.Part.from_function_response(name=name, response={"result": result})
            )

        contents.append(types.Content(role="user", parts=function_response_parts))

    raise RuntimeError(f"Agent did not reach final_answer within {MAX_AGENT_STEPS} steps.")


# ---------------------------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AI Research Agent", page_icon="🔎", layout="centered")

st.title("🔎 AI Research Agent")
st.caption("A Gemini-powered agent that decides for itself which tools to call to research a website.")

with st.form("agent_form"):
    url = st.text_input("Target URL", value="https://www.zomato.com/")
    task = st.text_area(
        "Task",
        value="Find out what this company/website is about, its main products, "
              "and the categories shown on its homepage.",
        height=80,
    )
    submitted = st.form_submit_button("Run agent")

if submitted:
    st.subheader("Agent trace")
    log_area = st.container()

    with st.spinner("Agent is researching..."):
        try:
            report = run_agent(url, task, log_area)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Agent failed: {exc}")
            st.stop()

    st.subheader("Report")
    st.markdown(f"**Reasoning trace:** {report.get('reasoning_trace', '(none provided)')}")

    st.markdown("**About**")
    st.write(report.get("about", ""))

    st.markdown("**Main products**")
    for p in report.get("products", []):
        st.write(f"- {p}")

    st.markdown("**Categories**")
    for c in report.get("categories", []):
        st.write(f"- {c}")

    st.download_button(
        "Download report as JSON",
        data=str(report),
        file_name="agent_report.json",
        mime="application/json",
    )