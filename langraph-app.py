import os
import io
import sys
import traceback
from typing import TypedDict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langserve import add_routes


# ============================================================
# 1. LLM INITIALIZATION
# ============================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. Add it as an Environment Variable in Render."
    )

llm_flash = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=GEMINI_API_KEY,
    temperature=0,
)


# ============================================================
# 2. LANGGRAPH STATE
# ============================================================

class CrewState(TypedDict, total=False):
    messages: List[BaseMessage]
    next_step: Optional[str]
    code: Optional[str]
    report: Optional[str]


# ============================================================
# 3. TOOLS
# ============================================================

@tool
def run_python_code(code: str) -> str:
    """Execute Python code and return standard output or an error trace."""
    if not isinstance(code, str):
        code = str(code)

    clean_code = (
        code.replace("```python", "")
        .replace("```py", "")
        .replace("```", "")
        .strip()
    )

    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    try:
        local_scope = {}
        exec(clean_code, {}, local_scope)
        result = new_stdout.getvalue()
    except Exception:
        result = f"Execution Error:\n{traceback.format_exc()}"
    finally:
        sys.stdout = old_stdout

    return result.strip() if result.strip() else "Success (no terminal output)"


@tool
def generate_test_cases(task_description: str) -> str:
    """Generate 3 to 5 specific test scenarios for a coding task."""
    prompt = (
        "You are a Senior QA Engineer. Generate 3 to 5 highly specific "
        f"test scenarios for the following coding task: '{task_description}'.\n"
        "Include standard cases and edge cases. Return them as a numbered list."
    )

    response = llm_flash.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


# ============================================================
# 4. LANGGRAPH NODES
# ============================================================

def real_time_developer(state: CrewState):
    """Developer node: generate Python code for the requested task."""
    task = state["messages"][-1].content

    dev_prompt = (
        "Write a clean Python script to solve this coding task:\n"
        f"{task}\n\n"
        "Only return the Python code. Do not return explanations or markdown."
    )

    response = llm_flash.invoke(dev_prompt)
    content = response.content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))
        code_str = "\n".join(parts)
    else:
        code_str = str(content)

    # Remove accidental markdown fences if the model returns them.
    code_str = (
        code_str.replace("```python", "")
        .replace("```py", "")
        .replace("```", "")
        .strip()
    )

    return {"code": code_str, "next_step": "tester"}


def real_time_tester(state: CrewState):
    """Tester node: generate test scenarios and execute generated code."""
    task = state["messages"][-1].content

    test_cases = generate_test_cases.invoke(task)
    cases_str = str(test_cases)

    execution_result = run_python_code.invoke({"code": state["code"]})

    report = (
        "### GENERATED CODE:\n"
        f"{state['code']}\n\n"
        "### EXECUTION OUTPUT:\n"
        f"{execution_result}\n\n"
        "### TEST SCENARIOS EVALUATED:\n"
        f"{cases_str}"
    )

    return {
        "report": report,
        "next_step": "complete",
    }


# ============================================================
# 5. LANGGRAPH CONSTRUCTION
# ============================================================

workflow = StateGraph(CrewState)

workflow.add_node("developer", real_time_developer)
workflow.add_node("tester", real_time_tester)

workflow.add_edge(START, "developer")
workflow.add_edge("developer", "tester")
workflow.add_edge("tester", END)

langgraph_app = workflow.compile()


# ============================================================
# 6. LANGSERVE INPUT / OUTPUT ADAPTER
# ============================================================

class LangGraphInput(BaseModel):
    input: str = Field(
        description="Coding task that the LangGraph developer/tester workflow should process."
    )


class LangGraphOutput(BaseModel):
    code: Optional[str] = None
    report: Optional[str] = None
    next_step: Optional[str] = None


def invoke_langgraph(request: LangGraphInput) -> dict:
    """Convert a simple API request into the CrewState expected by LangGraph."""
    result = langgraph_app.invoke(
        {
            "messages": [HumanMessage(content=request.input)],
            "next_step": "developer",
        }
    )

    return {
        "code": result.get("code"),
        "report": result.get("report"),
        "next_step": result.get("next_step"),
    }


langgraph_runnable = (
    RunnableLambda(invoke_langgraph)
    .with_types(
        input_type=LangGraphInput,
        output_type=LangGraphOutput,
    )
)


# ============================================================
# 7. FASTAPI + LANGSERVE
# ============================================================

app = FastAPI(
    title="LangGraph Coding Developer & Tester API",
    version="1.0.0",
    description=(
        "LangGraph workflow exposed through LangServe. "
        "The workflow generates Python code, creates test scenarios, "
        "executes the generated code, and returns a test report."
    ),
)

# Required LangServe route:
# POST /langraphs/invoke
# POST /langraphs/batch
# POST /langraphs/stream
# and related LangServe endpoints.
add_routes(
    app,
    langgraph_runnable,
    path="/langraphs",
)


@app.get("/")
def root():
    return {
        "message": "LangGraph + LangServe API is running.",
        "route": "/langraphs",
        "invoke_endpoint": "/langraphs/invoke",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


# ============================================================
# 8. RENDER STARTUP
# ============================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
