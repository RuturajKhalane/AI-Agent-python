"""
Autonomous Research & Reporting Agent — Python/LangChain version.

This file wires up the tool-calling agent loop explicitly using LangChain
primitives, so every piece (model, tools, memory, iteration limit) is
visible and tunable in code.

Two agent "modes" are built here:
  - build_agent(): the original single-shot agent -- given a whole research
    goal, it plans and executes on its own and returns a full report.
    Used by the CLI (python agent.py) and by eval.py for baseline runs.
  - build_subtask_agent(): a scoped-down variant used by orchestrator.py's
    execution phase. It answers ONE focused sub-question concisely rather
    than writing a full report, and defaults to a smaller iteration budget
    since its job is narrower.
"""

import os
from dotenv import load_dotenv

load_dotenv(override=True)  # move this up, before importing tools (or anything that reads env vars at import time)

from langchain_groq import ChatGroq
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tools import web_search, scrape_page, calculator

SYSTEM_PROMPT = """You are an autonomous research agent. Given a research goal, you will:
1. Break the goal into a clear multi-step plan
2. Use the available tools to gather real information -- never fabricate data
3. If a tool call fails or returns nothing useful, try a different query or approach before giving up
4. Keep track of everything you've learned so far
5. Only give a final answer once you have enough evidence to fully address the goal
6. Structure your final answer as a clean report with headers and, where relevant, a comparison table
7. Never guess a URL. Only call scrape_page on a URL that was returned by a prior web_search result.
8. When using the calculator tool, pass plain numeric expressions only -- no percent signs, no currency symbols, no units. Strip any currency symbols before calculating (example: use "24.99 + 30.99 + 17.99", not amounts written with a percent sign or currency symbol attached).
"""

SUBTASK_SYSTEM_PROMPT = """You are a research assistant answering ONE focused sub-question
as part of a larger research task planned by another process. You will:
1. Use the available tools to gather real information relevant to this sub-question -- never fabricate data
2. Never guess a URL. Only call scrape_page on a URL that was returned by a prior web_search result
3. When using the calculator tool, pass plain numeric expressions only -- no percent signs, no currency symbols, no units
4. Once you have enough evidence, give a CONCISE answer (roughly 3-6 sentences) to the sub-question.
   Do not write a full report, do not add headers -- just the direct findings, in plain prose.
5. If you cannot find a good answer after a reasonable effort, say so plainly rather than guessing.
"""


def _build_executor(
    system_prompt: str,
    callbacks=None,
    max_iterations: int = 10,
    return_intermediate_steps: bool = False,
) -> AgentExecutor:
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        model_kwargs={"parallel_tool_calls": False},
    )
    tools = [web_search, scrape_page, calculator]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        max_iterations=max_iterations,
        handle_parsing_errors=True,
        max_execution_time=90,
        callbacks=callbacks,
        return_intermediate_steps=return_intermediate_steps,
    )


def build_agent(
    callbacks=None,
    max_iterations: int = 10,
    return_intermediate_steps: bool = False,
) -> AgentExecutor:
    """Single-shot agent: goal in, full report out. Used by the CLI and eval.py."""
    return _build_executor(
        SYSTEM_PROMPT,
        callbacks=callbacks,
        max_iterations=max_iterations,
        return_intermediate_steps=return_intermediate_steps,
    )


def build_subtask_agent(
    callbacks=None,
    max_iterations: int = 4,
    return_intermediate_steps: bool = True,
) -> AgentExecutor:
    """Scoped agent: one sub-question in, a concise finding out. Used by orchestrator.py."""
    return _build_executor(
        SUBTASK_SYSTEM_PROMPT,
        callbacks=callbacks,
        max_iterations=max_iterations,
        return_intermediate_steps=return_intermediate_steps,
    )


def main():
    if not os.getenv("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not set. Copy .env.example to .env and fill it in.")
        return

    executor = build_agent()
    print("Autonomous Research Agent (Python/LangChain)")
    print("Type a research goal, or 'quit' to exit.\n")

    while True:
        goal = input("Research goal: ").strip()
        if goal.lower() in ("quit", "exit"):
            break
        if not goal:
            continue

        result = executor.invoke({"input": goal})

        print("\n--- FINAL REPORT ---\n")
        print(result["output"])
        print("\n---------------------\n")


if __name__ == "__main__":
    main()