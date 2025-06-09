import os
import json # Added import
import re # Added import for markdown JSON extraction

from agent.tools_and_schemas import SearchQueryList, Reflection
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage # BaseMessage was already there
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig
from google.genai import Client

from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from agent.configuration import Configuration
from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
    manager_agent_prompt,
    dtc_website_manager_prompt,
    ui_designer_prompt,
    copywriter_prompt,
    developer_prompt,
    asset_creator_prompt
)
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.utils import (
    get_citations,
    get_research_topic,
    insert_citation_markers,
    resolve_urls,
)

load_dotenv()

if os.getenv("GEMINI_API_KEY") is None:
    raise ValueError("GEMINI_API_KEY is not set")

# Used for Google Search API
genai_client = Client(api_key=os.getenv("GEMINI_API_KEY"))

# --- Utility Functions ---
def extract_json_from_markdown(md_string: str) -> str:
    """
    Extracts a JSON string from a markdown code block.
    Supports blocks starting with ```json or just ```.
    """
    # Pattern to find JSON within ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", md_string)
    if match:
        return match.group(1).strip() # Content within the backticks
    return md_string # Return original string if no markdown block is found


# --- Nodes ---
def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates a search queries based on the User's question.

    Uses Gemini 2.0 Flash to create an optimized search query for web research based on
    the User's question.

    Args:
        state: Current graph state containing the User's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated query
    """
    configurable = Configuration.from_runnable_config(config)

    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    llm = ChatGoogleGenerativeAI(
        model=configurable.query_generator_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    structured_llm = llm.with_structured_output(SearchQueryList)

    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    result = structured_llm.invoke(formatted_prompt)
    return {"query_list": result.query}


def continue_to_web_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the web research node.

    This is used to spawn n number of web research nodes, one for each search query.
    """
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["query_list"])
    ]


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    configurable = Configuration.from_runnable_config(config)
    formatted_prompt = web_searcher_instructions.format(
        current_date=get_current_date(),
        research_topic=state["search_query"],
    )

    response = genai_client.models.generate_content(
        model=configurable.query_generator_model,
        contents=formatted_prompt,
        config={
            "tools": [{"google_search": {}}],
            "temperature": 0,
        },
    )
    resolved_urls = resolve_urls(
        response.candidates[0].grounding_metadata.grounding_chunks, state["id"]
    )
    citations = get_citations(response, resolved_urls)
    modified_text = insert_citation_markers(response.text, citations)
    sources_gathered = [item for citation in citations for item in citation["segments"]]

    return {
        "sources_gathered": sources_gathered,
        "search_query": [state["search_query"]],
        "web_research_result": [modified_text],
    }


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    configurable = Configuration.from_runnable_config(config)
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reasoning_model = state.get("reasoning_model") or configurable.reasoning_model

    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )
    llm = ChatGoogleGenerativeAI(
        model=reasoning_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    result = llm.with_structured_output(Reflection).invoke(formatted_prompt)

    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": state["research_loop_count"],
        "number_of_ran_queries": len(state["search_query"]),
    }


def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> OverallState:
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


def finalize_answer(state: OverallState, config: RunnableConfig):
    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model") or configurable.reasoning_model

    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )

    llm = ChatGoogleGenerativeAI(
        model=reasoning_model,
        temperature=0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    result = llm.invoke(formatted_prompt)

    unique_sources = []
    if result.content and isinstance(result.content, str): # Ensure result.content is a string
        for source in state["sources_gathered"]:
            if source.get("short_url") and source.get("value") and source["short_url"] in result.content:
                 result.content = result.content.replace(source["short_url"], source["value"])
                 unique_sources.append(source)

    return {
        "messages": [AIMessage(content=result.content if result.content else "")], # Ensure content is not None
        "sources_gathered": unique_sources,
    }

# --- New Agent Nodes Start ---

def manager_agent_node(state: OverallState, config: RunnableConfig) -> OverallState:
    configurable = Configuration.from_runnable_config(config)
    llm = ChatGoogleGenerativeAI(
        model=configurable.query_generator_model,
        temperature=0.7,
        api_key=os.getenv("GEMINI_API_KEY"),
    )

    user_request = get_research_topic(state["messages"])
    if not user_request:
        user_request = "Develop a new website."

    prompt_input = manager_agent_prompt + f"\n\nHere is the user's specific goal:\n{user_request}"

    response = llm.invoke(prompt_input)
    llm_output_content = response.content if hasattr(response, 'content') else ""

    extracted_json_string = extract_json_from_markdown(llm_output_content)

    try:
        parsed_response = json.loads(extracted_json_string)
    except json.JSONDecodeError as e:
        error_message = (
            f"Error: Manager agent output was not valid JSON, even after attempting to extract from markdown. "
            f"Original content: '{llm_output_content}'. Extracted: '{extracted_json_string}'. Error: {e}"
        )
        return {
            "next_agent_to_call": "ERROR",
            "manager_instruction": "Failed to parse manager agent output.",
            "messages": state.get("messages", []) + [AIMessage(content=error_message)]
        }

    next_agent = parsed_response.get("next_agent_to_call")
    instruction = parsed_response.get("manager_instruction")
    manager_ai_message_content = ""

    if next_agent == "USER_CLARIFICATION":
        manager_ai_message_content = f"Manager: {instruction}"
    else:
        manager_ai_message_content = f"Manager decision: Next agent is {next_agent}. Instruction: {instruction}"

    manager_ai_message = AIMessage(content=manager_ai_message_content if manager_ai_message_content else "Manager: No instruction generated.")

    return {
        "next_agent_to_call": next_agent,
        "manager_instruction": instruction,
        "messages": state.get("messages", []) + [manager_ai_message]
    }

def dtc_website_manager_node(state: OverallState, config: RunnableConfig) -> OverallState:
    configurable = Configuration.from_runnable_config(config)
    llm = ChatGoogleGenerativeAI(
        model=configurable.query_generator_model,
        temperature=0.7,
        api_key=os.getenv("GEMINI_API_KEY"),
    )

    formatted_prompt = dtc_website_manager_prompt.format(
        manager_instruction=state.get("manager_instruction", "No instruction provided.")
    )
    response = llm.invoke(formatted_prompt)
    output_content = response.content if hasattr(response, 'content') else "DtC Manager: No output."

    return {
        "dtc_website_manager_output": output_content,
        "messages": state.get("messages", []) + [AIMessage(content=f"DtC Website Manager Output: {output_content}")]
    }

def ui_designer_node(state: OverallState, config: RunnableConfig) -> OverallState:
    configurable = Configuration.from_runnable_config(config)
    llm = ChatGoogleGenerativeAI(
        model=configurable.query_generator_model,
        temperature=0.7,
        api_key=os.getenv("GEMINI_API_KEY"),
    )

    formatted_prompt = ui_designer_prompt.format(
        manager_instruction=state.get("manager_instruction", "No instruction provided."),
        dtc_website_manager_output=state.get("dtc_website_manager_output", "No DtC output available.")
    )
    response = llm.invoke(formatted_prompt)
    output_content = response.content if hasattr(response, 'content') else "UI Designer: No output."

    return {
        "ui_designer_output": output_content,
        "messages": state.get("messages", []) + [AIMessage(content=f"UI Designer Output: {output_content}")]
    }

def copywriter_node(state: OverallState, config: RunnableConfig) -> OverallState:
    configurable = Configuration.from_runnable_config(config)
    llm = ChatGoogleGenerativeAI(
        model=configurable.query_generator_model,
        temperature=0.7,
        api_key=os.getenv("GEMINI_API_KEY"),
    )

    formatted_prompt = copywriter_prompt.format(
        manager_instruction=state.get("manager_instruction", "No instruction provided."),
        dtc_website_manager_output=state.get("dtc_website_manager_output", "No DtC output available."),
        ui_designer_output=state.get("ui_designer_output", "No UI design output available.")
    )
    response = llm.invoke(formatted_prompt)
    output_content = response.content if hasattr(response, 'content') else "Copywriter: No output."

    return {
        "copywriter_output": output_content,
        "messages": state.get("messages", []) + [AIMessage(content=f"Copywriter Output: {output_content}")]
    }

def developer_node(state: OverallState, config: RunnableConfig) -> OverallState:
    configurable = Configuration.from_runnable_config(config)
    llm = ChatGoogleGenerativeAI(
        model=configurable.query_generator_model,
        temperature=0.7,
        api_key=os.getenv("GEMINI_API_KEY"),
    )

    formatted_prompt = developer_prompt.format(
        manager_instruction=state.get("manager_instruction", "No instruction provided."),
        dtc_website_manager_output=state.get("dtc_website_manager_output", "No DtC output available."),
        ui_designer_output=state.get("ui_designer_output", "No UI design output available."),
        copywriter_output=state.get("copywriter_output", "No copywriter output available.")
    )
    response = llm.invoke(formatted_prompt)
    output_content = response.content if hasattr(response, 'content') else "Developer: No output."

    return {
        "developer_output": output_content,
        "messages": state.get("messages", []) + [AIMessage(content=f"Developer Output: {output_content}")]
    }

def asset_creator_node(state: OverallState, config: RunnableConfig) -> OverallState:
    configurable = Configuration.from_runnable_config(config)
    llm = ChatGoogleGenerativeAI(
        model=configurable.query_generator_model,
        temperature=0.7,
        api_key=os.getenv("GEMINI_API_KEY"),
    )

    formatted_prompt = asset_creator_prompt.format(
        manager_instruction=state.get("manager_instruction", "No instruction provided."),
        dtc_website_manager_output=state.get("dtc_website_manager_output", "No DtC output available."),
        ui_designer_output=state.get("ui_designer_output", "No UI design output available."),
        copywriter_output=state.get("copywriter_output", "No copywriter output available.")
    )
    response = llm.invoke(formatted_prompt)
    output_content = response.content if hasattr(response, 'content') else "Asset Creator: No output."

    return {
        "asset_creator_output": output_content,
        "messages": state.get("messages", []) + [AIMessage(content=f"Asset Creator Output: {output_content}")]
    }

# --- New Agent Nodes End ---

# --- Manager Agent Routing Logic ---
AGENT_TO_NODE_MAP = {
    "DtCWebsiteManager": "dtc_website_manager",
    "UIDesigner": "ui_designer",
    "Copywriter": "copywriter",
    "Developer": "developer",
    "AssetCreator": "asset_creator",
}

def route_to_next_agent(state: OverallState) -> str:
    next_agent_name = state.get("next_agent_to_call")

    if next_agent_name == "USER_CLARIFICATION":
        print(f"Manager is requesting user clarification. Instruction: {state.get('manager_instruction')}")
        return END

    if not next_agent_name or next_agent_name.upper() == "END" or next_agent_name.upper() == "ERROR":
        if next_agent_name and next_agent_name.upper() == "ERROR":
            print(f"Manager agent indicated an error or unparsable output for state: {state.get('messages')[-2:] if state.get('messages') and len(state.get('messages', [])) >= 2 else state.get('messages', [])}. Ending graph.")
        elif not next_agent_name:
            print(f"No next_agent_to_call specified. Ending graph. State: {state.get('messages')[-2:] if state.get('messages') and len(state.get('messages', [])) >= 2 else state.get('messages', [])}")
        return END

    node_name = AGENT_TO_NODE_MAP.get(next_agent_name)

    if node_name:
        return node_name
    else:
        print(f"Error: Unrecognized agent name '{next_agent_name}' received from manager. Ending graph. State: {state.get('messages')[-2:] if state.get('messages') and len(state.get('messages', [])) >= 2 else state.get('messages', [])}")
        return END
# --- End Manager Agent Routing Logic ---


# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

builder.add_node("manager_agent", manager_agent_node)
builder.add_node("dtc_website_manager", dtc_website_manager_node)
builder.add_node("ui_designer", ui_designer_node)
builder.add_node("copywriter", copywriter_node)
builder.add_node("developer", developer_node)
builder.add_node("asset_creator", asset_creator_node)

builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

builder.add_edge(START, "manager_agent")

conditional_routes_map = {node_name: node_name for node_name in AGENT_TO_NODE_MAP.values()}
conditional_routes_map[END] = END
conditional_routes_map["USER_CLARIFICATION"] = END

builder.add_conditional_edges(
    "manager_agent",
    route_to_next_agent,
    conditional_routes_map
)

for node_name in AGENT_TO_NODE_MAP.values():
    builder.add_edge(node_name, "manager_agent")

builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
builder.add_edge("web_research", "reflection")
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)
builder.add_edge("finalize_answer", END)


graph = builder.compile(name="multi-agent-team-workflow")
