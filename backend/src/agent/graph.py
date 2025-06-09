import os
import json # Added import

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
    manager_agent_prompt, # Added import
    dtc_website_manager_prompt, # Added import
    ui_designer_prompt, # Added import
    copywriter_prompt, # Added import
    developer_prompt, # Added import
    asset_creator_prompt # Added import
)
from langchain_google_genai import ChatGoogleGenerativeAI
from agent.utils import (
    get_citations,
    get_research_topic, # Already imported, will use for manager_agent_node
    insert_citation_markers,
    resolve_urls,
)

load_dotenv()

if os.getenv("GEMINI_API_KEY") is None:
    raise ValueError("GEMINI_API_KEY is not set")

# Used for Google Search API
genai_client = Client(api_key=os.getenv("GEMINI_API_KEY"))


# Nodes
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

    # check for custom initial search query count
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    # init Gemini 2.0 Flash
    llm = ChatGoogleGenerativeAI(
        model=configurable.query_generator_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    structured_llm = llm.with_structured_output(SearchQueryList)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    # Generate the search queries
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
    """LangGraph node that performs web research using the native Google Search API tool.

    Executes a web search using the native Google Search API tool in combination with Gemini 2.0 Flash.

    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including search API settings

    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
    """
    # Configure
    configurable = Configuration.from_runnable_config(config)
    formatted_prompt = web_searcher_instructions.format(
        current_date=get_current_date(),
        research_topic=state["search_query"],
    )

    # Uses the google genai client as the langchain client doesn't return grounding metadata
    response = genai_client.models.generate_content(
        model=configurable.query_generator_model,
        contents=formatted_prompt,
        config={
            "tools": [{"google_search": {}}],
            "temperature": 0,
        },
    )
    # resolve the urls to short urls for saving tokens and time
    resolved_urls = resolve_urls(
        response.candidates[0].grounding_metadata.grounding_chunks, state["id"]
    )
    # Gets the citations and adds them to the generated text
    citations = get_citations(response, resolved_urls)
    modified_text = insert_citation_markers(response.text, citations)
    sources_gathered = [item for citation in citations for item in citation["segments"]]

    return {
        "sources_gathered": sources_gathered,
        "search_query": [state["search_query"]],
        "web_research_result": [modified_text],
    }


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """LangGraph node that identifies knowledge gaps and generates potential follow-up queries.

    Analyzes the current summary to identify areas for further research and generates
    potential follow-up queries. Uses structured output to extract
    the follow-up query in JSON format.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """
    configurable = Configuration.from_runnable_config(config)
    # Increment the research loop count and get the reasoning model
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reasoning_model = state.get("reasoning_model") or configurable.reasoning_model

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )
    # init Reasoning Model
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
    """LangGraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.

    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_research_loops setting

    Returns:
        String literal indicating the next node to visit ("web_research" or "finalize_summary")
    """
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
    """LangGraph node that finalizes the research summary.

    Prepares the final output by deduplicating and formatting sources, then
    combining them with the running summary to create a well-structured
    research report with proper citations.

    Args:
        state: Current graph state containing the running summary and sources gathered

    Returns:
        Dictionary with state update, including running_summary key containing the formatted final summary with sources
    """
    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model") or configurable.reasoning_model

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )

    # init Reasoning Model, default to Gemini 2.5 Flash
    llm = ChatGoogleGenerativeAI(
        model=reasoning_model,
        temperature=0,
        max_retries=2,
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    result = llm.invoke(formatted_prompt)

    # Replace the short urls with the original urls and add all used urls to the sources_gathered
    unique_sources = []
    for source in state["sources_gathered"]:
        if source["short_url"] in result.content:
            result.content = result.content.replace(
                source["short_url"], source["value"]
            )
            unique_sources.append(source)

    return {
        "messages": [AIMessage(content=result.content)],
        "sources_gathered": unique_sources,
    }

# --- New Agent Nodes Start ---

def manager_agent_node(state: OverallState, config: RunnableConfig) -> OverallState:
    configurable = Configuration.from_runnable_config(config)
    llm = ChatGoogleGenerativeAI(
        model=configurable.query_generator_model, # Using query_generator_model for now
        temperature=0.7, # May need adjustment
        api_key=os.getenv("GEMINI_API_KEY"),
    )

    user_request = get_research_topic(state["messages"])
    if not user_request: # Default if no user request found
        user_request = "Develop a new website."

    # Combine the static system prompt with the dynamic user request for the LLM
    prompt_input = manager_agent_prompt + f"\n\nHere is the user's specific goal:\n{user_request}"

    response = llm.invoke(prompt_input)

    try:
        # Assuming response.content is a JSON string
        parsed_response = json.loads(response.content)
    except json.JSONDecodeError as e:
        error_message = f"Error: Manager agent output was not valid JSON. Content: {response.content}. Error: {e}"
        # Update state to reflect error and stop further processing by this path
        return {
            "next_agent_to_call": "ERROR", # Signal error
            "manager_instruction": "Failed to parse manager agent output.",
            # Add an AIMessage to log the error in the message history
            "messages": state.get("messages", []) + [AIMessage(content=error_message)]
        }

    next_agent = parsed_response.get("next_agent_to_call")
    instruction = parsed_response.get("manager_instruction")
    manager_ai_message_content = ""

    if next_agent == "USER_CLARIFICATION":
        manager_ai_message_content = f"Manager: {instruction}"
    else:
        manager_ai_message_content = f"Manager decision: Next agent is {next_agent}. Instruction: {instruction}"

    manager_ai_message = AIMessage(content=manager_ai_message_content)

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

    return {
        "dtc_website_manager_output": response.content,
        "messages": state.get("messages", []) + [AIMessage(content=f"DtC Website Manager Output: {response.content}")]
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

    return {
        "ui_designer_output": response.content,
        "messages": state.get("messages", []) + [AIMessage(content=f"UI Designer Output: {response.content}")]
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

    return {
        "copywriter_output": response.content,
        "messages": state.get("messages", []) + [AIMessage(content=f"Copywriter Output: {response.content}")]
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

    return {
        "developer_output": response.content,
        "messages": state.get("messages", []) + [AIMessage(content=f"Developer Output: {response.content}")]
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

    return {
        "asset_creator_output": response.content,
        "messages": state.get("messages", []) + [AIMessage(content=f"Asset Creator Output: {response.content}")]
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
    """
    Determines the next node to call based on the manager's decision.
    Routes to END if the agent name is "END", "ERROR", or not recognized, or "USER_CLARIFICATION".
    """
    next_agent_name = state.get("next_agent_to_call")

    # If manager asks for clarification, the graph should wait for new user input.
    # For now, we treat this as an effective end-point for the current automated run.
    # The overall application orchestrator would handle obtaining user input and reinvoking.
    if next_agent_name == "USER_CLARIFICATION":
        print(f"Manager is requesting user clarification. Instruction: {state.get('manager_instruction')}")
        return END # Or a special node if we want to handle this state differently within the graph. For now, END.

    if not next_agent_name or next_agent_name.upper() == "END" or next_agent_name.upper() == "ERROR":
        if next_agent_name and next_agent_name.upper() == "ERROR":
            # Optionally, log this event more formally if a logging system is in place
            print(f"Manager agent indicated an error or unparsable output for state: {state.get('messages')[-2:]}. Ending graph.")
        elif not next_agent_name:
            print(f"No next_agent_to_call specified. Ending graph. State: {state.get('messages')[-2:]}")
        return END

    node_name = AGENT_TO_NODE_MAP.get(next_agent_name)

    if node_name:
        # print(f"Routing from manager to: {node_name}") # Debugging log
        return node_name
    else:
        # Optionally, log this event
        print(f"Error: Unrecognized agent name '{next_agent_name}' received from manager. Ending graph. State: {state.get('messages')[-2:]}")
        return END
# --- End Manager Agent Routing Logic ---


# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# --- Define Nodes for the main agent team workflow ---
builder.add_node("manager_agent", manager_agent_node)
builder.add_node("dtc_website_manager", dtc_website_manager_node)
builder.add_node("ui_designer", ui_designer_node)
builder.add_node("copywriter", copywriter_node)
builder.add_node("developer", developer_node)
builder.add_node("asset_creator", asset_creator_node)

# --- Define Nodes for the (optional) research sub-flow ---
# These nodes are part of a separate flow and are not directly connected to START
# unless explicitly called by an agent (e.g., if a future tool/agent uses 'generate_query').
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)


# --- Define Edges for the main agent team workflow ---

# Set the entrypoint to the manager agent
builder.add_edge(START, "manager_agent")

# Conditional routing from the manager agent
# This uses the AGENT_TO_NODE_MAP to create a dictionary of { "agent_node_name": "agent_node_name" }
# and adds the END state to it. USER_CLARIFICATION also routes to END for the graph's internal flow.
conditional_routes_map = {node_name: node_name for node_name in AGENT_TO_NODE_MAP.values()}
conditional_routes_map[END] = END
conditional_routes_map["USER_CLARIFICATION"] = END # Manager asking for clarification effectively ends this run.

builder.add_conditional_edges(
    "manager_agent",
    route_to_next_agent,
    conditional_routes_map
)

# Edges from specialist agents back to the manager agent
for node_name in AGENT_TO_NODE_MAP.values():
    builder.add_edge(node_name, "manager_agent")


# --- Edges for the (optional) research sub-flow ---
# These define the connections for the research sub-graph.
# This sub-graph is currently orphaned from the START node of the main flow.
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
builder.add_edge("web_research", "reflection")
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)
builder.add_edge("finalize_answer", END) # This END is for the research flow specifically


# Compile the graph
graph = builder.compile(name="multi-agent-team-workflow") # Renamed graph
