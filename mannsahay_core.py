import os
import uuid
from typing import TypedDict, Annotated
import operator
import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

# Load environment variables (API Key)
load_dotenv()

# --- LangGraph State Definition ---

class AgentState(TypedDict):
    """The state used by the LangGraph application."""
    messages: Annotated[list[BaseMessage], operator.add]

# --- LangGraph and Chatbot Functions ---

# Use a module-level dictionary to simulate a checkpointer for LangGraph threads
# In a real production environment, this should be replaced by Redis, a database, or a persistent file system.
# MemorySaver is fine for a simple FastAPI server for testing/MVP.
MEMORY_CHECKPOINTER = MemorySaver()

# NOTE: The @st.cache_resource is removed. Caching is now up to the FastAPI server/process.
def create_graph(system_prompt: str):
    """Creates and compiles the chatbot graph."""
    # Ensure GOOGLE_API_KEY is set
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")
        
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.9)
    
    def chatbot_node(state: AgentState):
        """The core LLM invocation node."""
        # The system prompt is prepended to the messages list for each invocation
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("chatbot", chatbot_node)
    graph_builder.set_entry_point("chatbot")
    graph_builder.add_edge("chatbot", END)
    
    # Use the global MemorySaver
    return graph_builder.compile(checkpointer=MEMORY_CHECKPOINTER)


def get_or_create_thread_id(user_uuid: str, zone_name: str) -> str:
    """Generates the unique thread ID for the user and zone."""
    return f"{user_uuid}-{zone_name}"

def initialize_chat_thread(thread_id: str, graph, system_prompt: str) -> list[dict]:
    """Initializes the chat history if it doesn't exist."""
    config = {"configurable": {"thread_id": thread_id}}
    
    # Check if a state already exists for this thread ID
    if not graph.get_state(config):
        # Initialize with the AI's first message
        graph.update_state(
            config,
            {"messages": [AIMessage(content="Hello! 👋")]}
        )
    
    # Return the full history
    return get_chat_history(thread_id, graph)


def get_chat_history(thread_id: str, graph) -> list[dict]:
    """Retrieves the chat history for a given thread ID."""
    config = {"configurable": {"thread_id": thread_id}}
    state = graph.get_state(config)
    history = state.values.get("messages", [])
    
    # Convert BaseMessage objects to a serializable list of dicts for the API
    serializable_history = []
    for msg in history:
        serializable_history.append({
            "type": msg.type,
            "content": msg.content
        })
    return serializable_history


def invoke_chat(thread_id: str, prompt: str, graph) -> list[dict]:
    """Invokes the LangGraph agent with a new human message and returns the updated history."""
    config = {"configurable": {"thread_id": thread_id}}
    human_message = HumanMessage(content=prompt)
    
    # Run the graph
    event = graph.invoke({"messages": [human_message]}, config)
    
    # The LangGraph automatically saves the updated state
    return get_chat_history(thread_id, graph)

# --- Other Utility Functions ---

# Note: You need to create an 'affirmations.py' file containing a list of strings
# Example affirmations.py: affirmations = ["I am strong.", "I can handle this.", "Today is a fresh start."]
try:
    from affirmations import affirmations
except ImportError:
    affirmations = ["I am strong.", "I can handle this.", "Today is a a fresh start."]


def get_daily_quote() -> str:
    """Calculates the daily quote based on the current date."""
    date_obj = datetime.date.today()
    index = date_obj.toordinal() % len(affirmations)
    return affirmations[index]
