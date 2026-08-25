import os
import asyncio
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# ---------------------------------------------------------
# LOAD ENVIRONMENT
# ---------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------
# IMPORT CORE CHAT LOGIC
# ---------------------------------------------------------

from mannsahay_core import (
    create_graph,
    get_or_create_thread_id,
    initialize_chat_thread,
    invoke_chat,
    get_daily_quote,
)


# ---------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------

app = FastAPI(
    title="MannSahay Backend API",
    description="FastAPI service for the LangGraph chatbot and utilities.",
    version="2.0.0",
)


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# GEMINI API KEY MANAGEMENT
# ---------------------------------------------------------

def load_api_keys() -> List[str]:
    """
    Loads Gemini API keys from environment variables.

    Supported format:

        GOOGLE_API_KEY
        GOOGLE_API_KEY_1
        GOOGLE_API_KEY_2
        GOOGLE_API_KEY_3
        ...

    The plain GOOGLE_API_KEY is also supported for backwards
    compatibility.
    """

    keys = []

    # First, check the numbered keys.
    index = 1

    while True:
        key = os.getenv(f"GOOGLE_API_KEY_{index}")

        if not key:
            break

        key = key.strip()

        if key and key not in keys:
            keys.append(key)

        index += 1

    # Backwards compatibility with your original setup.
    original_key = os.getenv("GOOGLE_API_KEY")

    if original_key:
        original_key = original_key.strip()

        if original_key and original_key not in keys:
            keys.insert(0, original_key)

    return keys


GEMINI_API_KEYS = load_api_keys()


if not GEMINI_API_KEYS:
    raise RuntimeError(
        "No Gemini API keys found. "
        "Set GOOGLE_API_KEY or GOOGLE_API_KEY_1, "
        "GOOGLE_API_KEY_2, etc."
    )


print(f"🔑 Loaded {len(GEMINI_API_KEYS)} Gemini API key(s)")


# ---------------------------------------------------------
# GRAPH CACHE
# ---------------------------------------------------------

# One graph per API key.
#
# Example:
#
#   GRAPH_CACHE[0] -> Gemini graph using API key 1
#   GRAPH_CACHE[1] -> Gemini graph using API key 2
#
GRAPH_CACHE = {}


# Keep track of keys that are temporarily unavailable.
#
# Example:
#
#   KEY_COOLDOWN[0] = timestamp
#
KEY_COOLDOWN = {}


# Prevent multiple requests from simultaneously changing
# environment variables while graphs are being created.
GRAPH_CREATION_LOCK = asyncio.Lock()


# ---------------------------------------------------------
# QUOTA / RATE LIMIT DETECTION
# ---------------------------------------------------------

def is_key_error(error: Exception) -> bool:
    """
    Determines whether an exception is likely related to the
    Gemini API key, quota, or rate limit.

    We only rotate keys for these types of failures.

    Normal application errors should NOT automatically cause
    key rotation.
    """

    error_text = str(error).lower()

    key_error_terms = [
        "429",
        "resource_exhausted",
        "resource exhausted",
        "quota",
        "rate limit",
        "rate_limit",
        "too many requests",
        "permission denied",
        "api key",
        "invalid api key",
        "unauthenticated",
        "authentication",
    ]

    return any(term in error_text for term in key_error_terms)


# ---------------------------------------------------------
# CREATE GRAPH FOR A SPECIFIC KEY
# ---------------------------------------------------------

def create_graph_for_key(api_key: str):
    """
    Creates a LangGraph instance using a specific Gemini key.

    Your existing mannsahay_core.py reads GOOGLE_API_KEY from
    the environment, so we temporarily set it before creating
    the graph.
    """

    old_key = os.environ.get("GOOGLE_API_KEY")

    try:
        os.environ["GOOGLE_API_KEY"] = api_key

        graph = create_graph(
            system_prompt=(
                "You are MannSahay, a warm, empathetic and supportive "
                "mental-wellbeing companion. Listen carefully to the user, "
                "validate their feelings without being judgmental, and "
                "provide practical, gentle suggestions when appropriate. "
                "Do not pretend to be a human or a licensed professional. "
                "If the user appears to be in immediate danger or at risk "
                "of harming themselves or someone else, encourage them to "
                "seek immediate help from a trusted person or appropriate "
                "emergency/professional services."
            )
        )

        return graph

    finally:
        # Restore whatever environment value existed before.
        if old_key is not None:
            os.environ["GOOGLE_API_KEY"] = old_key
        else:
            os.environ.pop("GOOGLE_API_KEY", None)


# ---------------------------------------------------------
# GET GRAPH
# ---------------------------------------------------------

async def get_graph(key_index: int):
    """
    Returns a cached graph for the requested key.

    Graph creation happens only once per key.
    """

    if key_index in GRAPH_CACHE:
        return GRAPH_CACHE[key_index]

    async with GRAPH_CREATION_LOCK:

        # Another request may have created it while we waited.
        if key_index in GRAPH_CACHE:
            return GRAPH_CACHE[key_index]

        api_key = GEMINI_API_KEYS[key_index]

        print(f"🧠 Creating Gemini graph for key #{key_index + 1}")

        try:
            graph = create_graph_for_key(api_key)

            GRAPH_CACHE[key_index] = graph

            print(f"✅ Gemini graph ready for key #{key_index + 1}")

            return graph

        except Exception as e:
            print(
                f"❌ Failed to create graph for key "
                f"#{key_index + 1}: {e}"
            )
            raise


# ---------------------------------------------------------
# FIND WORKING GRAPH
# ---------------------------------------------------------

async def invoke_with_failover(
    thread_id: str,
    prompt: str,
):
    """
    Sends a message using Gemini.

    If the current key hits a quota/rate-limit/key error,
    automatically tries the next available key.

    Conversation history is preserved because the same
    thread_id is used for every graph.
    """

    last_error: Optional[Exception] = None

    for key_index in range(len(GEMINI_API_KEYS)):

        # -------------------------------------------------
        # Check temporary cooldown
        # -------------------------------------------------

        cooldown_until = KEY_COOLDOWN.get(key_index)

        if cooldown_until is not None:
            import time

            if time.time() < cooldown_until:
                print(
                    f"⏭️ Skipping key #{key_index + 1} "
                    f"(temporarily unavailable)"
                )
                continue

            # Cooldown expired.
            KEY_COOLDOWN.pop(key_index, None)

        try:
            graph = await get_graph(key_index)

            print(
                f"🤖 Trying Gemini key #{key_index + 1}"
            )

            # invoke_chat is synchronous, so run it in a thread
            # instead of blocking FastAPI's event loop.
            result = await asyncio.to_thread(
                invoke_chat,
                thread_id,
                prompt,
                graph,
            )

            print(
                f"✅ Request completed using key "
                f"#{key_index + 1}"
            )

            return result

        except Exception as e:

            last_error = e

            print(
                f"⚠️ Gemini key #{key_index + 1} failed: {e}"
            )

            # -------------------------------------------------
            # Only rotate when it looks like a key/quota issue.
            # -------------------------------------------------

            if is_key_error(e):

                import time

                # Temporarily avoid this key.
                #
                # 60 seconds is enough for basic rate-limit
                # recovery while still allowing it to return.
                KEY_COOLDOWN[key_index] = time.time() + 60

                print(
                    f"🔄 Key #{key_index + 1} unavailable. "
                    f"Trying another key..."
                )

                continue

            # For unrelated errors, don't silently switch
            # API keys.
            raise e

    # ---------------------------------------------------------
    # No key worked
    # ---------------------------------------------------------

    raise RuntimeError(
        "All configured Gemini API keys are currently "
        "unavailable."
    ) from last_error


# ---------------------------------------------------------
# REQUEST MODELS
# ---------------------------------------------------------

class ChatInput(BaseModel):
    user_uuid: str
    zone_name: str
    prompt: str


class ChatHistoryInput(BaseModel):
    user_uuid: str
    zone_name: str


# ---------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------

@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "MannSahay Backend API",
        "gemini_keys_configured": len(GEMINI_API_KEYS),
        "graphs_loaded": len(GRAPH_CACHE),
    }


# ---------------------------------------------------------
# HEALTH / KEY STATUS
# ---------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "gemini_keys_configured": len(GEMINI_API_KEYS),
        "graphs_loaded": len(GRAPH_CACHE),
    }


# ---------------------------------------------------------
# DAILY QUOTE
# ---------------------------------------------------------

@app.get("/quote")
async def get_quote():
    """
    Returns the daily affirmation/quote.
    """

    try:
        quote = get_daily_quote()

        return {
            "quote": quote
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not retrieve daily quote: {str(e)}",
        )


# ---------------------------------------------------------
# GET / CREATE CHAT THREAD
# ---------------------------------------------------------

@app.post("/chat/history")
async def get_history(data: ChatHistoryInput):
    """
    Gets the existing conversation history.

    If the conversation doesn't exist yet, initializes it.
    """

    thread_id = get_or_create_thread_id(
        data.user_uuid,
        data.zone_name,
    )

    # Use the first graph for initialization.
    graph = await get_graph(0)

    try:
        history = await asyncio.to_thread(
            initialize_chat_thread,
            thread_id,
            graph,
            (
                "You are MannSahay, a warm, empathetic and supportive "
                "mental-wellbeing companion. Listen carefully to the user "
                "and respond naturally and supportively."
            ),
        )

        return {
            "thread_id": thread_id,
            "messages": history,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not retrieve chat history: {str(e)}",
        )


# ---------------------------------------------------------
# SEND MESSAGE
# ---------------------------------------------------------

@app.post("/chat")
async def invoke_new_message(data: ChatInput):
    """
    Sends a new message to MannSahay.

    Automatically switches to another authorized Gemini API
    key if the current one encounters a quota/rate-limit error.
    """

    if not data.prompt.strip():
        raise HTTPException(
            status_code=400,
            detail="Prompt cannot be empty.",
        )

    thread_id = get_or_create_thread_id(
        data.user_uuid,
        data.zone_name,
    )

    try:

        history = await invoke_with_failover(
            thread_id=thread_id,
            prompt=data.prompt,
        )

        return {
            "thread_id": thread_id,
            "messages": history,
        }

    except HTTPException:
        raise

    except Exception as e:

        print(f"💥 Chat request failed: {e}")

        raise HTTPException(
            status_code=503,
            detail=(
                "MannSahay is temporarily unavailable. "
                "Please try again in a moment."
            ),
        )


# ---------------------------------------------------------
# INITIALIZE CHAT
# ---------------------------------------------------------

@app.post("/chat/init")
async def initialize_chat(data: ChatHistoryInput):
    """
    Explicitly initializes a chat thread.
    """

    thread_id = get_or_create_thread_id(
        data.user_uuid,
        data.zone_name,
    )

    try:

        graph = await get_graph(0)

        history = await asyncio.to_thread(
            initialize_chat_thread,
            thread_id,
            graph,
            (
                "You are MannSahay, a warm, empathetic and supportive "
                "mental-wellbeing companion."
            ),
        )

        return {
            "thread_id": thread_id,
            "messages": history,
        }

    except Exception as e:

        print(f"❌ Chat initialization failed: {e}")

        raise HTTPException(
            status_code=500,
            detail="Could not initialize chat.",
        )


# ---------------------------------------------------------
# STARTUP
# ---------------------------------------------------------

@app.on_event("startup")
async def startup_event():

    print("")
    print("==============================================")
    print("        🌿 MannSahay Backend Starting        ")
    print("==============================================")
    print(
        f"🔑 Gemini keys configured: "
        f"{len(GEMINI_API_KEYS)}"
    )

    # We deliberately don't create every graph here.
    #
    # Graphs are created lazily when needed.
    # This makes deployment faster and avoids unnecessary
    # initialization.
    #
    # The first graph will be created on the first request.

    print("🚀 Backend ready")
    print("==============================================")
    print("")


# ---------------------------------------------------------
# RUN SERVER
# ---------------------------------------------------------

if __name__ == "__main__":

    import uvicorn

    port = int(os.getenv("PORT", 8080))

    print(
        f"🚀 Starting MannSahay Backend on port {port}"
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
    )
