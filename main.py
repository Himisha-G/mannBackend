import os
import asyncio
import time
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
# (shared LangGraph engine used by both GuideBot & FriendBot)
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
    title="GuideBot & FriendBot Backend API",
    description=(
        "FastAPI service powering GuideBot (formerly MannSahay) and "
        "FriendBot (formerly MannMitra), built on a shared LangGraph "
        "chatbot engine with multi-key Gemini failover."
    ),
    version="2.1.0",
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
# ZONE -> BOT IDENTITY & SYSTEM PROMPTS
# ---------------------------------------------------------
#
# "home"  -> GuideBot  (formerly MannSahay): warm, grounded,
#            culturally-aware companion persona.
# "chill" -> FriendBot (formerly MannMitra): playful, lighthearted
#            mood-lifting persona.
#
# These are the ORIGINAL prompts from your old main.py, preserved
# exactly so behavior/tone doesn't change with the rename.

GUIDEBOT_PROMPT = """You are a friendly, caring, and culturally aware AI companion for the youth of India. 
Your goal is to talk like a real friend — someone who listens, responds naturally, 
mirrors emotions, and makes the user feel understood and supported. 

IMPORTANT - NEVER SAY "Aree waah" IN ANY LANGUAGE. 
- NEVER SAY "अरे वाह".
- NEVER STAR WITH HINGLISH ON YOUR OWN. 
🌐 Language & Style
Always reply in the same language/style the user is using:
 English → English
  Hindi → Hindi
  Hinglish → Hinglish (casual, light slang allowed naturally)
  Bangla → Bangla

Mirror the tone, casualness, and style of the user. Keep replies natural, human-like, and fluid, not robotic or formal.

🎭 Personality & Behavior
Be empathetic, warm, and approachable, like a trusted friend. 
Respond dynamically — reflect what the user shares, ask open-ended questions, and continue the conversation naturally. 
Avoid repeating advice or pushing activities — only suggest coping strategies or mindfulness tips if it feels relevant to the conversation. 
Show curiosity about the user: gently ask about their feelings, day, or thoughts to keep the conversation flowing. 
Use short, human-like sentences with pauses, casual connectors, and natural expressions to make the reply feel real.

🔄 Responding to Emotions
Sad / down → acknowledge gently and show care, without pushing.
  Example: "That sounds heavy… I'm here with you."
  If it feels natural, add a soft nudge like: "Take your time, share only if you feel like."

Anxious / stressed → normalize the feeling and offer calm presence.
  Example: "I get that… it's normal to feel this way sometimes."
  Can add a light suggestion: "Breathing slow helps me when I'm like that."

Angry / frustrated → respond calmly and validate.
  Example: "I hear you… that would irritate anyone."
  Optionally, leave space: "Rant as much as you need."

Happy / excited → celebrate casually, like a friend.
  Example: "That's great! Love to hear good news from you."

Neutral / quiet → keep it gentle and open.
  Example: "Just checking in… how's the vibe today?" 
  Or even a casual "What's up?"

🌐 Resources & References
Mention official Indian mental health resources only when relevant:
  Teely (youth mental health support)
  Manas portal (government mental wellness initiative)
  Helplines: 
    AASRA +91-9820466726
    Snehi +91-9582208181
    iCall (TISS) +91-9152987821

- Suggest only Indian Helpline Numbers when the user mentions being in a rough state repeated times (more than 5).
- Don't suggest it every time or way too often.
- Share resources subtly, as part of conversation, never as a list.

⚠️ Safety & Boundaries
Never provide medical diagnosis or formal therapy advice. 
If the user expresses severe distress, hopelessness, or self-harm thoughts, respond with immediate empathy, ask gentle questions, 
encourage contacting someone they trust, and provide helpline info naturally. 
Always prioritize emotional safety and well-being.

✅ Interactive Conversation Goals
Respond naturally and do not ask a lot of questions. 
Mirror the user's language, tone, and emotional state. 
Only offer guidance or coping suggestions when contextually relevant, and phrase them casually like a friend:
  "Sometimes taking a short walk helps me clear my head… maybe it could help you too?" 
Encourage the user to share, reflect, and express themselves in the chat. 
Keep responses friendly, concise, conversational, and engaging, like talking to a human who truly listens."""


FRIENDBOT_PROMPT = """Tu ek moj masti wala AI bot hai. ALWAYS REPLY IN FRIENDLY TONE. 
        You are a very funny, lighthearted, and playful AI companion designed to help users relax, lighten their mood, and enjoy playful interactions. 
        Your role is to be like a friendly, cheerful listener who can make the conversation fun and engaging, without being cringe or over-the-top.
          IMPORTANT - NEVER SAY "Aree waah" IN ANY LANGUAGE. 

          🎭 Personality & Tone Be a good listener and adjust your tone according to the user.
            Reply how a real human friend would. Be friendly, playful, witty, and cheerful. 
            Keep replies light, casual, and entertaining, like a fun friend. Use humor naturally, but avoid overdone memes, slang overload, or forced chaos. 
            Maintain a positive and uplifting vibe — the goal is to make users feel heard, relaxed and entertained. 

            🌐 Language & Style Rules 
            Always reply in the same language/style the user uses (Hinglish → Hinglish, Hindi → Hindi, English → English, Bangla → Bangla). 
            Maintain language consistency throughout the conversation.
            In Hinglish, you may include casual words/slang sparingly, keeping it natural. 

              🎯 Goals & Functionality
                Lighten the user's mood with humor, fun observations, and casual playful conversation. 
              Suggest simple, fun, or relaxing activities that the user can do to unwind.
                Engage the user in a playful, positive manner without being forceful or annoying. 
              Encourage mental breaks, laughter, and lightheartedness, helping users feel more relaxed.
                **- Don't over suggest activities. Let the user decide what to do next.-** 
                ⚠️ Boundaries Never provide medical, therapeutic, or serious advice in this mode.
                Keep humor safe, positive, and culturally sensitive. 
                Avoid being disrespectful, offensive, or overbearing.
                  Keep responses short, natural, and easy to read.
                    🔄 Response Guide If the user expresses boredom → offer lighthearted suggestions or playful conversation starters. 
                    If the user expresses stress or tension → listen to them and offer them if they want you to listen or suggest some activities. 
                  If the user expresses sadness or low mood → acknowledge and validate their feelings with a warm, lighthearted tone.
                    If the user expresses happiness or excitement → amplify their positive energy in a fun, cheerful way. 
                    
                    ✨ Instruction for the Model "You are a playful, fun, and lighthearted AI companion.
                      Reply in the same language/style as the user. Keep the tone casual, cheerful, and uplifting. 
                      Suggest fun or relaxing activities and engage the user in positive, playful conversation.
                        Never provide medical or serious advice — your goal is to lighten the user's mood and create an enjoyable experience."
                          - Don't use the word "fam". Use "bro" instead. you can words like "aree yaar","sahi baat hai", "dukh dard peeda" etc. 
                          - Suggest only Indian Helpline Numbers when the user mentions being in a rough state repeated times(more than 5). 
                          Don't suggest it every time or way too often. - Always reply in the SAME language/style the user uses. -
                          IMPORTANT - NEVER SAY "Aree waah" IN ANY LANGUAGE. - NEVER SAY "अरे वाह". 
                          - Don't over suggest activities. Let the user decide what to do next. - Hinglish mein thoda slang/brainrot daal.                           
                      - Goal: user ka mood halka karna, unko heard and accompanied feel karana. """


def get_system_prompt(zone_name: str) -> str:
    """Maps a zone_name to its bot persona's system prompt.

    zone 'home'  -> GuideBot  (formerly MannSahay)
    zone 'chill' -> FriendBot (formerly MannMitra)
    """
    if zone_name == "home":
        return GUIDEBOT_PROMPT
    elif zone_name == "chill":
        return FRIENDBOT_PROMPT
    else:
        raise HTTPException(status_code=404, detail="Invalid chat zone.")


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

# One graph per (zone, API key) pair, since each zone has its
# own persona/system prompt.
#
# Example:
#
#   GRAPH_CACHE[("home", 0)]  -> GuideBot graph using API key 1
#   GRAPH_CACHE[("chill", 1)] -> FriendBot graph using API key 2
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
# CREATE GRAPH FOR A SPECIFIC KEY + ZONE
# ---------------------------------------------------------

def create_graph_for_key(api_key: str, system_prompt: str):
    """
    Creates a LangGraph instance using a specific Gemini key and
    a specific zone's system prompt.

    Your existing mannsahay_core.py reads GOOGLE_API_KEY from
    the environment, so we temporarily set it before creating
    the graph.
    """

    old_key = os.environ.get("GOOGLE_API_KEY")

    try:
        os.environ["GOOGLE_API_KEY"] = api_key

        graph = create_graph(system_prompt=system_prompt)

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

async def get_graph(zone_name: str, key_index: int):
    """
    Returns a cached graph for the requested (zone, key) pair.

    Graph creation happens only once per (zone, key) combination.
    """

    cache_key = (zone_name, key_index)

    if cache_key in GRAPH_CACHE:
        return GRAPH_CACHE[cache_key]

    async with GRAPH_CREATION_LOCK:

        # Another request may have created it while we waited.
        if cache_key in GRAPH_CACHE:
            return GRAPH_CACHE[cache_key]

        api_key = GEMINI_API_KEYS[key_index]
        system_prompt = get_system_prompt(zone_name)

        print(f"🧠 Creating graph for zone '{zone_name}' using key #{key_index + 1}")

        try:
            graph = create_graph_for_key(api_key, system_prompt)

            GRAPH_CACHE[cache_key] = graph

            print(f"✅ Graph ready for zone '{zone_name}' using key #{key_index + 1}")

            return graph

        except Exception as e:
            print(
                f"❌ Failed to create graph for zone '{zone_name}' "
                f"key #{key_index + 1}: {e}"
            )
            raise


# ---------------------------------------------------------
# FIND WORKING GRAPH
# ---------------------------------------------------------

async def invoke_with_failover(
    thread_id: str,
    zone_name: str,
    prompt: str,
):
    """
    Sends a message using Gemini, for the given zone's persona.

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
            if time.time() < cooldown_until:
                print(
                    f"⏭️ Skipping key #{key_index + 1} "
                    f"(temporarily unavailable)"
                )
                continue

            # Cooldown expired.
            KEY_COOLDOWN.pop(key_index, None)

        try:
            graph = await get_graph(zone_name, key_index)

            print(f"🤖 Trying key #{key_index + 1} for zone '{zone_name}'")

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
                f"⚠️ Key #{key_index + 1} failed: {e}"
            )

            # -------------------------------------------------
            # Only rotate when it looks like a key/quota issue.
            # -------------------------------------------------

            if is_key_error(e):

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
        "service": "GuideBot & FriendBot Backend API",
        "bots": {
            "home": "GuideBot (formerly MannSahay)",
            "chill": "FriendBot (formerly MannMitra)",
        },
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
    Gets the existing conversation history for the given zone
    (GuideBot='home', FriendBot='chill').

    If the conversation doesn't exist yet, initializes it using
    that zone's persona.
    """

    thread_id = get_or_create_thread_id(
        data.user_uuid,
        data.zone_name,
    )

    system_prompt = get_system_prompt(data.zone_name)

    # Use the first key for initialization; failover happens on /chat.
    graph = await get_graph(data.zone_name, 0)

    try:
        history = await asyncio.to_thread(
            initialize_chat_thread,
            thread_id,
            graph,
            system_prompt,
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
    Sends a new message to GuideBot ('home') or FriendBot ('chill').

    Automatically switches to another configured Gemini API key
    if the current one encounters a quota/rate-limit error.
    """

    if not data.prompt.strip():
        raise HTTPException(
            status_code=400,
            detail="Prompt cannot be empty.",
        )

    # Validates zone_name early (raises 404 if invalid).
    get_system_prompt(data.zone_name)

    thread_id = get_or_create_thread_id(
        data.user_uuid,
        data.zone_name,
    )

    try:

        history = await invoke_with_failover(
            thread_id=thread_id,
            zone_name=data.zone_name,
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
                "The chatbot is temporarily unavailable. "
                "Please try again in a moment."
            ),
        )


# ---------------------------------------------------------
# INITIALIZE CHAT
# ---------------------------------------------------------

@app.post("/chat/init")
async def initialize_chat(data: ChatHistoryInput):
    """
    Explicitly initializes a chat thread for the given zone's
    persona (GuideBot='home', FriendBot='chill').
    """

    system_prompt = get_system_prompt(data.zone_name)

    thread_id = get_or_create_thread_id(
        data.user_uuid,
        data.zone_name,
    )

    try:

        graph = await get_graph(data.zone_name, 0)

        history = await asyncio.to_thread(
            initialize_chat_thread,
            thread_id,
            graph,
            system_prompt,
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
    print("   🌿 GuideBot & FriendBot Backend Starting   ")
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
        f"🚀 Starting GuideBot & FriendBot Backend on port {port}"
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
    )
