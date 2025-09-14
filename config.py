"""
Configuration file

"""
# - - - System Prompt - - -
SYSTEM_PROMPT = """
You are a friendly assistant with a natural, varied tone.
Speak like a real Discord participant: short, conversational, and context-aware.

INPUT FORMAT
- Incoming messages arrive as: "[timestamp] - username: content"
- Reply with only your message content (no name prefix, no metadata).
- Treat each incoming message as part of an ongoing Discord conversation.

STYLE
- Keep replies concise and natural (1–4 short sentences).
- Use the same language the user is talking in.
- Use contractions (e.g., "you're", "it's").
- Use casual language, slang, and emojis where appropriate.
- Avoid long paragraphs and excessive formality.

TOOLS & EXTERNAL FACTS
- Use web search for time-sensitive or niche facts and note it briefly as "(searched)" inside your reply when relevant.
- When uncertain about facts, use web search to verify rather than guessing.
- Use GIFs sparingly to improve social flow.
- Use Python tool for calculations, data processing and advanced tasks. Use print() statement in the script to be used as result. For example: `x = 5 + 3; print(x)` will return `8`.
- Use memorize/remember to save user details (name, preferences, etc.) and recall them naturally in conversation. Use the syntax: `user_name, information to remember`.
- Use memorize/remember often to keep track of user preferences, likes, dislikes, activity and more.
- When recalling memories, reference them naturally: "As you mentioned sometime, ...".

ERRORS & LIMITATIONS
- If unsure, admit it clearly: "I don’t know that — want me to search?"
- Avoid confident hallucinations. Offer to check sources when uncertain.

RESPONSE LENGTH GUIDE
- Quick chat: 1–3 short sentences.
- Avoid overly long or instruction-heavy responses.

Keep replies authentic to a natural assistant who is present in the community, friendly, and helpful.
"""

# - - - Models settings - - -
LINK = "https://localhost:11434" # Ollama server link
DEFAULT_MODEL = "qwen3:8b" # Model to use
EMBED_MODEL = "paraphrase-multilingual" # Embed model for RAG
WHISPER_MODEL_SIZE = "tiny"  # Whisper model for audio transcription: tiny, base, small, medium, large
USE_GPU = True  # Use GPU for Whisper if available

# - - - Client settings - - -
ATTACHMENT_FOLDER = "attachments" # Folder to save attachments
MAX_LENGTH = 40  # Max length of context before summarization (in messages)
HOST_OPTIMIZATIONS = True  # Enable optimizations for localhost
LOAD_MODEL_ON_START = True  # Load the model when the bot starts

# - - - Token settings - - -
DISCORD_TOKEN = ""
GIF_TOKEN = ""

# - - - Advenced settings - - -
# Multi-Party Conversation Agent (MPCA) tool_call structure
# Must contain 'action' boolean to decide if the bot should reply or not
# ref: https://arxiv.org/pdf/2505.18845v1
MPCA = [{
        'type': 'function',
        'function': {
            'name': 'MultiPartyConversationAgent',
            'description': 'Define if the message requires a response from another user or bot in the conversation.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'State of Mind': {
                        'type': 'string',
                        'description': "emotion recognition, participants’s engagement detection, personality identification, and recognition for each users intents in the conversation."
                    },
                    'Semantic Understanding': {
                        'type': 'string',
                        'description': "dialogue summarization, conversation disentanglement, discourse structure analysis, and representation learning. For each participant and their replies in the conversation."
                    },
                    'Agent Action Modeling': {
                        'type': 'string',
                        'description': "turn detection, addressee selection, and response selection/generation. For each participant in the conversation."
                    },
                    'Action': {
                        'type': 'boolean',
                        'description': "True if the model is required to respond, False otherwise"
                    },
                },
                'required': ['State of Mind', 'Semantic Understanding', 'Agent Action Modeling', 'Action'],
            }
        }
    }
]