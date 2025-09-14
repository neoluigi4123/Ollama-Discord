"""
main.py
Runs the discord bot, handles messages, attachments, reactions...
"""
import time
from datetime import datetime
from pathlib import Path
import json
import magic
from PIL import Image
import os
import asyncio
import copy
import whisper

import discord
from discord import app_commands

import torch

from Llm import chat, summarize_chat, save_context, load
import Llm
import conf_module
import load_file
from rag_embedding import read_memory

SYSTEM_PROMPT = conf_module.load_conf('SYSTEM_PROMPT')
MAX_LENGTH = conf_module.load_conf('MAX_LENGTH') # Max length of context before summarization (in messages)
MPCA = conf_module.load_conf('MPCA') # Multi-Party Conversation Agent

USE_GPU = conf_module.load_conf('USE_GPU')

if USE_GPU:
    try:
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            gpu = torch.cuda.get_device_name(0)
            print(f"GPU detected: {gpu}")
            device = "cuda"
        else:
            print("No GPU detected, using CPU.")
            device = "cpu"
    except:
        device = "cpu"
else:
    device = "cpu"

context_path = Path("context.json")
prev_channel_name = ""

ATTACHMENT_FOLDER = conf_module.load_conf('ATTACHMENT_FOLDER')
if not os.path.exists(ATTACHMENT_FOLDER):
    os.makedirs(ATTACHMENT_FOLDER)

Llm.context = [{
    'role': 'system',
    'content': SYSTEM_PROMPT
}]

def split_message(message, max_length=2000) -> list:
    """
    Splits a message into chunks of at most max_length characters.

    Args:
        message (str): The message to split.
        max_length (int): The maximum length of each chunk (default is 2000).

    Returns:
        list: A list of message chunks.
    """
    chunks = []
    while len(message) > max_length:
        split_point = message.rfind("\n", 0, max_length)
        if split_point == -1:
            split_point = max_length  # If no newline, split at max_length
        chunks.append(message[:split_point])
        message = message[split_point:].lstrip("\n")
    chunks.append(message)
    return chunks

def format_elapsed(seconds: float) -> str:
    """
    Format elapsed time in a human-readable way.

    Args:
        seconds (float): The elapsed time in seconds.

    Returns:
        str: The formatted elapsed time.
    """
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    parts = []
    if days: parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours: parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes: parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if sec: parts.append(f"{sec} second{'s' if sec != 1 else ''}")
    return ", ".join(parts) if parts else "0 seconds"

def file_ext(filepath: str, user: str) -> str:
    """
    Handle file based on its extension and type.
    
    Args:
        filepath (str): Path to the file.
        user (str): User who uploaded the file.
    
    Returns:
        str: Result or error message.
    """
    try:
        filepath = os.path.abspath(filepath)
        mime_type = magic.from_file(filepath, mime=True)

        # If image, convert to png for vision models
        if mime_type.startswith("image"):
            with Image.open(filepath) as img:
                png_path = "/tmp/converted_image.png"
                img.save(png_path, "PNG")
                save_context(f"Image uploaded by {user}.", 'user', image_path=[png_path])

        # If audio, use whisper to transcribe
        elif mime_type.startswith("audio"):
            model_size = conf_module.load_conf('WHISPER_MODEL_SIZE')
            whisper_model = whisper.load_model(model_size, device=device)
            result = whisper_model.transcribe(filepath)
            transcription = result.get("text", "")
            save_context(f"Audio file uploaded by {user}. Transcription: {transcription}", "user")

        # If text-based file, load content
        else:
            file_content = load_file.load_file(filepath)
            save_context(f"File uploaded by {user}:\n{file_content}", "user")

    except Exception as e:
        return(f"[Error]: {e}")

def fetch_previous_chat() -> None:
    """
    Load previous chat context from context.json if it exists.

    Returns: None
    """
    if context_path.exists():
        content = context_path.read_text(encoding="utf-8").strip()
        if content:
            Llm.context = json.loads(content)
            elapsed = time.time() - context_path.stat().st_mtime

            last_msg = Llm.context[-1] if Llm.context else None
            if (
                last_msg
                and last_msg.get("role") == "system"
                and last_msg.get("content", "").startswith("You've been disconnected")
            ):
                Llm.context.pop()
            
            save_context(f"You've been disconnected for {format_elapsed(elapsed)}", 'system')

intents = discord.Intents.default()
intents.messages = True
intents.reactions = True
intents.members = True
intents.message_content = True
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

# On bot ready
@client.event
async def on_ready():
    if conf_module.load_conf('LOAD_MODEL_ON_START'):
        load() # load Llm
    fetch_previous_chat() # Get the time since last connection
    print(f"Logged in as {client.user}")

# On reacted message
@client.event
async def on_reaction_add(reaction, user):
    if user == client.user:
        return

    save_context(f"{user} reacted with {reaction.emoji} to message: {reaction.message.content}", role="system")

# Process each message with Llm
@client.event
async def on_message(msg):
    global prev_channel_name
    
    # Ignore messages conditions
    if msg.author == client.user or msg.content.startswith('/silent'):
        return
    
    # Reset system prompt just in case
    Llm.context[0] = {
        'role': 'system',
        'content': SYSTEM_PROMPT
    }

    # Channel change detection
    current_channel_name = msg.channel.name if hasattr(msg.channel, 'name') else "Direct Message"

    if prev_channel_name != current_channel_name:
        channel_desc = msg.channel.topic if hasattr(msg.channel, 'topic') else "No description"
        server_name = msg.guild.name if msg.guild else None
        if server_name:
            save_context(f"Now in {msg.guild.name}, {current_channel_name} channel. Description: {channel_desc}", 'system')
        else:
            save_context(f"Now in {current_channel_name} channel. Description: {channel_desc}", 'system')
        prev_channel_name = current_channel_name

    # Summarize chat if too long
    if len(Llm.context) > MAX_LENGTH:
        summarize_chat(15)
    
    # Append memory from RAG if new user
    if all('user' not in x or str(msg.author) not in x['user'] for x in Llm.context):
        memory = read_memory(5, str(msg.author), msg.content)
        if memory:
            save_context(f"(Remembered from past conversations) {memory}", 'system')
    
    content = msg.content

    # Replace mentions with usernames, usefull for Llm understanding
    if msg.mentions:
        for user in msg.mentions:
            content = content.replace(f"<@{user.id}>", f"@{user.name}")
    
    # handle attachments
    if msg.attachments:
        for attachment in msg.attachments:
            try:
                path = os.path.join(ATTACHMENT_FOLDER, attachment.filename)

                with open(path, "wb") as file_object:
                    await attachment.save(file_object)

                file_ext(path, msg.author)
            except:
                pass

    # handle replied message with attachments
    if msg.reference:
        replied_message = await msg.channel.fetch_message(msg.reference.message_id)
        replied_content = replied_message.content
        replied_author = replied_message.author
        replied_attachments = replied_message.attachments

        if replied_attachments:
            for attachment in replied_attachments:
                try:
                    path = os.path.join(ATTACHMENT_FOLDER, attachment.filename)

                    with open(path, "wb") as file:
                        await attachment.save(file)

                    file_ext(path, replied_author)
                except:
                    pass
        
        save_context(f"{msg.author} replied to a message by {replied_author}: {replied_content}")
         
    prompt = f"{datetime.now().strftime("%H:%M")} - {msg.author}: {content}"

    if msg.guild == None: # Direct message
        # Must always reply for each message (LLM/GPT like)
        async with msg.channel.typing():
            reply = await asyncio.to_thread(chat, prompt, custom_field=f'user, {msg.author}')

    else: # Server message
        # Simulate real conversation flow-
        base_chat_context = copy.deepcopy(Llm.context) # Store context for context swap

        chat_context = Llm.context[1:]

        Llm.context = [{
                'role': 'system',
                'content': "You're a Multi-Party Conversation Agent. Decide if you should reply to the user or not based on the conversation context. Always reply using tool_calls with the proper JSON structure: State_of_Mind, Semantic Understanding, Agent Action Modeling, and Action."
            },
            {
                'role': 'user',
                'content': str(chat_context)
            }]

        mpca_reply = await asyncio.to_thread(chat, prompt, thinking = 'False', custom_tools=MPCA)

        for tools in mpca_reply:
            action = tools['function']['arguments'].get('Action')

        # bring context back
        Llm.context = base_chat_context

        if action == True:
            async with msg.channel.typing():
                reply = await asyncio.to_thread(chat, prompt, custom_field=f'user, {msg.author}')
        else:
            return

    # length check for discord message limit
    if len(reply) > 2000:
        message_chunks = split_message(reply, max_length=2000)
        for chunk in message_chunks:
            await msg.channel.send(chunk)
    else:
        await msg.channel.send(reply)

# Run the bot
client.run(conf_module.load_conf('DISCORD_TOKEN'))