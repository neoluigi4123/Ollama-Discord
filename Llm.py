"""
Llm.py
Handles chat, tool calling, context saving/loading, model loading...
"""
from ollama import Client
from ollama._types import ResponseError
import json
import os

import conf_module
from web_search import browse, gif
import scripting
from rag_embedding import write_memory

# Constant:
LINK = conf_module.load_conf('LINK')
DEFAULT_MODEL = conf_module.load_conf('DEFAULT_MODEL')

if conf_module.load_conf('HOST_OPTIMIZATIONS'):
    if "localhost" in LINK or "127.0.0.1" in LINK:
        os.environ["OLLAMA_MAX_LOADED_MODELS"] = "2"
        os.environ["OLLAMA_KEEP_ALIVE"] = "-1"
        os.environ["OLLAMA_FLASH_ATTENTION"] = "true"

ollama_client = Client(
    host=LINK
)

tools = [{
        'type': 'function',
        'function': {
            'name': 'browse',
            'description': 'Get online information',
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'The browse query or a direct link.'
                    }
                },
                'required': ['query']
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'gif',
            'description': 'Send a gif',
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {
                        'type': 'string',
                        'description': 'The query to search a gif about.'
                    }
                },
                'required': ['query']
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'python',
            'description': 'Use python for advanced task.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'script': {
                        'type': 'string',
                        'description': "The script for your task."
                    }
                },
                'required': ['script']
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'memorize',
            'description': 'Save information into memory',
            'parameters': {
                'type': 'object',
                'properties': {
                    'user': {
                        'type': 'string',
                        'description': "The user the information relate to."
                    },
                    'information': {
                        'type': 'string',
                        'description': "The information to save."
                    }
                },
                'required': ['user', 'information']
            }
        }
    }
]

context = []

logs = []

def load(model: str = DEFAULT_MODEL) -> str:
    """Infer with a model in streaming to load it. Returns when the model output the first token.
    
    Args:
        model (str, optional): The model to load. Defaults to DEFAULT_MODEL.

    Returns:
        str: "model loaded" when the model is loaded.
    """
    if model is None:
        model = DEFAULT_MODEL

    # start a streaming chat
    stream = ollama_client.chat(
        model=model,
        messages=[{'role': 'user', 'content': 'Hi'}],
        stream=True
    )

    try:
        first = next(stream) # Fails if no stream
        stream.close()
    except StopIteration:
        pass

    return "model loaded"


def get_model_capabilities(model: str = DEFAULT_MODEL) -> list:
    """Get the capabilities of a model.

    Args:
        model (str, optional): The model to get the capabilities of. Defaults to DEFAULT_MODEL

    Returns:
        list: List of capabilities.
    """
    if model is None:
        model = DEFAULT_MODEL

    info = ollama_client._request_raw("POST", "/api/show",json={"name":model}).json()
    capabilities = info["capabilities"]
    return capabilities


def summarize_chat(num: int = 10, model: str = DEFAULT_MODEL) -> None:
    """Summarize the first `num` messages (after system prompt) and replace them with a summary.

    Args:
        num (int, optional): Number of messages to summarize. Defaults to 10.
        model (str, optional): The model to use for summarization. Defaults to DEFAULT_MODEL.
    
    Returns: None
    """
    if len(context) <= num + 1:
        print("Not enough messages to summarize.")
        return

    system_msg = context[0]

    to_summarize = context[1:num+1]
    keep_rest = context[num+1:]

    summarization_prompt = (
        "Summarize the following conversation in a concise but clear way."
        "Keep important details, but remove fluff. Make it short enough to fit in one message.\n\n"
        f"{json.dumps(to_summarize, ensure_ascii=False, indent=2)}"
    )

    response = ollama_client.chat(
        model=model,
        messages=[
            {"role": "system", "content": "You are a summarizer. Do not tell what you're about to do, summarize only."},
            {"role": "user", "content": summarization_prompt}
        ],
        think=False
    )

    summary_text = response["message"]["content"].strip()

    # Build new summarized context
    summarized_msg = {
        "role": "system",
        "content": f"(Summary of earlier conversation)\n{summary_text}"
    }

    new_context = [system_msg, summarized_msg] + keep_rest

    context.clear()
    context.extend(new_context)

    with open("context.json", "w", encoding="utf-8") as f:
        json.dump(context, f, ensure_ascii=False, indent=2)


def get_tool_call(tool_call) -> str:
    """Run the proper tool called and output its result.

    Args:
        tool_call (dict): The tool call structure from the model.

    Returns:
        str: The result of the tool call.
    """
    tool_name = tool_call['function'].get('name')

    if tool_name == 'browse':
        query = tool_call['function']['arguments'].get('query')
        result = browse(str(query))
        return(str(result))

    elif tool_name == 'gif':
        query = tool_call['function']['arguments'].get('query')
        result = gif(str(query))
        return(str(result))

    elif tool_name == "python":
        script = tool_call['function']['arguments'].get('script')
        result = scripting.run_script(script)
        return(str(result))
    
    elif tool_name == "memorize":
        information = tool_call['function']['arguments'].get('information')
        info_user = tool_call['function']['arguments'].get('user')
        
        write_memory(info_user, information)

        result = f"Information about {info_user} saved: {information}"
        return(str(result))

    else:
        error_msg = f"Unknown tool type: {tool_name}"
        return(error_msg, 'tool')


def save_context(content, role='user', image_path: list = None, custom_field: str = None) -> None:
    """
    Save a message to the context and write it to context.json.

    Args:
        content (str): The message content.
        role (str): The role of the message ('user', 'assistant', 'system', 'tool'). Defaults to 'user'.
        image_path (list, optional): List of image paths associated with the message. Defaults to None.
        custom_field (str, optional): Extra field in format "field, value". Defaults to None.

    Raises:
        RuntimeError: If unable to save context.

    Returns: None
    """
    entry = {
        'role': role,
        'content': content
    }

    if image_path:
        entry['images'] = image_path

    if custom_field:
        try:
            field, value = [x.strip() for x in custom_field.split(",", 1)]
            entry[field] = value
        except ValueError:
            raise ValueError("custom_field must be in format 'field, value'")

    context.append(entry)

    try:
        data_str = json.dumps(context, ensure_ascii=False, indent=2)
        json.loads(data_str)

        path = "context.json"
        with open(path, "w", encoding="utf-8") as f:
            f.write(data_str)

    except Exception as e:
        raise RuntimeError(f"Couldn't save context: {e}")


def chat(content: str, role: str = 'user', model: str = DEFAULT_MODEL, thinking: str = 'auto', num_retry_fail: int = 5, custom_field: str = None, custom_tools: str = None) -> str:
    """Generate a reply from the LLM with optional multimodal tool calling.

    Args:
        content (str): The prompt given to the model.
        role (str, optional): The role to label the message with. 
            Options: 'user', 'assistant', 'system'. Defaults to 'user'.
        model (str, optional): The model to use for generation. Defaults to DEFAULT_MODEL.
        thinking (str, optional): Whether to enable tool calling.
            Options: 'auto', 'true', 'false'. Defaults to 'auto'.
        num_retry_fail (int, optional): Number of retries on failure. Defaults to 5.
        custom_field (str, optional): Extra field in format "field, value". Defaults to None.
        custom_tools (str, optional): Custom tool_call structure in JSON format. Defaults to None.

    Returns:
        str: The generated reply from the model.
    """
    if num_retry_fail >= 0:
        try:
            if model is None:
                model = DEFAULT_MODEL

            generate = True
            tool_calling = False
            final_output = ""

            if custom_field:
                save_context(content, role=role, custom_field=custom_field)
            else:
                save_context(content, role=role)

            while generate:
                if thinking.lower() == 'auto':
                    pass
                elif thinking.lower() == 'true':
                    tool_calling = True
                elif thinking.lower() == 'false':
                    tool_calling = False

                if custom_tools:
                    response = ollama_client.chat(
                        model=model,
                        messages=context,
                        tools=custom_tools,
                        think=tool_calling,
                        stream=False,
                    )
                else:
                    response = ollama_client.chat(
                        model=model,
                        messages=context,
                        tools=tools,
                        think=tool_calling,
                        stream=False,
                    )

                logs.append(response.model_dump(mode='json'))

                with open("logs.json", "w", encoding="utf-8") as f:
                    json.dump(logs, f, ensure_ascii=False, indent=2)

                if response['message'].get('content'):
                    final_output = response['message']['content']
                    save_context(final_output, 'assistant')
                    generate = False
                    tool_calling = False

                elif 'tool_calls' in response['message']:
                    if custom_tools:
                        return(response['message']['tool_calls'])
                    else:
                        for tool_call in response['message']['tool_calls']:
                            result = get_tool_call(tool_call)

                            save_context(result, 'tool')

                    generate = True
                    tool_calling = True

            return final_output

        except ResponseError as e:
            if e.status_code == "524":
                num_retry_fail -= 1
    else:
        return "couldn't generate the message. Please retry later."