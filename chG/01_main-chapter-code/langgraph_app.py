# langgraph_app.py
import chainlit as cl
from typing import TypedDict, List
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
import torch, asyncio

from reasoning_from_scratch.ch02 import get_device
from reasoning_from_scratch.ch03 import load_model_and_tokenizer, generate_text_basic_stream_cache

# Device setup
DEVICE = get_device()

# Load Qwen3 reasoning model
MODEL, TOKENIZER = load_model_and_tokenizer(
    which_model="reasoning",
    device=DEVICE,
    use_compile=False,
    local_dir="qwen3"
)

# Define agent state
class AgentState(TypedDict):
    messages: List[BaseMessage]

# Streaming function
async def stream_llm_response(messages: List[BaseMessage]) -> AIMessage:
    msg = cl.Message(content="")
    await msg.send()

    # Build prompt from messages
    prompt = "\n".join([m.content for m in messages])
    input_ids = TOKENIZER.encode(prompt)
    input_ids_tensor = torch.tensor(input_ids, device=DEVICE).unsqueeze(0)

    content = ""
    try:
        for tok in generate_text_basic_stream_cache(
            model=MODEL,
            token_ids=input_ids_tensor,
            max_new_tokens=512,
        ):
            token_id = tok.squeeze(0)
            piece = TOKENIZER.decode(token_id.tolist())
            content += piece
            await msg.stream_token(piece)
    except asyncio.CancelledError:
        print("\n⚠️ Streaming was interrupted.")
        pass

    await msg.update()
    return AIMessage(content=content)

# LangGraph node
async def llm_node(state: AgentState) -> AgentState:
    response = await stream_llm_response(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    print("\nFINAL STATE: ", state)
    return state

# Build LangGraph
graph = StateGraph(AgentState)
graph.add_node("llm", llm_node)
graph.set_entry_point("llm")
graph.set_finish_point("llm")
agent = graph.compile()
