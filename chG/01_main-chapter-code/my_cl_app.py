# my_cl_app.py
import chainlit as cl

@cl.on_chat_start
async def main():
    await cl.Message(content="Hello World from Chainlit!").send()
