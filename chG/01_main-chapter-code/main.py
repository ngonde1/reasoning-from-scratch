from fastapi import FastAPI
from chainlit.utils import mount_chainlit

app = FastAPI()

@app.get("/app")
def read_main():
    return {"message": "Hello World from FastAPI"}

# Mount your Qwen3 Chainlit app at /qwen3
mount_chainlit(app=app, target="qwen3_app.py", path="/qwen3")
