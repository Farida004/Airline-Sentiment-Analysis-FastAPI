from fastapi import FastAPI
from router import gets

app = FastAPI()
app.include_router(gets.router)
