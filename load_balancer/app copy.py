from starlette.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, Form
from contextlib import asynccontextmanager
import subprocess as sp
import os


import requests


def send_request_to_container(port, file):

    url = f"http://localhost:{port}/process"

    payload = {}
    files = [("input_frame", (file.filename, file.file, file.content_type))]
    headers = {}

    response = requests.request(
        "POST", url, headers=headers, data=payload, files=files, timeout=30
    )

    return response.json()


# Approximate GPU VRAM usage per process: 2GB
def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = (
        sp.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
    )
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


def get_possible_number_of_processes():
    memory_free_values = get_gpu_memory()[0]
    memory_usage_per_process = 2048

    return memory_free_values // memory_usage_per_process


async def lifespan(app: FastAPI):
    # Load the ML model
    n_processes = get_possible_number_of_processes()

    print(f"Total free memory: {get_gpu_memory()}")
    print(f"Number of processes: {n_processes}")

    sp.Popen(f"gunicorn -b 0.0.0.0:8000 -w {n_processes} app:app".split())

    yield
    # Clean kill the process running at port 8000

    sp.Popen(f"kill -9 $(lsof -t -i:8000)".split())


n_processes = get_possible_number_of_processes()
app = FastAPI(lifespan=lifespan)

## app.include_router(get_info.router)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/process")
def process_image(file: UploadFile = File(...)):
    target_port = 8000
    return send_request_to_container(target_port, file)
