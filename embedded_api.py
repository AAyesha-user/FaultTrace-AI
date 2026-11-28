import socket
import threading
import time

import streamlit as st

API_HOST = "127.0.0.1"
API_PORT = 8000


def _port_is_open(host, port):
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


@st.cache_resource
def ensure_api_running(port: int = API_PORT):
    """Start the API once per server process. Safe to call on every rerun."""
    if _port_is_open(API_HOST, port):
        return "already-running"

    import uvicorn
    from api import app as fastapi_app

    config = uvicorn.Config(fastapi_app, host=API_HOST, port=port, log_level="warning")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()

    for _ in range(25):
        if _port_is_open(API_HOST, port):
            return "started"
        time.sleep(0.2)

    return "failed"