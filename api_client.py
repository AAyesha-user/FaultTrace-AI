"""
FaultTrace AI - API client
--------------------------
Put this file in the PROJECT ROOT (next to app.py and api.py).
 
It lets the Streamlit dashboard talk to the FastAPI server over HTTP.
Every function returns plain Python data, or raises ApiError with a
message that is safe to show to the user.
"""
 
import requests
 
 
class ApiError(Exception):
    pass
 
 
def _request(method, base_url, path, timeout=60, **kwargs):
    url = base_url.rstrip("/") + path
 
    try:
        response = requests.request(method, url, timeout=timeout, **kwargs)
    except requests.exceptions.ConnectionError:
        raise ApiError(
            f"Cannot reach the API at {base_url}. "
            "Is the server running? (uvicorn api:app --reload)"
        )
    except requests.exceptions.Timeout:
        raise ApiError(
            "The API took too long to answer. A sleeping free server can need "
            "about a minute to wake up. Wait a moment and try again."
        )
    except requests.exceptions.RequestException as error:
        raise ApiError(f"Request failed: {error}")
 
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        raise ApiError(f"HTTP {response.status_code}: {detail}")
 
    try:
        return response.json()
    except ValueError:
        raise ApiError("The API did not return valid JSON.")
 
 
def get_status(base_url):
    return _request("GET", base_url, "/status", timeout=60)
 
 
def run_simulation(base_url, payload):
    return _request("POST", base_url, "/simulate", json=payload, timeout=120)
 
 
def get_diagnosis(base_url):
    return _request("GET", base_url, "/diagnose")
 
 
def run_recovery(base_url, suspects):
    return _request("POST", base_url, "/recover", json={"suspects": suspects})
 
 
def get_history(base_url, limit=100, robot_id=None):
    params = {"limit": limit}
    if robot_id:
        params["robot_id"] = robot_id
    return _request("GET", base_url, "/history", params=params)
 