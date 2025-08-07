import os

def read_last_session(log_path):
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            return f.read().strip()
    return None

def save_session_timestamp(log_path, timestamp):
    with open(log_path, "w") as f:
        f.write(timestamp)