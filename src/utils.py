import os, sys
from datetime import datetime

def log_message(message: str, file_path: str) -> None:
    """
    Log a message with a timestamp.
    Args:
        message (str): Message to log.
    """
    if os.path.exists(file_path) == False:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(file_path, 'a') as f:
        f.write(f"[{now}] {message}\n")