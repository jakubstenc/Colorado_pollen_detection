#!/usr/bin/env python3
import os
import time
import smtplib
from email.message import EmailMessage
from pathlib import Path

# Load env
ENV_PATH = "/home/meow/Documents/Antigravity/Colorado_pollen_detection/.env"
def load_env():
    if not os.path.exists(ENV_PATH): return {}
    env = {}
    with open(ENV_PATH) as f:
        for line in f:
            if '=' in line:
                k, v = line.strip().split('=', 1)
                env[k] = v
    return env

config = load_env()
GMAIL_APP_PASSWORD = config.get('GMAIL_APP_PASSWORD')
RECEIVER_EMAIL = "jakubstenc@gmail.com"

def send_notification(subject, body):
    if not GMAIL_APP_PASSWORD:
        print("⚠️ GMAIL_APP_PASSWORD not set.")
        return
    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = RECEIVER_EMAIL
    msg["To"] = RECEIVER_EMAIL
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(RECEIVER_EMAIL, GMAIL_APP_PASSWORD)
            server.send_message(msg)
        print("✅ Notification sent!")
    except Exception as e:
        print(f"❌ Failed to send: {e}")

def monitor(target_path):
    print(f"👀 Monitoring {target_path} for completion...")
    while not os.path.exists(target_path):
        time.sleep(60)
    
    # Wait for file to be closed (primitive check)
    size = -1
    while True:
        new_size = os.path.getsize(target_path)
        if new_size == size and new_size > 0:
            break
        size = new_size
        time.sleep(10)

    send_notification(
        "🚀 Ingestion Complete: Colorado Pollen",
        f"The unified multi-species dataset is ready in {os.path.dirname(target_path)}.\nYou can now proceed with pseudo-labeling or training."
    )

if __name__ == "__main__":
    monitor("./dataset_colorado/tile_manifest.json")
