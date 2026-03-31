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

import subprocess
import argparse

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

def monitor_file(target_path):
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

def monitor_k8s_pod(pod_name, namespace):
    print(f"👀 Monitoring Kubernetes pod {pod_name} in namespace {namespace}...")
    while True:
        try:
            cmd = ["./kubectl", "get", "pod", pod_name, "-n", namespace, "-o", "jsonpath={.status.phase}"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            phase = result.stdout.strip()
            
            if phase == 'Succeeded':
                send_notification(
                    "🎉 Training Complete: General Pollen Model",
                    f"The Kubernetes training pod '{pod_name}' successfully finished its workload!\nYour YOLOv8 model weights and training statistics should now be safely synced to your S3 bucket."
                )
                print("Training finished.")
                break
            elif phase in ['Failed', 'Error']:
                send_notification(
                    "❌ Training Failed: General Pollen Model",
                    f"The Kubernetes training pod '{pod_name}' failed to complete.\nPlease check the logs by running:\n./kubectl logs {pod_name} -n {namespace}"
                )
                print("Training failed.")
                break
            elif phase == 'NotFound' or 'NotFound' in result.stderr:
                print(f"Pod {pod_name} not found. Ensure it exists.")
                time.sleep(60)
        except Exception as e:
            print(f"Error checking pod: {e}")
            
        time.sleep(60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor processes and send email completion notifications.")
    parser.add_argument("--file", type=str, help="Path to a file to monitor for ingestion completion.")
    parser.add_argument("--pod", type=str, help="Name of Kubernetes Pod to monitor.")
    parser.add_argument("--namespace", type=str, default="stenc-ns", help="Kubernetes namespace (for --pod).")
    
    args = parser.parse_args()
    
    if args.file:
        monitor_file(args.file)
    elif args.pod:
        monitor_k8s_pod(args.pod, args.namespace)
    else:
        print("No target specified. Defaulting to original tile manifest.")
        monitor_file("./dataset_colorado/tile_manifest.json")
