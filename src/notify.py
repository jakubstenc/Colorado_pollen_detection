#!/usr/bin/env python3
import os
import smtplib
import argparse
import csv
from email.message import EmailMessage

def parse_results(file_path):
    """Parse YOLOv8 results.csv to format the final metrics."""
    if not os.path.exists(file_path):
        return f"File {file_path} not found. Could not parse metrics."
        
    csv_path = file_path
    if file_path.endswith('.pt'):
        # Resolve symlink (e.g., latest.pt -> run_dir/weights/best.pt)
        real_pt = os.path.realpath(file_path)
        run_dir = os.path.dirname(os.path.dirname(real_pt))
        csv_path = os.path.join(run_dir, "results.csv")
    
    if not os.path.exists(csv_path):
        return f"Model {csv_path} not found. Could not parse metrics."
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = list(csv.DictReader(f))
            if not reader:
                return "Model results.csv is empty."
            
            # YOLO results usually have spaces in header names, strip them
            last_epoch = reader[-1]
            last_epoch = {k.strip(): v.strip() for k, v in last_epoch.items()}
            
            epoch = last_epoch.get('epoch', 'N/A')
            precision = last_epoch.get('metrics/precision(B)', 'N/A')
            recall = last_epoch.get('metrics/recall(B)', 'N/A')
            map50 = last_epoch.get('metrics/mAP50(B)', 'N/A')
            map50_95 = last_epoch.get('metrics/mAP50-95(B)', 'N/A')
            
            summary = (
                f"🎉 Final Epoch: {epoch}\n"
                f"🎯 Precision: {precision}\n"
                f"🔎 Recall: {recall}\n"
                f"📏 mAP50: {map50}\n"
                f"📏 mAP50-95: {map50_95}\n"
            )
            return summary
    except Exception as e:
        return f"Error parsing results.csv: {e}"

def send_email(subject, body):
    sender_email = "jakubstenc@gmail.com"
    receiver_email = "jakubstenc@gmail.com"
    password = os.environ.get("GMAIL_APP_PASSWORD")

    if not password:
        print("⚠️ GMAIL_APP_PASSWORD not set. Cannot send email.")
        return

    msg = EmailMessage()
    msg.set_content(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email

    try:
        # Gmail SMTP configuration
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(sender_email, password)
        server.send_message(msg)
        server.quit()
        print("✅ Email notification sent successfully to", receiver_email)
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=['training', 'detection'], required=True, help="Job type")
    parser.add_argument("--job_name", default="Pollen Detection Job", help="Name of the job")
    parser.add_argument("--results", default=None, help="Path to results.csv (for training)")
    args = parser.parse_args()

    if args.type == "training":
        subject = f"✅ Training Completed: {args.job_name}"
        body = f"Hello Jakub,\n\nYour YOLOv8 training job '{args.job_name}' has successfully completed its run on the Kubernetes cluster.\n\n"
        if args.results:
            body += "📊 Training Metrics Summary:\n"
            body += parse_results(args.results)
            body += "\n\nThe weights and metric graphs have been uploaded to your S3 bucket.\n\nBest,\nYour Automated Cluster pipeline"
        else:
            body += "The weights and resulting graphs have been synced to your S3 bucket.\n\nBest,\nYour Automated Cluster pipeline"
    
    elif args.type == "detection":
        subject = f"✅ Detection Completed: {args.job_name}"
        body = (
            f"Hello Jakub,\n\n"
            f"Your pollen detection inference job '{args.job_name}' has successfully completed.\n"
            f"The extracted pseudo-labels or CSV results have been uploaded to your S3 bucket.\n\n"
            f"Best,\nYour Automated Cluster pipeline"
        )
    
    send_email(subject, body)

if __name__ == "__main__":
    main()
