import mlflow
import mlflow.pytorch
from typing import Dict, Any, Optional
import torch
import logging
from datetime import datetime
import os

import mlflow
import os
import subprocess
import time
from pyngrok import ngrok
import signal
import sys

class MLflowUIServer:
    def __init__(self):
        ngrok_token="2rXqcyPnmP4dGvJxxvcatkcS8i0_7C1EFhK7TUEJmzYMUYRDt"
        self.mlflow_process = None
        self.ngrok_url = None
        self.is_running = False
        self.ngrok_token = ngrok_token

    def start(self):
        """Start MLflow UI server with ngrok tunnel"""
        try:
            # Set up ngrok authentication
            if self.ngrok_token:
                ngrok.set_auth_token(self.ngrok_token)
            # Set local directory for mlflow
            os.makedirs("fedprox/mlruns", exist_ok=True)
            mlflow.set_tracking_uri("file://" + os.path.abspath("fedprox/mlruns"))
            
            # Start MLflow UI server
            self.mlflow_process = subprocess.Popen(
                ['mlflow', 'ui', '--backend-store-uri', 'fedprox/mlruns', '--port', '5000'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for server to start
            time.sleep(5)
            
            # Start ngrok tunnel
            self.ngrok_url = ngrok.connect(5000)
            self.is_running = True
            
            print(f"\nMLflow UI is available at: {self.ngrok_url.public_url}")
            print("This URL will remain active until you call stop_server()")
            
            # Set up signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self.signal_handler)
            signal.signal(signal.SIGTERM, self.signal_handler)
            
            return mlflow
            
        except Exception as e:
            print(f"Error starting server: {e}")
            self.stop()
            raise

    def stop(self):
        """Stop MLflow UI server and cleanup"""
        if self.is_running:
            try:
                # Kill MLflow server
                if self.mlflow_process:
                    self.mlflow_process.terminate()
                    self.mlflow_process.wait()
                
                # Disconnect ngrok
                if self.ngrok_url:
                    ngrok.disconnect(self.ngrok_url.public_url)
                
                ngrok.kill()
                self.is_running = False
                print("\nMLflow UI server stopped")
                
            except Exception as e:
                print(f"Error stopping server: {e}")

    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\nShutdown signal received. Stopping server...")
        self.stop()
        sys.exit(0)

    def keep_alive(self):
        """Keep the server running until manually stopped"""
        try:
            print("\nServer is running. Press Ctrl+C to stop...")
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

def setup_tracking():
    """Initialize MLflow tracking with persistent UI"""
    server = MLflowUIServer()
    mlflow = server.start()
    
    experiment_name = "GPAF_Medical_FL1"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    
    return mlflow, experiment_id, server
'''
# Example usage
if __name__ == "__main__":
    mlflow, experiment_id, server = setup_tracking()
    
    try:
        with mlflow.start_run(experiment_id=experiment_id):
            # Your training code here
            print("Training in progress...")
            time.sleep(10)  # Simulate training
            print("Training complete!")
            
        # Keep the server running after training
        print("\nTraining finished but UI server still running!")
        server.keep_alive()
        
    finally:
        server.stop()
'''