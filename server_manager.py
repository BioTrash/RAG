import subprocess
import time

class LlamaServer:
    def __init__(self, name="llama_server_1", port=8082, image="llama-server:local"):
        self.name = name
        self.port = port
        self.image = image

    def start(self):
        # If container already exists and is running, do nothing
        res = subprocess.run(["docker", "ps", "-q", "-f", f"name={self.name}"], capture_output=True, text=True)
        if res.stdout.strip():
            return

        # If stopped container exists, remove it first
        res = subprocess.run(["docker", "ps", "-aq", "-f", f"name={self.name}"], capture_output=True, text=True)
        if res.stdout.strip():
            subprocess.run(["docker", "rm", "-f", self.name], check=False)

        cmd = [
            "docker", "run", "-d",
            "--name", self.name,
            "-p", f"127.0.0.1:{self.port}:{self.port}",
            "-v", "/home/rufus/llama.cpp:/llama.cpp:ro",
            "-v", "/opt/intel/oneapi:/opt/intel/oneapi:ro",
            "--device", "/dev/dri",
            "--group-add", "video",
            self.image
        ]
        subprocess.run(cmd, check=True)

        # Wait for server to become ready (simple sleep, replace with health-check if available)
        time.sleep(20)

    def stop(self):
        subprocess.run(["docker", "stop", self.name], check=False)
        subprocess.run(["docker", "rm", self.name], check=False)

    def restart(self):
        self.stop()
        time.sleep(2)
        self.start()