import subprocess, time

class LlamaServer:
    def __init__(self, cmd):
        self.cmd = cmd
        self.proc = None

    def start(self):
        if self.proc is None or self.proc.poll() is not None:
            self.proc = subprocess.Popen(self.cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait()

    def restart(self):
        self.stop()
        time.sleep(3)
        self.start()
        time.sleep(20)