import os
import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class ZigHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith(".zig"):
            print(f"File {event.src_path} has been modified")
            try:
                subprocess.run(["zig", "build", "run"], check=True)
                print("Build successful")
            except subprocess.CalledProcessError:
                print("Build failed")


def watch_zig_files(path="./src"):
    event_handler = ZigHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    print("Watching for changes to .zig files")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    watch_zig_files()
