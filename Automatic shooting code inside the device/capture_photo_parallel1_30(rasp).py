import os
import time
from PIL import Image
import subprocess
import threading
from datetime import datetime
# Import merge processing function
# We'll replace the process_reactors import with our new script

folder_cam1 = "/home/huang/pictures/camera1"
os.makedirs(folder_cam1, exist_ok=True)

#Raspberry Pi lens specifications
RESOLUTION = "1920x1080"
QUALITY = "100"
EXPOSURE = "auto"
GAIN = "2.0"
SHUTTER_SPEED = "10000"
FORMAT = "png"

stop_event = threading.Event()


def create_batch_folder():
    """Create batch folder /home/huang/pictures/camera1/YYYYMMDD/batch_xxx"""
    today = datetime.now().strftime("%Y%m%d")
    today_folder = os.path.join(folder_cam1, today)
    os.makedirs(today_folder, exist_ok=True)

    batch_id = len([d for d in os.listdir(today_folder) if d.startswith("batch_")]) + 1
    batch_folder = os.path.join(today_folder, f"batch_{batch_id:03d}")
    os.makedirs(batch_folder, exist_ok=True)

    print(f"\n? Start Batch {batch_id:03d}: {batch_folder}")
    return batch_folder


def take_one_picture(camera_id, batch_folder):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{camera_id}_{timestamp}.{FORMAT}"
    file_path = os.path.join(batch_folder, filename)

    cmd = [
        "/usr/bin/rpicam-still",
        "-o", file_path,
        "--camera", str(camera_id),
        "--width", "1642",
        "--height", "1232",
        "--nopreview" #Output image parameters
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"? Camera {camera_id} failed: {result.stderr.decode()}")
        return False

    try:
        img = Image.open(file_path)
        img = img.rotate(180, expand=True)
        img.save(file_path)
        print(f"? Saved: {file_path}")
    except Exception as e:
        print(f"?? Rotation failed: {e}")

    return True

#Shooting duration and intervals
def run_one_batch(duration=900, interval=10):
    """Capture one batch and automatically merge"""
    batch_folder = create_batch_folder()
    start_time = time.time()

    print(f"? Capture duration: 15 minutes | Interval: {interval} seconds\n")

    while time.time() - start_time < duration and not stop_event.is_set():
        interval_start = time.time()  # Start time of this interval

        # Take one picture
        take_one_picture(camera_id=0, batch_folder=batch_folder)

        # Time compensation to maintain consistent interval
        elapsed = time.time() - interval_start
        remaining = interval - elapsed
        if remaining > 0:
            time.sleep(remaining)

        total_elapsed = time.time() - start_time
        print(f"Elapsed: {int(total_elapsed)} sec, Remaining: {int(duration - total_elapsed)} sec")

    print("\n? Capture finished, start merging and processing...")
    # Call our new image processing script
    import sys
    sys.path.append('/home/pi/sludge_prediction_project/src')
    from process_batch import process_reactors
    process_reactors(batch_folder)  # Automatic processing
    print("? Batch processing complete ?\n")


def start_capture():
    t = threading.Thread(target=run_one_batch, daemon=True)
    t.start()
    return t


def stop_capture():
    stop_event.set()


if __name__ == "__main__":
    try:
        print("? Start capture task with automatic batch merge...")
        run_one_batch()  # Default 15 minutes
    except KeyboardInterrupt:
        print("\n? User interrupted capture")
        stop_capture()
        time.sleep(1)
        print("? Capture stopped")
