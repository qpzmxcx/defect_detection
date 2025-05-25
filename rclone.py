import subprocess
import json

picture_path = "D:/User/Desktop/Graduation-Project1/output_test"
myvm_path = "myvm:/home/azureuser/picture"


def run_rclone(command):
    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding='utf-8',
            errors='replace',
            text=True
        )
        print("Output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error:", e.stderr)


run_rclone(["rclone", "copy", picture_path, myvm_path, "--progress"])
