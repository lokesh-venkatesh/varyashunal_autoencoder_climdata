import subprocess
import os

os.makedirs("results", exist_ok=True)

def run_script(script_name):
    try:
        print(f"Running {script_name}...")
        subprocess.run(["python", script_name], check=True)
        print(f"{script_name} completed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script_name}: {e}")
        exit(1)

if __name__ == "__main__":
    scripts = ["train.py", "generate.py", "plots.py"]
    for script in scripts:
        run_script(script)