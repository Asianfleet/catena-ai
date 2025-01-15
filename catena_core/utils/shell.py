from typing import List

def run_shell_command(args: List[str], tail: int = 100) -> str:
    print(f"Running shell command: {args}")

    import subprocess

    try:
        result = subprocess.run(args, capture_output=True, text=True)
        print(f"Result: {result}")
        print(f"Return code: {result.returncode}")
        if result.returncode != 0:
            return f"Error: {result.stderr}"

        # return only the last n lines of the output
        return "\n".join(result.stdout.split("\n")[-tail:])
    except Exception as e:
        print(f"Failed to run shell command: {e}")
        return f"Error: {e}"