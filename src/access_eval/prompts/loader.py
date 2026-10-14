from pathlib import Path

PROMPTS_ROOT = Path(__file__).parent

def load_prompt(path_template: str) -> str:
    path_to_read = PROMPTS_ROOT / f"{path_template}.txt"
    with open(path_to_read, "r") as f:
        raw_template = f.read()
    return raw_template.strip()