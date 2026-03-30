import argparse
import json
import jsonlines
import re
import time
import os
from functools import lru_cache
from pathlib import Path

from openai import OpenAI

from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import tomllib
except ImportError:
    import toml as tomllib

def extract_code_blocks(text: str) -> str:
    if text is None:
        return ""
    pattern = re.compile(r"```(?:python|py|python3)[^\n]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
    matches = [m.group(1).strip() for m in pattern.finditer(text)]
    if not matches:
        return ""
    longest = max(matches, key=len)
    return longest

def build_code_opt_prompt(original_code: str) -> str:
    """Build an optimization prompt using a simple on-disk template with safe token replacement.

    The template must contain tokens {{LANGUAGE_NAME}}, {{SOURCE_FILENAME}}, {{BASELINE_SECONDS}},
    and markers :::HEADER::: and :::CODE::: which will be replaced verbatim.
    Falls back to an inline template if the file is missing.
    """

    return (
        f"You are an expert python performance engineer.\n\n"
        f"Python version: 3.10\n"
        f"Task: Optimize the provided `solve` function.\n"
        f"- Do not change the function signature.\n"
        f"- Provide a full replacement for this code.\n"
        f"- Return only the code in a single fenced block.\n\n"
        f"--- current source ---\n{original_code}\n"
    )


@lru_cache(maxsize=1)
def get_dynamic_package_list(pyproject_path: str = "pyproject.toml") -> str:
    """Return a comma-separated package list derived from pyproject.toml."""
    if not os.path.exists(pyproject_path):
        return ""

    file_mode = "rb" if tomllib.__name__ == "tomllib" else "r"
    with open(pyproject_path, file_mode) as f:
        project = tomllib.load(f)

    deps = project.get("project", {}).get("dependencies")
    if deps is None:
        deps = project.get("tool", {}).get("poetry", {}).get("dependencies", [])

    if isinstance(deps, dict):
        dep_iterable = deps.keys()
    elif isinstance(deps, list):
        dep_iterable = deps
    else:
        dep_iterable = []

    exclude = {
        "openai",
        "huggingface-hub",
        "litellm",
        "google-generativeai",
        "pandas",
        "pylint",
        "line_profiler",
        "pytest",
        "toml",
        "python",
        "orjson",
        "pyaml",
        "pillow",
    }

    package_names = []
    for dep in dep_iterable:
        name = (
            dep.split("[")[0]
            .split(" ")[0]
            .split("=")[0]
            .split(">")[0]
            .split("<")[0]
            .strip()
            .strip('"')
            .strip("'")
        )
        if name and name not in exclude:
            package_names.append(name)

    return ", ".join(sorted(dict.fromkeys(package_names)))


def build_code_opt_prompt_with_output_instructions(original_code: str) -> str:
    """Build the original optimization prompt plus stronger output instructions."""
    package_list = get_dynamic_package_list()
    package_line = (
        f"Available packages: {package_list}\n\n"
        if package_list
        else ""
    )

    return (
        f"You are an expert python performance engineer.\n\n"
        f"Environment: Python version: 3.10\n"
        f"{package_line}"
        f"Task: Optimize the provided `solve` function to make it run as fast as possible.\n"
        f"IMPORTANT: Compilation time of your init function will not count towards your function's runtime.\n"
        f"Return exactly one Python code block containing the full contents of `solver.py`.\n"
        f"Your code must define a class named `Solver` with a callable `solve` method.\n"
        f"--- source code ---\n{original_code}\n"
    )

def llm_generate(client, model: str, prompt: str) -> str:
    last_error = None
    for attempt in range(1, 11):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                reasoning_effort="low",
                # temperature=0.6,
                # top_p=0.95,
                temperature=1.0,
                top_p=1.0,
            )

            text = completion.choices[0].message.content
            return text
        except Exception as exc:
            last_error = exc
            if attempt == 10:
                break
            sleep_seconds = min(2 ** (attempt - 1), 30)
            print(
                f"OpenAI request failed on attempt {attempt}/10: {exc}. "
                f"Retrying in {sleep_seconds}s..."
            )
            time.sleep(sleep_seconds)

    raise last_error

def generate_solutions(problems: dict, model: str):
    generations = {}
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="ryan123",
    )

    num_none = 0

    start = time.time()

    def process_problem(problem_name: str, baseline_code: str) -> tuple[str, str]:
        prompt = build_code_opt_prompt_with_output_instructions(baseline_code)
        response = llm_generate(client, model, prompt)
        code = extract_code_blocks(response)
        return problem_name, code

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process_problem, problem_name, baseline_code) for (problem_name, baseline_code) in problems.items()]
        for future in as_completed(futures):
            problem_name, solution_code = future.result()
            generations[problem_name] = solution_code
            if not solution_code:
                num_none += 1

    end = time.time()

    print(f"num none: {num_none}, time: {end - start}")
    return generations

def main():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate solutions to Effibench-X")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="output file for generations"
    )
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="output file for generations"
    )

    args = parser.parse_args()
    baseline_codes = {}

    for benchmark in os.listdir("baseline_codes"):
        with open(Path("baseline_codes") / benchmark / "solver.py") as f:
            baseline_code = f.read()
            baseline_codes[benchmark] = baseline_code
    
    generations = generate_solutions(baseline_codes, args.model)

    for benchmark, generation in generations.items():
        (Path(args.output) / benchmark).mkdir(parents=True, exist_ok=True)

        with open(Path(args.output) / benchmark / "solver.py", "w") as f:
            f.write(generation)

if __name__ == "__main__":
    main()
