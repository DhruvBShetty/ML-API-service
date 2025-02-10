import subprocess
import sys
from pathlib import Path
import re

current_dir = Path(__file__).resolve()
parent_dir = current_dir.parent
python_files = [str(file) for file in parent_dir.glob("*.py")]

# Extract the score from the pylint output
if python_files:
    result = subprocess.run(
        ["pylint", "--output-format=text"] + python_files,
        capture_output=True,
        text=True,
    )
    # Extract the score from the pylint output
    score_match = re.search(
        r"Your code has been rated at (10(\.0{1,2})?|[0-9](\.\d{1,2})?)/10",
        result.stdout,
    )

    if score_match:
        score = float(score_match.group(1))
        # Enforce the score threshold
        if score < 8:
            print(f"Pylint score is {score}, which is below 8. Failing the process!")
            sys.exit(1)  # Exit with non-zero status to fail the process
        else:
            print(f"Pylint score is {score}. Pass!")
