"""
Utilities for code parsing, diffing, and manipulation
"""

import re
from typing import Dict, List, Optional, Tuple, Union


def parse_evolve_blocks(code: str) -> List[Tuple[int, int, str]]:
    """
    Parse evolve blocks from code

    Args:
        code: Source code with evolve blocks

    Returns:
        List of tuples (start_line, end_line, block_content)
    """
    lines = code.split("\n")
    blocks = []

    in_block = False
    start_line = -1
    block_content = []

    for i, line in enumerate(lines):
        if "# EVOLVE-BLOCK-START" in line:
            in_block = True
            start_line = i
            block_content = []
        elif "# EVOLVE-BLOCK-END" in line and in_block:
            in_block = False
            blocks.append((start_line, i, "\n".join(block_content)))
        elif in_block:
            block_content.append(line)

    return blocks


def get_code_lines(code: str) -> list[str]:
    """
    Split code into lines

    Args:
        code: Source code

    Returns:
        List of code lines
    """
    code_lines = code.split("\n")
    # Remove trailing spaces from each line
    code_lines = [line.rstrip() for line in code_lines]
    return code_lines

def remove_indentation(lines: list[str]) -> Tuple[List[str], int]:
    """
    Remove common indentation from code

    Args:
        code: Source code

    Returns:
        Code with common indentation removed
    """
    # Find minimum indentation
    indent_levels = [
        len(re.match(r"^( *)", line).group(1)) for line in lines if line.strip()
    ]
    if not indent_levels:
        return lines, 0
    min_indent = min(indent_levels)

    # Remove common indentation
    unindented_lines = [line[min_indent:] if len(line) >= min_indent else line for line in lines]
    return unindented_lines, min_indent

def add_indentation(lines: list[str], indent: int) -> list[str]:
    """
    Add indentation to code

    Args:
        code: Source code
        indent: Number of spaces to add

    Returns:
        Indented code
    """
    indented_lines = [(" " * indent) + line if line.strip() else line for line in lines]
    return indented_lines

def apply_diff(
    original_code: str,
    diff_text: str,
    diff_pattern: str = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE",
) -> str:
    """
    Apply a diff to the original code

    Args:
        original_code: Original source code
        diff_text: Diff in the SEARCH/REPLACE format
        diff_pattern: Regex pattern for the SEARCH/REPLACE format

    Returns:
        Modified code
    """
    # Split into lines for easier processing
    original_lines = get_code_lines(original_code)
    result_lines = original_lines.copy()

    # Extract diff blocks
    diff_blocks = extract_diffs(diff_text, diff_pattern)

    # Apply each diff block
    for search_text, replace_text in diff_blocks:
        search_lines = get_code_lines(search_text)
        replace_lines = get_code_lines(replace_text)
        unindented_search_lines, _ = remove_indentation(search_lines)
        unindented_replace_lines, _ = remove_indentation(replace_lines)

        # Find where the search pattern starts in the original code
        for i in range(len(result_lines) - len(search_lines) + 1):
            current_original_lines = result_lines[i : i + len(search_lines)].copy()
            unindented_original_lines, min_indent = remove_indentation(current_original_lines)
            if unindented_original_lines == unindented_search_lines:
                indented_replace_lines = add_indentation(unindented_replace_lines, min_indent)
                # Replace the matched section
                result_lines[i : i + len(search_lines)] = indented_replace_lines
                break

    return "\n".join(result_lines)


def extract_diffs(
    diff_text: str, diff_pattern: str = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE"
) -> List[Tuple[str, str]]:
    """
    Extract diff blocks from the diff text

    Args:
        diff_text: Diff in the SEARCH/REPLACE format
        diff_pattern: Regex pattern for the SEARCH/REPLACE format

    Returns:
        List of tuples (search_text, replace_text)
    """
    diff_blocks = re.findall(diff_pattern, diff_text, re.DOTALL)
    return [(match[0].rstrip(), match[1].rstrip()) for match in diff_blocks]


def parse_full_rewrite(llm_response: str, language: str = "python") -> Optional[str]:
    """
    Extract a full rewrite from an LLM response

    Args:
        llm_response: Response from the LLM
        language: Programming language

    Returns:
        Extracted code or None if not found
    """
    code_block_pattern = r"```" + language + r"\n(.*?)```"
    matches = re.findall(code_block_pattern, llm_response, re.DOTALL)

    if matches:
        return matches[0].strip()

    # Fallback to any code block
    code_block_pattern = r"```(.*?)```"
    matches = re.findall(code_block_pattern, llm_response, re.DOTALL)

    if matches:
        return matches[0].strip()

    # Fallback to plain text
    return llm_response


def format_diff_summary(diff_blocks: List[Tuple[str, str]]) -> str:
    """
    Create a human-readable summary of the diff

    Args:
        diff_blocks: List of (search_text, replace_text) tuples

    Returns:
        Summary string
    """
    summary = []

    for i, (search_text, replace_text) in enumerate(diff_blocks):
        search_lines = search_text.strip().split("\n")
        replace_lines = replace_text.strip().split("\n")

        # Create a short summary
        if len(search_lines) == 1 and len(replace_lines) == 1:
            summary.append(f"Change {i+1}: '{search_lines[0]}' to '{replace_lines[0]}'")
        else:
            search_summary = (
                f"{len(search_lines)} lines" if len(search_lines) > 1 else search_lines[0]
            )
            replace_summary = (
                f"{len(replace_lines)} lines" if len(replace_lines) > 1 else replace_lines[0]
            )
            summary.append(f"Change {i+1}: Replace {search_summary} with {replace_summary}")

    return "\n".join(summary)


def calculate_edit_distance(code1: str, code2: str) -> int:
    """
    Calculate the Levenshtein edit distance between two code snippets

    Args:
        code1: First code snippet
        code2: Second code snippet

    Returns:
        Edit distance (number of operations needed to transform code1 into code2)
    """
    if code1 == code2:
        return 0

    # Simple implementation of Levenshtein distance
    m, n = len(code1), len(code2)
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i

    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if code1[i - 1] == code2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )

    return dp[m][n]


def extract_code_language(code: str) -> str:
    """
    Try to determine the language of a code snippet

    Args:
        code: Code snippet

    Returns:
        Detected language or "unknown"
    """
    # Look for common language signatures
    if re.search(r"^(import|from|def|class)\s", code, re.MULTILINE):
        return "python"
    elif re.search(r"^(package|import java|public class)", code, re.MULTILINE):
        return "java"
    elif re.search(r"^(#include|int main|void main)", code, re.MULTILINE):
        return "cpp"
    elif re.search(r"^(function|var|let|const|console\.log)", code, re.MULTILINE):
        return "javascript"
    elif re.search(r"^(module|fn|let mut|impl)", code, re.MULTILINE):
        return "rust"
    elif re.search(r"^(SELECT|CREATE TABLE|INSERT INTO)", code, re.MULTILINE):
        return "sql"

    return "unknown"

def get_fix_diff_response_prompt(llm_response: str):
    return {
        "system": """You are an expert in fixing your collegues code. You know that the code should be in format:
<<<<<<< SEARCH
# Original code to find and replace (must match exactly, indentations and endlines as well)
=======
# New replacement code
>>>>>>> REPLACE
Provided the incorrect format of SEARCH/REPLACE fix it to a correct format.
        """,
        "user": f"An incorrect diff format was detected in this change:\n{llm_response}\nPlease fix it to the correct format."
    }