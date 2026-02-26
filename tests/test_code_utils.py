"""
Tests for code utilities in openevolve.utils.code_utils
"""

import unittest

from openevolve.utils.code_utils import (
    _format_block_lines,
    apply_diff,
    extract_diffs,
    format_diff_summary,
)


class TestCodeUtils(unittest.TestCase):
    """Tests for code utilities"""

    def test_extract_diffs(self):
        """Test extracting diffs from a response"""
        diff_text = """
        Let's improve this code:

        <<<<<<< SEARCH
        def hello():
            print("Hello")
        =======
        def hello():
            print("Hello, World!")
        >>>>>>> REPLACE

        Another change:

        <<<<<<< SEARCH
        x = 1
        =======
        x = 2
        >>>>>>> REPLACE
        """

        diffs = extract_diffs(diff_text)
        self.assertEqual(len(diffs), 2)
        self.assertEqual(
            diffs[0][0],
            """        def hello():
            print(\"Hello\")""",
        )
        self.assertEqual(
            diffs[0][1],
            """        def hello():
            print(\"Hello, World!\")""",
        )
        self.assertEqual(diffs[1][0], "        x = 1")
        self.assertEqual(diffs[1][1], "        x = 2")

    def test_apply_diff(self):
        """Test applying diffs to code"""
        original_code = """
        def hello():
            print("Hello")

        x = 1
        y = 2
        """

        diff_text = """
        <<<<<<< SEARCH
        def hello():
            print("Hello")
        =======
        def hello():
            print("Hello, World!")
        >>>>>>> REPLACE

        <<<<<<< SEARCH
        x = 1
        =======
        x = 2
        >>>>>>> REPLACE
        """

        expected_code = """
        def hello():
            print("Hello, World!")

        x = 2
        y = 2
"""

        result = apply_diff(original_code, diff_text)

        # Normalize whitespace for comparison
        self.assertEqual(
            result,
            expected_code,
        )

    def test_extract_diffs_real(self):
        """Test extracting diffs from a response"""
        diff_text = """
To improve the fitness score of the program, we can introduce a mechanism to escape local minima. One idea is to add a random perturbation to the best_x and best_y values during each iteration. This will help the algorithm to explore different regions of the search space and avoid getting stuck in local minima.

Here's the suggested improvement:

<<<<<<< SEARCH
    best_x = 0
    best_y = 0
    best_value = evaluate_function(best_x, best_y)

    for _ in range(iterations):
        x = best_x + 1
        y = best_y - 1
        value = evaluate_function(x, y)

        if value < best_value:
            best_value = value
            best_x, best_y = x, y
=======
    # Initialize with a random point
    best_x = np.random.uniform(bounds[0], bounds[1])
    best_y = np.random.uniform(bounds[0], bounds[1])
    best_value = evaluate_function(best_x, best_y)
    for _ in range(iterations):
        # Add a random perturbation to the best_x and best_y values
        perturbation_x = np.random.uniform(-1, 1)
        perturbation_y = np.random.uniform(-1, 1)
        x = best_x + perturbation_x
        y = best_y + perturbation_y
        # Ensure x and y are within the bounds
        x = max(bounds[0], min(x, bounds[1]))
        y = max(bounds[0], min(y, bounds[1]))
        value = evaluate_function(x, y)
        if value < best_value:
            best_value = value
            best_x, best_y = x, y
>>>>>>> REPLACE

In this improved version, a random perturbation is added to the best_x and best_y values during each iteration. The perturbation is uniformly distributed between -1 and 1. The x and y values are then updated to reflect the perturbation, and the evaluate_function is called with the new values. This will help the algorithm to explore different regions of the search space and avoid getting stuck in local minima.
        """

        search_text = """    best_x = 0
    best_y = 0
    best_value = evaluate_function(best_x, best_y)

    for _ in range(iterations):
        x = best_x + 1
        y = best_y - 1
        value = evaluate_function(x, y)

        if value < best_value:
            best_value = value
            best_x, best_y = x, y"""
        
        replace_text = """    # Initialize with a random point
    best_x = np.random.uniform(bounds[0], bounds[1])
    best_y = np.random.uniform(bounds[0], bounds[1])
    best_value = evaluate_function(best_x, best_y)
    for _ in range(iterations):
        # Add a random perturbation to the best_x and best_y values
        perturbation_x = np.random.uniform(-1, 1)
        perturbation_y = np.random.uniform(-1, 1)
        x = best_x + perturbation_x
        y = best_y + perturbation_y
        # Ensure x and y are within the bounds
        x = max(bounds[0], min(x, bounds[1]))
        y = max(bounds[0], min(y, bounds[1]))
        value = evaluate_function(x, y)
        if value < best_value:
            best_value = value
            best_x, best_y = x, y"""

        diffs = extract_diffs(diff_text)
        self.assertEqual(len(diffs), 1)
        self.assertEqual(
            diffs[0][0],
            search_text,
        )
        self.assertEqual(
            diffs[0][1],
            replace_text,
        )

    def test_apply_diff_real(self):
        """Test applying diffs to code"""
        diff_text = """
To improve the fitness score of the program, we can introduce a mechanism to escape local minima. One idea is to add a random perturbation to the best_x and best_y values during each iteration. This will help the algorithm to explore different regions of the search space and avoid getting stuck in local minima.

Here's the suggested improvement:

<<<<<<< SEARCH
    best_x = 0
    best_y = 0
    best_value = evaluate_function(best_x, best_y)
    
    for _ in range(iterations):
        x = best_x + 1
        y = best_y - 1
        value = evaluate_function(x, y)
        
        if value < best_value:
            best_value = value
            best_x, best_y = x, y
=======
    # Initialize with a random point
    best_x = np.random.uniform(bounds[0], bounds[1])
    best_y = np.random.uniform(bounds[0], bounds[1])
    best_value = evaluate_function(best_x, best_y)
    for _ in range(iterations):
        # Add a random perturbation to the best_x and best_y values
        perturbation_x = np.random.uniform(-1, 1)
        perturbation_y = np.random.uniform(-1, 1)
        x = best_x + perturbation_x
        y = best_y + perturbation_y
        # Ensure x and y are within the bounds
        x = max(bounds[0], min(x, bounds[1]))
        y = max(bounds[0], min(y, bounds[1]))
        value = evaluate_function(x, y)
        if value < best_value:
            best_value = value
            best_x, best_y = x, y
>>>>>>> REPLACE

In this improved version, a random perturbation is added to the best_x and best_y values during each iteration. The perturbation is uniformly distributed between -1 and 1. The x and y values are then updated to reflect the perturbation, and the evaluate_function is called with the new values. This will help the algorithm to explore different regions of the search space and avoid getting stuck in local minima.
        """

        original_code = """# EVOLVE-BLOCK-START
\"\"\"Function minimization example for OpenEvolve\"\"\"
import numpy as np


def search_algorithm(iterations=1000, bounds=(-5, 5)):
    \"\"\"
    A simple random search algorithm that often gets stuck in local minima.

    Args:
        iterations: Number of iterations to run
        bounds: Bounds for the search space (min, max)

    Returns:
        Tuple of (best_x, best_y, best_value)
    \"\"\"
    best_x = 0
    best_y = 0
    best_value = evaluate_function(best_x, best_y)

    for _ in range(iterations):
        x = best_x + 1
        y = best_y - 1
        value = evaluate_function(x, y)

        if value < best_value:
            best_value = value
            best_x, best_y = x, y

    return best_x, best_y, best_value


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def evaluate_function(x, y):
    \"\"\"The complex function we're trying to minimize\"\"\"
    return np.sin(x) * np.cos(y) + np.sin(x * y) + (x**2 + y**2) / 20


def run_search():
    x, y, value = search_algorithm()
    return x, y, value


if __name__ == "__main__":
    x, y, value = run_search()
    print(f"Found minimum at ({x}, {y}) with value {value}")"""

        expected_code = """# EVOLVE-BLOCK-START
\"\"\"Function minimization example for OpenEvolve\"\"\"
import numpy as np


def search_algorithm(iterations=1000, bounds=(-5, 5)):
    \"\"\"
    A simple random search algorithm that often gets stuck in local minima.

    Args:
        iterations: Number of iterations to run
        bounds: Bounds for the search space (min, max)

    Returns:
        Tuple of (best_x, best_y, best_value)
    \"\"\"
    # Initialize with a random point
    best_x = np.random.uniform(bounds[0], bounds[1])
    best_y = np.random.uniform(bounds[0], bounds[1])
    best_value = evaluate_function(best_x, best_y)
    for _ in range(iterations):
        # Add a random perturbation to the best_x and best_y values
        perturbation_x = np.random.uniform(-1, 1)
        perturbation_y = np.random.uniform(-1, 1)
        x = best_x + perturbation_x
        y = best_y + perturbation_y
        # Ensure x and y are within the bounds
        x = max(bounds[0], min(x, bounds[1]))
        y = max(bounds[0], min(y, bounds[1]))
        value = evaluate_function(x, y)
        if value < best_value:
            best_value = value
            best_x, best_y = x, y

    return best_x, best_y, best_value


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def evaluate_function(x, y):
    \"\"\"The complex function we're trying to minimize\"\"\"
    return np.sin(x) * np.cos(y) + np.sin(x * y) + (x**2 + y**2) / 20


def run_search():
    x, y, value = search_algorithm()
    return x, y, value


if __name__ == "__main__":
    x, y, value = run_search()
    print(f"Found minimum at ({x}, {y}) with value {value}")"""

        result = apply_diff(original_code, diff_text)

        # Normalize whitespace for comparison
        self.assertEqual(
            result,
            expected_code,
        )

    def test_apply_diff_real_no_indentation(self):
        """Test applying diffs to code"""
        diff_text = """
To improve the fitness score of the program, we can introduce a mechanism to escape local minima. One idea is to add a random perturbation to the best_x and best_y values during each iteration. This will help the algorithm to explore different regions of the search space and avoid getting stuck in local minima.

Here's the suggested improvement:

<<<<<<< SEARCH
best_x = 0
best_y = 0
best_value = evaluate_function(best_x, best_y)

for _ in range(iterations):
    x = best_x + 1
    y = best_y - 1
    value = evaluate_function(x, y)
    
    if value < best_value:
        best_value = value
        best_x, best_y = x, y
=======
# Initialize with a random point
best_x = np.random.uniform(bounds[0], bounds[1])
best_y = np.random.uniform(bounds[0], bounds[1])
best_value = evaluate_function(best_x, best_y)
for _ in range(iterations):
    # Add a random perturbation to the best_x and best_y values
    perturbation_x = np.random.uniform(-1, 1)
    perturbation_y = np.random.uniform(-1, 1)
    x = best_x + perturbation_x
    y = best_y + perturbation_y
    # Ensure x and y are within the bounds
    x = max(bounds[0], min(x, bounds[1]))
    y = max(bounds[0], min(y, bounds[1]))
    value = evaluate_function(x, y)
    if value < best_value:
        best_value = value
        best_x, best_y = x, y
>>>>>>> REPLACE

In this improved version, a random perturbation is added to the best_x and best_y values during each iteration. The perturbation is uniformly distributed between -1 and 1. The x and y values are then updated to reflect the perturbation, and the evaluate_function is called with the new values. This will help the algorithm to explore different regions of the search space and avoid getting stuck in local minima.
        """

        original_code = """# EVOLVE-BLOCK-START
\"\"\"Function minimization example for OpenEvolve\"\"\"
import numpy as np


def search_algorithm(iterations=1000, bounds=(-5, 5)):
    \"\"\"
    A simple random search algorithm that often gets stuck in local minima.

    Args:
        iterations: Number of iterations to run
        bounds: Bounds for the search space (min, max)

    Returns:
        Tuple of (best_x, best_y, best_value)
    \"\"\"
    best_x = 0
    best_y = 0
    best_value = evaluate_function(best_x, best_y)

    for _ in range(iterations):
        x = best_x + 1
        y = best_y - 1
        value = evaluate_function(x, y)

        if value < best_value:
            best_value = value
            best_x, best_y = x, y

    return best_x, best_y, best_value


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def evaluate_function(x, y):
    \"\"\"The complex function we're trying to minimize\"\"\"
    return np.sin(x) * np.cos(y) + np.sin(x * y) + (x**2 + y**2) / 20


def run_search():
    x, y, value = search_algorithm()
    return x, y, value


if __name__ == "__main__":
    x, y, value = run_search()
    print(f"Found minimum at ({x}, {y}) with value {value}")"""

        expected_code = """# EVOLVE-BLOCK-START
\"\"\"Function minimization example for OpenEvolve\"\"\"
import numpy as np


def search_algorithm(iterations=1000, bounds=(-5, 5)):
    \"\"\"
    A simple random search algorithm that often gets stuck in local minima.

    Args:
        iterations: Number of iterations to run
        bounds: Bounds for the search space (min, max)

    Returns:
        Tuple of (best_x, best_y, best_value)
    \"\"\"
    # Initialize with a random point
    best_x = np.random.uniform(bounds[0], bounds[1])
    best_y = np.random.uniform(bounds[0], bounds[1])
    best_value = evaluate_function(best_x, best_y)
    for _ in range(iterations):
        # Add a random perturbation to the best_x and best_y values
        perturbation_x = np.random.uniform(-1, 1)
        perturbation_y = np.random.uniform(-1, 1)
        x = best_x + perturbation_x
        y = best_y + perturbation_y
        # Ensure x and y are within the bounds
        x = max(bounds[0], min(x, bounds[1]))
        y = max(bounds[0], min(y, bounds[1]))
        value = evaluate_function(x, y)
        if value < best_value:
            best_value = value
            best_x, best_y = x, y

    return best_x, best_y, best_value


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def evaluate_function(x, y):
    \"\"\"The complex function we're trying to minimize\"\"\"
    return np.sin(x) * np.cos(y) + np.sin(x * y) + (x**2 + y**2) / 20


def run_search():
    x, y, value = search_algorithm()
    return x, y, value


if __name__ == "__main__":
    x, y, value = run_search()
    print(f"Found minimum at ({x}, {y}) with value {value}")"""

        result = apply_diff(original_code, diff_text)

        # Normalize whitespace for comparison
        self.assertEqual(
            result,
            expected_code,
        )


class TestFormatDiffSummary(unittest.TestCase):
    """Tests for format_diff_summary showing actual diff content"""

    def test_single_line_changes(self):
        """Single-line changes should show inline format"""
        diff_blocks = [("x = 1", "x = 2")]
        result = format_diff_summary(diff_blocks)
        self.assertEqual(result, "Change 1: 'x = 1' to 'x = 2'")

    def test_multi_line_changes_show_actual_content(self):
        """Multi-line changes should show actual SEARCH/REPLACE content"""
        diff_blocks = [
            (
                "def old():\n    return False",
                "def new():\n    return True",
            )
        ]
        result = format_diff_summary(diff_blocks)
        # Should contain actual code, not "2 lines"
        self.assertIn("def old():", result)
        self.assertIn("return False", result)
        self.assertIn("def new():", result)
        self.assertIn("return True", result)
        self.assertIn("Replace:", result)
        self.assertIn("with:", result)
        # Should NOT contain generic line count
        self.assertNotIn("2 lines", result)

    def test_multiple_diff_blocks(self):
        """Multiple diff blocks should be numbered"""
        diff_blocks = [
            ("a = 1", "a = 2"),
            ("def foo():\n    pass", "def bar():\n    return 1"),
        ]
        result = format_diff_summary(diff_blocks)
        self.assertIn("Change 1:", result)
        self.assertIn("Change 2:", result)
        self.assertIn("'a = 1' to 'a = 2'", result)
        self.assertIn("def foo():", result)
        self.assertIn("def bar():", result)

    def test_configurable_max_line_len(self):
        """max_line_len parameter should control line truncation"""
        long_line = "x" * 50
        # Must be multi-line to trigger block format (single-line uses inline format)
        diff_blocks = [(long_line + "\nline2", "short\nline2")]
        # With default (100), no truncation
        result_default = format_diff_summary(diff_blocks)
        self.assertNotIn("...", result_default)
        # With max_line_len=30, should truncate the long line
        result_short = format_diff_summary(diff_blocks, max_line_len=30)
        self.assertIn("...", result_short)

    def test_configurable_max_lines(self):
        """max_lines parameter should control block truncation"""
        many_lines = "\n".join([f"line{i}" for i in range(20)])
        diff_blocks = [(many_lines, "replacement")]
        # With max_lines=10, should truncate
        result = format_diff_summary(diff_blocks, max_lines=10)
        self.assertIn("... (10 more lines)", result)

    def test_block_lines_basic_formatting(self):
        """Lines should be indented with 2 spaces"""
        lines = ["line1", "line2"]
        result = _format_block_lines(lines)
        self.assertEqual(result, "  line1\n  line2")

    def test_block_lines_long_line_truncation(self):
        """Lines over 100 chars should be truncated by default"""
        long_line = "x" * 150
        result = _format_block_lines([long_line])
        self.assertIn("...", result)
        self.assertLess(len(result.split("\n")[0]), 110)

    def test_block_lines_many_lines_truncation(self):
        """More than 30 lines should show truncation message by default"""
        lines = [f"line{i}" for i in range(50)]
        result = _format_block_lines(lines)
        self.assertIn("... (20 more lines)", result)
        self.assertEqual(len(result.split("\n")), 31)

    def test_block_lines_empty_input(self):
        """Empty input should return '(empty)'"""
        result = _format_block_lines([])
        self.assertEqual(result, "  (empty)")


if __name__ == "__main__":
    unittest.main()
