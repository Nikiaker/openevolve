"""
Tests for code utilities in openevolve.utils.code_utils
"""

import unittest
from openevolve.utils.code_utils import apply_diff, extract_diffs


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


if __name__ == "__main__":
    unittest.main()
