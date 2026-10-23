"""Convenient test runner script for running the pytest test suite."""

import os
import sys

# Disable setuptools plugin autoloading to prevent conflicts with global environment plugins
os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

import pytest

if __name__ == "__main__":
    args = sys.argv[1:] if len(sys.argv) > 1 else ["-v"]
    sys.exit(pytest.main(args))
