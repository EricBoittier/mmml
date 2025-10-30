#!/bin/bash
# Run MMML test suite

set -e

echo "======================================"
echo "  MMML Test Suite"
echo "======================================"
echo

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "❌ pytest not found. Installing..."
    pip install pytest
fi

# Run tests
echo "Running tests..."
echo

# Run with different options based on arguments
if [ "$1" == "quick" ]; then
    echo "Running quick tests only..."
    pytest tests/ -v -k "not slow"
elif [ "$1" == "coverage" ]; then
    echo "Running with coverage..."
    if ! command -v pytest-cov &> /dev/null; then
        echo "Installing pytest-cov..."
        pip install pytest-cov
    fi
    pytest tests/ --cov=mmml --cov-report=html --cov-report=term
elif [ "$1" == "cli" ]; then
    echo "Running CLI tests only..."
    pytest tests/test_cli.py -v
elif [ "$1" == "integration" ]; then
    echo "Running integration tests..."
    pytest tests/test_xml_conversion.py tests/test_cli.py -v
elif [ "$1" == "unit" ]; then
    echo "Running unit tests..."
    pytest tests/test_data_loading.py -v
else
    echo "Running all tests..."
    pytest tests/ -v
fi

echo
echo "======================================"
echo "  Tests complete!"
echo "======================================"

