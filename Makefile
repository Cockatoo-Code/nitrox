.PHONY: help test test-verbose coverage lint format type-check install install-dev build clean docs examples

# Default target
help:
	@echo "🫧 Nitrox Development Commands"
	@echo "==============================="
	@echo "install        Install nitrox for development"
	@echo "install-dev    Install with development dependencies"
	@echo "test           Run tests"
	@echo "test-verbose   Run tests with verbose output"
	@echo "coverage       Run tests with coverage report"
	@echo "lint           Run linting (flake8)"
	@echo "format         Format code (black + isort)"
	@echo "type-check     Run type checking (mypy)"
	@echo "build          Build distribution packages"
	@echo "clean          Clean build artifacts"
	@echo "examples       Run example scripts"
	@echo "check-ffmpeg   Check if ffmpeg is installed"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# Testing
test:
	pytest

test-verbose:
	pytest -v -s

coverage:
	pytest --cov=nitrox --cov-report=html --cov-report=term-missing

# Code quality
lint:
	flake8 nitrox tests examples

format:
	black nitrox tests examples
	isort nitrox tests examples

type-check:
	mypy nitrox

# Building
build:
	python -m build

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Examples and demos
examples:
	@echo "Running basic examples..."
	python examples/basic_usage.py

check-ffmpeg:
	@echo "Checking for ffmpeg..."
	@which ffmpeg > /dev/null && echo "✅ ffmpeg found" || echo "❌ ffmpeg not found - install from https://ffmpeg.org"
	@which ffprobe > /dev/null && echo "✅ ffprobe found" || echo "❌ ffprobe not found - install from https://ffmpeg.org"

# Development workflow
dev-setup: install-dev check-ffmpeg
	@echo "🚀 Development environment ready!"

# Quality check all
check-all: lint type-check test
	@echo "✅ All quality checks passed!" 