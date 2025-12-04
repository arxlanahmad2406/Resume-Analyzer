#!/usr/bin/env bash
# Run the Resume Relevancy Analyzer web application

set -e

cd "$(dirname "$0")/.."

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found. Copy .env.sample to .env and add your API key."
    echo "   cp .env.sample .env"
fi

make run