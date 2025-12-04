#!/usr/bin/env bash
# Install dependencies for Resume Relevancy Analyzer

set -e

cd "$(dirname "$0")/.."

echo "📦 Installing Dependencies"
echo "=========================="

# Check for pipenv
if ! command -v pipenv &> /dev/null; then
    echo "❌ pipenv not found. Installing..."
    pip install pipenv
fi

# Install dependencies
pipenv install

# Setup .env if not exists
if [ ! -f .env ]; then
    echo ""
    echo "📝 Creating .env file from sample..."
    cp .env.sample .env
    echo "⚠️  Please edit .env and add your OPENAI_API_KEY"
fi

echo ""
echo "✅ Installation complete!"
echo ""
echo "Next steps:"
echo "  1. Add your OpenAI API key to .env"
echo "  2. Run: make run (or ./scripts/run.sh)"
