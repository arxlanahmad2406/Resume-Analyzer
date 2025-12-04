#!/usr/bin/env bash
# Build Windows .exe (must be run on Windows with Git Bash or WSL)

set -e

cd "$(dirname "$0")/.."

echo "🪟 Building Windows Executable"
echo "==============================="

# Check if running on Windows
if [[ "$OSTYPE" != "msys" && "$OSTYPE" != "cygwin" && "$OSTYPE" != "win32" ]]; then
    echo "⚠️  Warning: This script should be run on Windows."
    echo "   Cross-compilation is not supported by PyInstaller."
    echo ""
    read -p "Continue anyway? (y/n): " continue_build
    if [ "$continue_build" != "y" ] && [ "$continue_build" != "Y" ]; then
        exit 1
    fi
fi

make build-win

echo ""
echo "🎉 Done! Executable: dist/ResumeAnalyzer.exe"
