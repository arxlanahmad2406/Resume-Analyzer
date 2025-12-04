#!/usr/bin/env bash
# Build macOS .app bundle and optionally create DMG

set -e

cd "$(dirname "$0")/.."

echo "🍎 Building macOS Application"
echo "=============================="

# Check for .env
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found. The app will use fallback mode without AI."
fi

# Build the app
make build-mac

echo ""
read -p "Create DMG installer? (y/n): " create_dmg

if [ "$create_dmg" = "y" ] || [ "$create_dmg" = "Y" ]; then
    make dmg
fi

echo ""
echo "🎉 Done!"
