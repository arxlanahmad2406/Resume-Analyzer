#!/bin/bash

echo "🔧 Building Resume Relevancy Analyzer..."

# Install build dependencies
pipenv install pyinstaller pywebview

# Build the app
pipenv run pyinstaller build_app.spec --clean

echo ""
echo "✅ Build complete!"
echo ""
echo "📁 Output locations:"
echo "   - macOS App: dist/Resume Relevancy Analyzer.app"
echo ""
echo "📦 To create DMG (optional):"
echo "   hdiutil create -volname 'Resume Relevancy Analyzer' -srcfolder 'dist/Resume Relevancy Analyzer.app' -ov -format UDZO ResumeAnalyzer.dmg"
echo ""
