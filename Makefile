.PHONY: install run build-mac build-win dmg clean help

# Default target
help:
	@echo "Resume Relevancy Analyzer - Available Commands"
	@echo "=============================================="
	@echo ""
	@echo "  make install     - Install all dependencies using pipenv"
	@echo "  make run         - Run the Flask web application"
	@echo "  make build-mac   - Build macOS .app bundle"
	@echo "  make build-win   - Build Windows .exe (run on Windows)"
	@echo "  make dmg         - Create macOS .dmg installer"
	@echo "  make clean       - Remove build artifacts"
	@echo "  make help        - Show this help message"
	@echo ""

# Install dependencies
install:
	pipenv install

# Run the Flask web application
run:
	pipenv run python app.py

# Build macOS .app bundle
build-mac:
	@echo "Building macOS application..."
	pipenv run pyinstaller --clean build_app.spec
	@echo ""
	@echo "✅ Build complete! App located at: dist/Resume Relevancy Analyzer.app"
	@echo "Run 'make dmg' to create a distributable DMG file."

# Build Windows .exe (must be run on Windows)
build-win:
	@echo "Building Windows executable..."
	pipenv run pyinstaller --clean --onefile --windowed \
		--name "ResumeAnalyzer" \
		--add-data "templates;templates" \
		--add-data "static;static" \
		--hidden-import sklearn.utils._typedefs \
		--hidden-import sklearn.utils._heap \
		--hidden-import sklearn.utils._sorting \
		--hidden-import sklearn.neighbors._partition_nodes \
		desktop_app.py
	@echo ""
	@echo "✅ Build complete! Executable located at: dist/ResumeAnalyzer.exe"

# Create macOS DMG installer
dmg: build-mac
	@echo "Creating DMG installer..."
	@rm -f ResumeAnalyzer.dmg
	@mkdir -p dmg_temp
	@cp -R "dist/Resume Relevancy Analyzer.app" dmg_temp/
	@ln -s /Applications dmg_temp/Applications
	hdiutil create -volname "Resume Relevancy Analyzer" \
		-srcfolder dmg_temp \
		-ov -format UDZO \
		ResumeAnalyzer.dmg
	@rm -rf dmg_temp
	@echo ""
	@echo "✅ DMG created: ResumeAnalyzer.dmg"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/ dist/ dmg_temp/
	rm -f *.dmg *.spec.bak
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Clean complete"
