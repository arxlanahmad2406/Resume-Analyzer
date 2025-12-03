# Resume Relevancy Analyzer

A tool that analyzes how well a resume matches a job description using TF-IDF cosine similarity. Available as a web app or standalone desktop application.

## Features

- Paste job descriptions in the left panel
- Upload resumes (PDF, DOCX, or TXT) in the right panel
- Get instant relevancy score with match category (Strong/Moderate/Weak)

## Requirements

- Python 3.x
- pipenv

## Installation

```bash
# Install pipenv if you don't have it
pip install pipenv

# Install dependencies
pipenv install
```

## Usage

### Web App (Development)

```bash
pipenv run python app.py
```

Open http://127.0.0.1:5000 in your browser.

### Desktop App (Development)

```bash
pipenv run python desktop_app.py
```

## Building Standalone App

### macOS (.app and .dmg)

**Step 1: Build the .app**

```bash
# Install dependencies (includes pyinstaller and pywebview)
pipenv install

# Build the app
pipenv run pyinstaller build_app.spec --clean
```

The `.app` will be created at: `dist/Resume Relevancy Analyzer.app`

**Step 2: Create DMG for distribution**

```bash
hdiutil create -volname "Resume Relevancy Analyzer" -srcfolder "dist/Resume Relevancy Analyzer.app" -ov -format UDZO ResumeAnalyzer.dmg
```

This creates `ResumeAnalyzer.dmg` which you can share with others.

**Alternative: Use the build script**

```bash
chmod +x build.sh
./build.sh
```

### Windows (.exe)

> **Note:** Must be run on a Windows machine.

**Step 1: Install dependencies**

```bash
pip install pipenv
pipenv install
```

**Step 2: Build the executable**

```bash
pipenv run pyinstaller build_app.spec --clean
```

The `.exe` will be created at: `dist/ResumeAnalyzer/ResumeAnalyzer.exe`

**Step 3: Distribute**

You can zip the entire `dist/ResumeAnalyzer/` folder and share it. Users run `ResumeAnalyzer.exe` to launch the app.

## Match Categories

| Score | Category |
|-------|----------|
| > 70% | Strong Match |
| 40-70% | Moderate Match |
| < 40% | Weak Match |
