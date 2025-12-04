# Resume Relevancy Analyzer

AI-powered tool that analyzes how well a resume matches a job description using OpenAI GPT-4o-mini and semantic embeddings. Available as a web app or standalone desktop application.

## Features

- **AI-Powered Analysis** — Uses OpenAI GPT-4o-mini for intelligent matching
- **Semantic Similarity** — OpenAI embeddings for deep understanding
- **Detailed Results** — Shows matched skills, missing skills, and experience match
- **Multiple Formats** — Supports PDF, DOCX, and TXT resumes
- **Fallback Mode** — Works without API key using TF-IDF (less accurate)

## Requirements

- Python 3.x
- pipenv
- OpenAI API Key (optional, but recommended for best accuracy)

## Quick Start

```bash
# Install dependencies
./scripts/install.sh

# Or using make
make install

# Add your OpenAI API key to .env
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# Run the app
make run
```

Open http://127.0.0.1:5000 in your browser.

## Available Commands

Run `make help` to see all available commands:

| Command | Description |
|---------|-------------|
| `make install` | Install all dependencies using pipenv |
| `make run` | Run the Flask web application |
| `make build-mac` | Build macOS .app bundle |
| `make build-win` | Build Windows .exe (run on Windows) |
| `make dmg` | Create macOS .dmg installer |
| `make clean` | Remove build artifacts |
| `make help` | Show help message |

## Scripts

Helper scripts are available in the `scripts/` directory:

| Script | Description |
|--------|-------------|
| `./scripts/install.sh` | Install dependencies and setup .env |
| `./scripts/run.sh` | Run the web application |
| `./scripts/build-mac.sh` | Build macOS app with optional DMG |
| `./scripts/build-win.sh` | Build Windows executable |

## Configuration

Copy `.env.sample` to `.env` and add your API key:

```bash
cp .env.sample .env
```

Edit `.env`:
```
OPENAI_API_KEY=sk-your-openai-api-key-here
```

## Building Standalone Apps

### macOS (.app and .dmg)

**Option 1: Using Make (Recommended)**

```bash
# Build .app only
make build-mac

# Build .app and create .dmg
make dmg
```

**Option 2: Using Script**

```bash
./scripts/build-mac.sh
```

**Output:**
- `.app` bundle: `dist/Resume Relevancy Analyzer.app`
- DMG installer: `ResumeAnalyzer.dmg`

### Windows (.exe)

> ⚠️ **Note:** Must be run on a Windows machine. Cross-compilation is not supported.

**Option 1: Using Make**

```bash
make build-win
```

**Option 2: Using Script (Git Bash/WSL)**

```bash
./scripts/build-win.sh
```

**Option 3: Manual Build**

```bash
pipenv run pyinstaller --clean --onefile --windowed ^
    --name "ResumeAnalyzer" ^
    --add-data "templates;templates" ^
    --add-data "static;static" ^
    desktop_app.py
```

**Output:** `dist/ResumeAnalyzer.exe`

**Distribution:** Zip the `dist/ResumeAnalyzer/` folder and share. Users run `ResumeAnalyzer.exe`.

## Match Categories

| Score | Category | Description |
|-------|----------|-------------|
| > 70% | Strong Match | Excellent fit for the role |
| 40-70% | Moderate Match | Good potential, some gaps |
| < 40% | Weak Match | Significant skill gaps |

## Project Structure

```
python-ar/
├── app.py              # Flask web application
├── desktop_app.py      # Desktop app wrapper (pywebview)
├── build_app.spec      # PyInstaller configuration
├── MakeFile            # Build automation
├── Pipfile             # Python dependencies
├── .env.sample         # Environment variables template
├── templates/          # HTML templates
├── static/             # Static assets (favicon)
└── scripts/            # Helper scripts
    ├── install.sh
    ├── run.sh
    ├── build-mac.sh
    └── build-win.sh
```

## License

MIT
