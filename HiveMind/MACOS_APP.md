# Creating a macOS App for HiveMind AI

This approach keeps HiveMind AI local, but launches it like a desktop application. The app wrapper starts Streamlit from your project virtual environment on `127.0.0.1`, waits for it to become available, and opens it inside a native macOS window using `pywebview`.

## 1. Install Packaging Dependencies

From your project folder:

```bash
cd /path/to/oracle-agent-memory
source .venv/bin/activate
pip install pyinstaller pywebview
```

The HiveMind app itself also needs the normal application dependencies, including Streamlit, FastAPI, Uvicorn, Oracle Agent Memory, `oracledb`, `langchain-oci`, and the OCI SDK.

## 2. Store Connection Details Outside the Script

Keep database and OCI values out of source files. For local use, set them in your shell profile, a private `.env` file, or your app launcher:

```bash
export HIVEMIND_DB_USER="your_db_user"
export HIVEMIND_DB_PASSWORD="your_db_password"
export HIVEMIND_DB_DSN="your_adb_connection_string"
export HIVEMIND_USER_ID="your-user-id"
export HIVEMIND_AGENT_ID="engagement_tracker"
export OCI_CONFIG_FILE="$HOME/.oci/config"
export OCI_PROFILE="DEFAULT"
```

Do not commit real OCIDs, database passwords, wallet locations, or connection strings to a public repository.

## 3. Create a Desktop Launcher Script

Create `hivemind_desktop.py` in the project root. Replace `/path/to/oracle-agent-memory` with your local project folder:

```python
from __future__ import annotations

import os
import subprocess
import time
import urllib.request
from pathlib import Path

import webview


URL = "http://127.0.0.1:8501"
PROJECT_DIR = Path("/path/to/oracle-agent-memory")
LOG_DIR = PROJECT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)


def is_ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=1) as response:
            return 200 <= response.status < 500
    except Exception:
        return False


def wait_until_ready(url: str, process: subprocess.Popen | None, timeout: int = 45) -> None:
    start = time.time()
    while time.time() - start < timeout:
        if is_ready(url):
            return
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                "Streamlit exited before HiveMind AI became available. "
                f"See {LOG_DIR / 'hivemind_desktop_streamlit.log'}."
            )
        time.sleep(0.5)
    raise RuntimeError("HiveMind AI did not start in time.")


env = os.environ.copy()
# Optional: set private values here if your Finder-launched app cannot see shell exports.
# env["HIVEMIND_DB_USER"] = "your_db_user"
# env["HIVEMIND_DB_PASSWORD"] = "your_db_password"
# env["HIVEMIND_DB_DSN"] = "your_adb_connection_string"

log_file = (LOG_DIR / "hivemind_desktop_streamlit.log").open("a", encoding="utf-8")
streamlit: subprocess.Popen | None = None
started_streamlit = False

if not is_ready(URL):
    log_file.write(f"\n\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting Streamlit\n")
    log_file.flush()
    streamlit = subprocess.Popen(
        [
            str(PROJECT_DIR / ".venv/bin/streamlit"),
            "run",
            "engagement_tracker.py",
            "--server.address",
            "127.0.0.1",
            "--server.port",
            "8501",
            "--server.headless",
            "true",
            "--server.fileWatcherType",
            "none",
            "--browser.gatherUsageStats",
            "false",
        ],
        cwd=PROJECT_DIR,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        start_new_session=True,
    )
    started_streamlit = True

try:
    wait_until_ready(URL, streamlit)
    webview.create_window("HiveMind AI", URL, width=1200, height=850)
    webview.start()
except Exception as exc:
    error_html = f"""
    <html>
      <body style="font-family: -apple-system; background: #111; color: #eee; padding: 32px;">
        <h2>HiveMind AI could not start</h2>
        <p>{exc}</p>
        <p>Check the log file:</p>
        <code>{LOG_DIR / 'hivemind_desktop_streamlit.log'}</code>
      </body>
    </html>
    """
    webview.create_window("HiveMind AI startup error", html=error_html, width=760, height=420)
    webview.start()
finally:
    if started_streamlit and streamlit is not None and streamlit.poll() is None:
        streamlit.terminate()
```

## 4. Build the App with PyInstaller

Use an `.icns` file for a proper macOS app icon:

```bash
cd /path/to/oracle-agent-memory
source .venv/bin/activate
pyinstaller \
  --windowed \
  --name "HiveMind AI" \
  --icon assets/hivemind_ai_icon.icns \
  hivemind_desktop.py
```

The built app will be created at:

```text
dist/HiveMind AI.app
```

## 5. Move It to Applications

If you already have an older copy in Applications, remove it first:

```bash
rm -rf "/Applications/HiveMind AI.app"
cp -R "dist/HiveMind AI.app" "/Applications/HiveMind AI.app"
```

If macOS reports a permission error, run the same copy command with `sudo`.

## 6. Run It

Launch `HiveMind AI` from Applications or Spotlight. The app starts the local Streamlit service in the background and displays it in a native window, so there is no need to keep a browser tab open.

If the app cannot connect, check that the database is active and that your private environment variables are available to the launched process.
