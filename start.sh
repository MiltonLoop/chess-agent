#!/bin/bash
# Launcher script for the chess agent daemon
# Sources environment variables and starts the agent

source /Users/magnus/.zshrc 2>/dev/null
cd /Users/magnus/Documents/chess-agent
source .venv/bin/activate
exec python run.py
