#!/bin/bash

# TTS Benchmarking Tool Startup Script
echo "🎙️ Starting TTS Benchmarking Tool..."

# Navigate to project directory
cd "$(dirname "$0")"

# Load environment variables
if [ -f .env ]; then
    echo "📋 Loading API keys from .env file..."
    source .env
else
    echo "⚠️ No .env file found. Please create one with your API keys."
fi

# Activate virtual environment
if [ -d "venv" ]; then
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
else
    echo "❌ Virtual environment not found. Please run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Check API keys (Gen2 + NewModel blind test)
if [ -z "$MURF_API_KEY" ] || [ -z "$OMNI_API_KEY" ] || { [ -z "$OMNI_HOST" ] && [ -z "$OMNI_BASE_URL" ]; }; then
    echo "⚠️ For the listening test, set in .env (see .env.example):"
    echo "   MURF_API_KEY=..."
    echo "   OMNI_API_KEY=..."
    echo "   OMNI_HOST=host:port   # or OMNI_BASE_URL=https://.../tts"
fi

# Start the application
echo "🚀 Starting Streamlit application..."
echo "   📱 Open your browser to: http://localhost:8501"
echo "   ⏹️  Press Ctrl+C to stop the application"
echo "----------------------------------------"

streamlit run app.py --server.port=8501 --server.address=0.0.0.0
