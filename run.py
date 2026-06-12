#!/usr/bin/env python3
"""
Startup script for the TTS Benchmarking Tool
"""
import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if requirements are installed"""
    try:
        import streamlit
        import plotly
        import pandas
        import numpy
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def check_environment():
    """Check environment configuration"""
    print("\n🔍 Checking environment configuration...")
    
    # Check API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    murf_key = os.getenv("MURF_API_KEY")
    
    if openai_key:
        print("✅ OpenAI API key found")
    else:
        print("⚠️  OpenAI API key not found (set OPENAI_API_KEY)")
    
    if murf_key:
        print("✅ Murf AI API key found")
    else:
        print("⚠️  Murf AI API key not found (set MURF_API_KEY)")
    
    if not openai_key and not murf_key:
        print("\n📝 To set API keys:")
        print("   export OPENAI_API_KEY=your_openai_key")
        print("   export MURF_API_KEY=your_murf_key")
        print("\n   Or create a .env file with these variables")
    
    return True

def main():
    """Main startup function"""
    print("🎙️  TTS Benchmarking Tool")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not Path("app.py").exists():
        print("❌ app.py not found. Please run from the project directory.")
        sys.exit(1)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check environment
    check_environment()
    
    # Start the application
    print("\n🚀 Starting Streamlit application...")
    print("   Open your browser to: http://localhost:8501")
    print("   Press Ctrl+C to stop the application")
    print("-" * 40)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port=8501",
            "--server.address=0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
