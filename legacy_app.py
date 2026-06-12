"""
TTS Benchmarking Tool - Streamlit Application
"""
import streamlit as st
import asyncio
import pandas as pd
import plotly.express as px
import json
import base64
import time 
import os
import csv
import io
from datetime import datetime
from typing import Dict, List, Any

from dotenv import load_dotenv
load_dotenv()

try:
    import openai
except ImportError:
    openai = None

from config import TTS_PROVIDERS, UI_CONFIG, validate_config
from dataset import DatasetGenerator, TestSample
from benchmarking_engine import BenchmarkEngine, BenchmarkResult
from tts_providers import TTSProviderFactory, TTSRequest
import visualizations
from security import session_manager
from geolocation import geo_service
from database import BenchmarkDatabase
from voice_battle_locale_defaults import bundled_default_sentences_voice_battle

st.set_page_config(
    page_title=UI_CONFIG["page_title"],
    page_icon=UI_CONFIG["page_icon"],
    layout=UI_CONFIG["layout"],
    initial_sidebar_state="expanded"
)

def load_css():
    with open('styles.css', 'r') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css()

if "benchmark_engine" not in st.session_state:
    st.session_state.benchmark_engine = BenchmarkEngine()

if "dataset_generator" not in st.session_state:
    st.session_state.dataset_generator = DatasetGenerator()

if "results" not in st.session_state:
    st.session_state.results = []

db = BenchmarkDatabase()

if "config_valid" not in st.session_state:
    st.session_state.config_valid = False

if "navigate_to" not in st.session_state:
    st.session_state.navigate_to = None

def get_model_name(provider: str) -> str:
    """Helper function to get model name from config"""
    return TTS_PROVIDERS.get(provider).model_name if provider in TTS_PROVIDERS else provider

def get_provider_display_name(provider_id: str) -> str:
    """Get display name for provider - shows 'Murf' instead of 'Murf Falcon Oct 23'"""
    if provider_id in TTS_PROVIDERS:
        return TTS_PROVIDERS[provider_id].name
    return provider_id.title()

def get_provider_name_for_de_anonymization(provider_id: str) -> str:
    """Get full provider name for de-anonymization in comments
    Returns full names like 'Murf Falcon' and 'Murf Zeroshot' instead of just 'Murf'
    """
    # Special handling for Murf providers to distinguish between Falcon and Zeroshot
    if provider_id == "murf_gen2":
        return "Murf Gen 2"
    elif provider_id == "omni_tts":
        return "NewModel"
    elif provider_id in TTS_PROVIDERS:
        return TTS_PROVIDERS[provider_id].name
    return provider_id.title()

def de_anonymize_comment(comment: str, provider_a_id: str, provider_b_id: str) -> str:
    """De-anonymize comment by replacing 'Sample A'/'Sample B' and standalone 'A'/'B' with actual provider names
    
    Examples:
    - "Sample B is more natural" -> "Zeroshot is more natural"
    - "A is good" -> "Zeroshot is good"
    - "b sounds better" -> "Murf sounds better"
    - "Sample a is great" -> "Zeroshot is great"
    """
    if not comment:
        return comment
    
    # Use full provider names for de-anonymization (e.g., "Murf Falcon" not just "Murf")
    provider_a_name = get_provider_name_for_de_anonymization(provider_a_id)
    provider_b_name = get_provider_name_for_de_anonymization(provider_b_id)
    
    import re
    
    # Replace "Sample A" or "Sample a" (case-insensitive, word boundaries)
    # Handle various patterns: "Sample A", "sample A", "Sample A's", "Sample A is", etc.
    comment = re.sub(r'\bSample\s+A\b', provider_a_name, comment, flags=re.IGNORECASE)
    
    # Replace "Sample B" or "Sample b" (case-insensitive, word boundaries)
    comment = re.sub(r'\bSample\s+B\b', provider_b_name, comment, flags=re.IGNORECASE)
    
    # Replace standalone "A" or "a" (case-insensitive, word boundaries ensure it's not part of another word)
    # This handles: "A is good", "a sounds better", "I prefer A", etc.
    # Note: Do this AFTER "Sample A" replacement to avoid double replacement
    comment = re.sub(r'\bA\b', provider_a_name, comment, flags=re.IGNORECASE)
    
    # Replace standalone "B" or "b" (case-insensitive, word boundaries ensure it's not part of another word)
    # This handles: "B is good", "b sounds better", "I prefer B", etc.
    # Note: Do this AFTER "Sample B" replacement to avoid double replacement
    comment = re.sub(r'\bB\b', provider_b_name, comment, flags=re.IGNORECASE)
    
    return comment

def de_anonymize_comment_from_result(result_record: dict) -> str:
    """De-anonymize comment from a result record by inferring which provider was A and B
    
    Uses user_choice to determine: if user_choice == "A", then winner was Sample A, loser was Sample B
    """
    comment = result_record.get('comment', '')
    if not comment:
        return comment
    
    user_choice = result_record.get('user_choice', '')
    winner = result_record.get('winner', '')
    loser = result_record.get('loser', '')
    
    # Infer which provider was Sample A and which was Sample B
    if user_choice == "A":
        provider_a_id = winner
        provider_b_id = loser
    elif user_choice == "B":
        provider_a_id = loser
        provider_b_id = winner
    else:
        # If no user_choice, can't de-anonymize - return as is
        return comment
    
    return de_anonymize_comment(comment, provider_a_id, provider_b_id)

def get_location_display(result: BenchmarkResult = None, country: str = None, city: str = None) -> str:
    """Helper function to format location display with flag"""
    if result:
        country = result.location_country
        city = result.location_city
    
    if not country or country == 'Unknown':
        return '🌍 Unknown'
    
    flag = geo_service.get_country_flag(getattr(result, 'location_country', None) if result else None)
    
    if city and city != 'Unknown':
        return f"{flag} {city}, {country}"
    return f"{flag} {country}"

def check_configuration():
    """Check if API keys are configured"""
    config_status = validate_config()
    st.session_state.config_valid = config_status["valid"]
    return config_status

def main():
    """Main application function"""
    
    st.markdown("""
    <style>
    @keyframes catchyPulse {
        0% { 
            transform: scale(1);
            box-shadow: 0 0 0 0 rgba(255, 75, 75, 0.7);
        }
        50% { 
            transform: scale(1.15);
            box-shadow: 0 0 10px 5px rgba(255, 75, 75, 0);
        }
        100% { 
            transform: scale(1);
            box-shadow: 0 0 0 0 rgba(255, 75, 75, 0);
        }
    }
    .feature-banner {
        padding: 0;
        margin: 0 0 10px 0;
        display: flex;
        align-items: center;
        gap: 10px;
        justify-content: flex-end;
        position: relative;
    }
    .new-badge {
        animation: catchyPulse 1.5s ease-in-out infinite;
        display: inline-block;
        background: #ff4b4b;
        color: white;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
    }
    .feature-text {
        color: #262730;
        font-size: 15px;
        margin: 0;
        position: relative;
        display: inline-block;
    }
    .feature-text strong {
        position: relative;
        display: inline-block;
        padding-bottom: 5px;
    }
    .feature-text strong::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        height: 2px;
        overflow: hidden;
        background: repeating-linear-gradient(
            to right,
            #808080 0px,
            #808080 6px,
            transparent 6px,
            transparent 12px
        );
        background-size: 12px 2px;
        animation: moveDots 1.5s linear infinite;
    }
    @keyframes moveDots {
        0% { 
            background-position: 0 0;
        }
        100% { 
            background-position: 12px 0;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        default_page = "Blind A/B Test"
        
        st.subheader("Navigator")
        
        pages = ["Blind A/B Test", "Leaderboard", "Comments"]
        
        for i, page_name in enumerate(pages):
            if st.button(page_name, key=f"nav_{page_name}", use_container_width=True):
                st.session_state.current_page = page_name
                st.rerun()
        
        if "navigate_to" in st.session_state and st.session_state.navigate_to:
            page = st.session_state.navigate_to
            st.session_state.navigate_to = None
        else:
            page = st.session_state.get("current_page", "Blind A/B Test")
        
        st.divider()
        
        st.subheader("Models Configured")
        
        config_status = check_configuration()
        
        if config_status["valid"]:
            for provider_id, status in config_status["providers"].items():
                provider_name = TTS_PROVIDERS[provider_id].name
                # Show "Murf Falcon" in configuration sidebar for Murf
                if provider_id == "murf_gen2":
                    display_name = "Murf Gen 2"
                elif provider_id == "omni_tts":
                    display_name = "NewModel"
                else:
                    display_name = provider_name
                if status["configured"]:
                    st.write(f"🟢 {display_name}")
                else:
                    st.write(f"🔴 {display_name}")
        else:
            st.warning("⚠️ Both systems must be configured for the blind test.")
            st.markdown("**Set these environment variables:**")
            for provider_id, status in config_status["providers"].items():
                if not status["configured"]:
                    env_var = TTS_PROVIDERS[provider_id].api_key_env
                    provider_name = TTS_PROVIDERS[provider_id].name
                    # Show "Murf Falcon" in configuration sidebar for Murf
                    if provider_id == "murf_gen2":
                        display_name = "Murf Gen 2"
                    elif provider_id == "omni_tts":
                        display_name = "NewModel"
                    else:
                        display_name = provider_name
                    st.code(f"export {env_var}=your_api_key_here")
                    st.caption(f"For {display_name}")
    
    # During an active Voice Battle round, keep only the test UI (no marketing title).
    _hide_main_blind_chrome = (
        page == "Blind A/B Test"
        and st.session_state.get("blind_test_2_setup_complete", False)
        and not st.session_state.get("show_final_results_2", False)
    )
    if not _hide_main_blind_chrome:
        st.title("Murf Gen 2 vs NewModel")
        st.markdown("Blind listening test: same sentence, two systems. Listen before you vote.")
    
    if page == "Blind A/B Test":
        blind_test_2_page()
    elif page == "Leaderboard":
        leaderboard_page()
    elif page == "Comments":
        ranked_blind_comments_page()

def quick_test_page():
    """Quick test page for single TTS comparisons"""
    
    st.header("Quick Test")
    st.markdown("Test a single text prompt across multiple TTS providers")
    
    if "quick_test_results" not in st.session_state:
        st.session_state.quick_test_results = None
    
    config_status = check_configuration()
    
    if not st.session_state.config_valid:
        st.warning("Please configure at least one API key in the sidebar first.")
        return
    
    configured_providers = [
        provider_id for provider_id, status in config_status["providers"].items() 
        if status["configured"]
    ]
    
    if not configured_providers:
        st.error("No providers are configured. Please set API keys in the sidebar.")
        return
    
    text_input = st.text_area(
        "Enter text to synthesize:",
        value="Just to confirm, the co-applicant's name is spelled M-A-R-I-S-A, correct? I'll need her consent before I can proceed with income verification.",
        height=100,
        max_chars=1000
    )
    
    word_count = len(text_input.split())
    
    # Create display names for multiselect
    provider_display_options = []
    for p in configured_providers:
        # Show "Murf Falcon" in quick test dropdown for Murf
        if p == "murf_falcon_oct23":
            display_name = "Murf Falcon"
        elif p == "murf_zeroshot":
            display_name = "Murf Zeroshot"
        else:
            display_name = get_provider_display_name(p)
        provider_display_options.append(display_name)
    
    selected_display_names = st.multiselect(
        "Select providers:",
        provider_display_options,
        default=provider_display_options,
        help=f"Available providers: {', '.join(provider_display_options)}"
    )
    
    # Map back to provider IDs
    selected_providers = []
    for display_name in selected_display_names:
        for p in configured_providers:
            # Handle "Murf Falcon" mapping
            if display_name == "Murf Falcon" and p == "murf_falcon_oct23":
                selected_providers.append(p)
                break
            elif display_name == "Murf Zeroshot" and p == "murf_zeroshot":
                selected_providers.append(p)
                break
            elif get_provider_display_name(p) == display_name:
                selected_providers.append(p)
                break
        
    voice_options = {}
    if selected_providers:
        st.markdown("**Voice Selection:**")
        
        for i in range(0, len(selected_providers), 4):
            cols = st.columns(4)
            for j, provider in enumerate(selected_providers[i:i+4]):
                with cols[j]:
                    voices = TTS_PROVIDERS[provider].supported_voices
                    # Show "Murf Falcon" in quick test dropdown for Murf
                    if provider == "murf_falcon_oct23":
                        provider_display = "Murf Falcon"
                    elif provider == "murf_zeroshot":
                        provider_display = "Murf Zeroshot"
                    else:
                        provider_display = get_provider_display_name(provider)
                    voice_options[provider] = st.selectbox(
                        f"{provider_display} voice:",
                        voices,
                        key=f"voice_{provider}"
                    )
        
    if st.button("Generate & Compare", type="primary"):
        if text_input and selected_providers:
            valid, error_msg = session_manager.validate_request(text_input)
            if valid:
                run_quick_test(text_input, selected_providers, voice_options)
            else:
                st.error(f"❌ {error_msg}")
        else:
            st.warning("Please enter text and select at least one provider.")
    
    if st.session_state.quick_test_results is not None:
        st.markdown("---")  # Separator line
        display_quick_test_results(st.session_state.quick_test_results)

def run_quick_test(text: str, providers: List[str], voice_options: Dict[str, str]):
    """Run quick test for selected providers"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    
    async def test_provider(provider_id: str, voice: str):
        try:
            provider = TTSProviderFactory.create_provider(provider_id)
            
            # Create test sample
            sample = TestSample(
                id="quick_test",
                text=text,
                word_count=len(text.split()),
                category="user_input",
                length_category="custom",
                complexity_score=0.5
            )
            
            result = await st.session_state.benchmark_engine.run_single_test(
                provider, sample, voice
            )
            return result
            
        except Exception as e:
            st.error(f"Error testing {provider_id}: {str(e)}")
            return None
    
    # Run tests
    for i, provider_id in enumerate(providers):
        status_text.text(f"Testing {provider_id}...")
        
        voice = voice_options[provider_id]
        result = asyncio.run(test_provider(provider_id, voice))
        
        if result:
            results.append(result)
        
        progress_bar.progress((i + 1) / len(providers))
    
    status_text.text("✅ Tests completed!")
    
    # Clean up progress indicators after a moment
    import time
    time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()
    
    # Store results in session state for display
    if results:
        st.session_state.quick_test_results = results
    else:
        st.error("No successful results to display.")
        st.session_state.quick_test_results = None

def display_quick_test_results(results: List[BenchmarkResult]):
    """Display quick test results"""
    
    st.subheader("Test Results")
    
    data = []
    for result in results:
        provider_display = get_provider_display_name(result.provider)
        data.append({
            "Provider": provider_display,
            "Model": result.model_name,
            "Location": get_location_display(result),
            "Success": "✅" if result.success else "❌",
            "TTFB (ms)": f"{result.ttfb:.1f}" if result.success and result.ttfb > 0 else "N/A",
            "File Size (KB)": f"{result.file_size_bytes / 1024:.1f}" if result.success else "N/A",
            "Voice": result.voice,
            "Error": result.error_message if not result.success else ""
        })
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)
    
    successful_results = [r for r in results if r.success]
    
    if len(successful_results) > 1:
        col1, col2 = st.columns(2)
        
        with col1:
            fig_ttfb = px.bar(
                x=[get_provider_display_name(r.provider) for r in successful_results],
                y=[r.ttfb for r in successful_results],
                title="TTFB Comparison",
                labels={"x": "Provider", "y": "TTFB (ms)"}
            )
            st.plotly_chart(fig_ttfb, use_container_width=True)
        
        with col2:
            fig_size = px.bar(
                x=[get_provider_display_name(r.provider) for r in successful_results],
                y=[r.file_size_bytes / 1024 for r in successful_results],
                title="File Size Comparison",
                labels={"x": "Provider", "y": "File Size (KB)"}
            )
            st.plotly_chart(fig_size, use_container_width=True)
    
    st.subheader("Audio Playback")
    
    if len(successful_results) >= 1:
        st.markdown("**Listen to the audio samples:**")
        
        for i in range(0, len(successful_results), 4):
            cols = st.columns(4)
            for j, result in enumerate(successful_results[i:i+4]):
                with cols[j]:
                    provider_display = get_provider_display_name(result.provider)
                    st.markdown(f"**{provider_display}**")
                    st.caption(f"Model: {result.model_name}")
                    
                    if result.audio_data:
                        audio_base64 = base64.b64encode(result.audio_data).decode()
                        audio_html = f"""
                        <audio controls controlsList="nodownload" style="width: 100%;">
                            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mpeg">
                        </audio>
                        """
                        st.markdown(audio_html, unsafe_allow_html=True)
                        st.caption(f"TTFB: {result.ttfb:.1f}ms")
                        st.caption(f"Size: {result.file_size_bytes/1024:.1f} KB")
                        
                        st.download_button(
                            label="Download MP3",
                            data=result.audio_data,
                            file_name=f"{result.provider}_{result.voice}.mp3",
                            mime="audio/mpeg",
                            key=f"download_{result.provider}_{i}_{j}"
                        )

def blind_test_page():
    """Blind test page for unbiased audio quality comparison - Voice Battles style"""
    
    st.header("Blind Test")
    st.markdown("Compare TTS providers head-to-head. Listen to at least 3 seconds of each sample before voting.")
    
    config_status = check_configuration()
    
    if not st.session_state.config_valid:
        st.warning("Please configure at least one API key in the sidebar first.")
        return
    
    configured_providers = [
        provider_id for provider_id, status in config_status["providers"].items() 
        if status["configured"]
    ]
    
    if len(configured_providers) < 2:
        st.warning("⚠️ Blind test requires at least 2 configured providers. Please configure more API keys.")
        return
    
    # Initialize session state for progressive blind test
    if "blind_test_sentences" not in st.session_state:
        st.session_state.blind_test_sentences = []
    if "blind_test_current_pair" not in st.session_state:
        st.session_state.blind_test_current_pair = None
    if "blind_test_comparison_count" not in st.session_state:
        st.session_state.blind_test_comparison_count = 0
    if "blind_test_max_comparisons" not in st.session_state:
        st.session_state.blind_test_max_comparisons = 25
    if "blind_test_results_history" not in st.session_state:
        st.session_state.blind_test_results_history = []
    if "blind_test_selected_competitors" not in st.session_state:
        st.session_state.blind_test_selected_competitors = []
    if "blind_test_murf_voice" not in st.session_state:
        st.session_state.blind_test_murf_voice = None
    if "blind_test_murf_voices" not in st.session_state:
        st.session_state.blind_test_murf_voices = []  # List of selected MURF voices (up to 4)
    if "blind_test_gender_filter" not in st.session_state:
        st.session_state.blind_test_gender_filter = "female"
    if "blind_test_setup_complete" not in st.session_state:
        st.session_state.blind_test_setup_complete = False
    if "blind_test_audio_played" not in st.session_state:
        st.session_state.blind_test_audio_played = {"A": 0, "B": 0}
    if "show_final_results" not in st.session_state:
        st.session_state.show_final_results = False
    
    # If final results should be shown, display them directly
    if st.session_state.get("show_final_results", False):
        display_final_results()
        return
    
    # Show setup or comparison view
    # Preserve test state: if test is in progress, show comparison view (even if error occurred)
    # This ensures that if user navigates away and comes back, they continue where they left off
    test_in_progress = (
        st.session_state.blind_test_setup_complete or 
        st.session_state.blind_test_comparison_count > 0 or 
        st.session_state.blind_test_current_pair is not None or
        len(st.session_state.blind_test_results_history) > 0
    )
    
    if test_in_progress:
        # Test is in progress - show comparison view (will handle errors gracefully)
        display_blind_test_comparison()
    else:
        # No test in progress - show setup
        display_blind_test_setup(configured_providers)

def blind_test_2_page():
    """Blind test page for unbiased audio quality comparison - Voice Battles style"""
    config_status = check_configuration()
    
    if not st.session_state.config_valid:
        st.warning(
            "Configure **both** providers: `MURF_API_KEY`, `OMNI_API_KEY`, and `OMNI_HOST` or `OMNI_BASE_URL`."
        )
        return
    
    configured_providers = [
        provider_id for provider_id, status in config_status["providers"].items() 
        if status["configured"]
    ]
    
    if len(configured_providers) < 2:
        st.warning("⚠️ The blind test needs **Murf Gen 2** and **NewModel** fully configured (see sidebar).")
        return
    
    # Initialize session state for progressive blind test (using _2 suffix)
    if "blind_test_2_sentences" not in st.session_state:
        st.session_state.blind_test_2_sentences = []
    if "blind_test_2_current_pair" not in st.session_state:
        st.session_state.blind_test_2_current_pair = None
    if "blind_test_2_comparison_count" not in st.session_state:
        st.session_state.blind_test_2_comparison_count = 0
    if "blind_test_2_max_comparisons" not in st.session_state:
        st.session_state.blind_test_2_max_comparisons = 25
    if "blind_test_2_results_history" not in st.session_state:
        st.session_state.blind_test_2_results_history = []
    if "blind_test_2_selected_competitors" not in st.session_state:
        st.session_state.blind_test_2_selected_competitors = []
    if "blind_test_2_murf_voice" not in st.session_state:
        st.session_state.blind_test_2_murf_voice = None
    if "blind_test_2_murf_voices" not in st.session_state:
        st.session_state.blind_test_2_murf_voices = []  # List of selected MURF voices (up to 4)
    if "blind_test_2_gender_filter" not in st.session_state:
        st.session_state.blind_test_2_gender_filter = "female"
    if "blind_test_2_setup_complete" not in st.session_state:
        st.session_state.blind_test_2_setup_complete = False
    if "blind_test_2_audio_played" not in st.session_state:
        st.session_state.blind_test_2_audio_played = {"A": 0, "B": 0}
    if "show_final_results_2" not in st.session_state:
        st.session_state.show_final_results_2 = False
    if "blind_test_2_comparison_pairs" not in st.session_state:
        st.session_state.blind_test_2_comparison_pairs = None
    if "blind_test_2_pair_index" not in st.session_state:
        st.session_state.blind_test_2_pair_index = 0
    if "blind_test_2_locale_filter" not in st.session_state:
        st.session_state.blind_test_2_locale_filter = "US"
    
    # If final results should be shown, display them directly
    if st.session_state.get("show_final_results_2", False):
        display_final_results_2()
        return
    
    # Show setup or comparison view
    # Preserve test state: if test is in progress, show comparison view (even if error occurred)
    # This ensures that if user navigates away and comes back, they continue where they left off
    test_in_progress = (
        st.session_state.blind_test_2_setup_complete or 
        st.session_state.blind_test_2_comparison_count > 0 or 
        st.session_state.blind_test_2_current_pair is not None or
        len(st.session_state.blind_test_2_results_history) > 0
    )
    
    if test_in_progress:
        # Test is in progress - comparison only (no page header; prompt + audio + vote only)
        display_blind_test_2_comparison()
    else:
        st.header("Blind A/B: Murf Gen 2 vs NewModel")
        st.markdown("Listen to both samples (at least a few seconds each), then vote. Order is randomized each round.")
        display_blind_test_2_setup(configured_providers)

def falcon_vs_zeroshot_page():
    """Falcon vs Zeroshot comparison page - does NOT update leaderboard"""
    
    st.header("Falcon vs Zeroshot")
    st.markdown("Compare Murf Falcon and Zeroshot models head-to-head. Results do not affect leaderboard.")
    
    config_status = check_configuration()
    
    if not st.session_state.config_valid:
        st.warning("Please configure at least one API key in the sidebar first.")
        return
    
    # Check if both Murf providers are configured
    murf_falcon_configured = config_status["providers"].get("murf_falcon_oct23", {}).get("configured", False)
    murf_zeroshot_configured = config_status["providers"].get("murf_zeroshot", {}).get("configured", False)
    
    if not murf_falcon_configured or not murf_zeroshot_configured:
        st.error("⚠️ Both Murf Falcon and Murf Zeroshot must be configured to use this page.")
        return
    
    # Initialize session state for Falcon vs Zeroshot test
    if "fvs_sentences" not in st.session_state:
        st.session_state.fvs_sentences = []
    if "fvs_current_pair" not in st.session_state:
        st.session_state.fvs_current_pair = None
    if "fvs_comparison_count" not in st.session_state:
        st.session_state.fvs_comparison_count = 0
    if "fvs_max_comparisons" not in st.session_state:
        st.session_state.fvs_max_comparisons = 25
    # Don't reset it here - let the slider control it
    if "fvs_results_history" not in st.session_state:
        st.session_state.fvs_results_history = []
    if "fvs_falcon_voice" not in st.session_state:
        st.session_state.fvs_falcon_voice = None
    if "fvs_falcon_voices" not in st.session_state:
        st.session_state.fvs_falcon_voices = []
    if "fvs_zeroshot_voice" not in st.session_state:
        st.session_state.fvs_zeroshot_voice = None
    if "fvs_zeroshot_voices" not in st.session_state:
        st.session_state.fvs_zeroshot_voices = []
    if "fvs_gender_filter" not in st.session_state:
        st.session_state.fvs_gender_filter = "female"
    if "fvs_setup_complete" not in st.session_state:
        st.session_state.fvs_setup_complete = False
    if "fvs_audio_played" not in st.session_state:
        st.session_state.fvs_audio_played = {"A": 0, "B": 0}
    if "fvs_show_final_results" not in st.session_state:
        st.session_state.fvs_show_final_results = False
    if "fvs_locale_filter" not in st.session_state:
        st.session_state.fvs_locale_filter = None  # None means all locales
    
    # If final results should be shown, display them directly
    if st.session_state.get("fvs_show_final_results", False):
        display_fvs_final_results()
        return
    
    # Show setup or comparison view
    test_in_progress = (
        st.session_state.fvs_setup_complete or 
        st.session_state.fvs_comparison_count > 0 or 
        st.session_state.fvs_current_pair is not None or
        len(st.session_state.fvs_results_history) > 0
    )
    
    if test_in_progress:
        # Test is in progress - show comparison view
        display_fvs_comparison()
    else:
        # No test in progress - show setup
        display_fvs_setup()

def fvs_results_page():
    """Display aggregate locale-level win rates for Falcon vs Zeroshot (Dev-Prod)"""
    st.header("Results (Falcon vs 0shot)")
    st.markdown("Aggregate locale-level win rates for Dev-Prod comparison (persistent across sessions)")
    
    # Get results from database (persistent, like leaderboard)
    votes = db.get_fvs_votes()
    
    if not votes:
        st.info("No results available. Complete a Falcon vs Zeroshot test to see aggregate win rates.")
        return
    
    # Parse votes from database into result format
    results = []
    for winner, loser, text_sample, timestamp, metadata_json in votes:
        try:
            metadata = json.loads(metadata_json) if metadata_json else {}
            # Extract locale from metadata or voice
            locale = metadata.get("locale", "Unknown")
            if locale == "Unknown" and metadata.get("winner_voice"):
                voice_parts = metadata.get("winner_voice", "").split("-")
                if len(voice_parts) >= 2:
                    locale = "-".join(voice_parts[:2])
            
            result = {
                "winner": winner,
                "loser": loser,
                "winner_voice": metadata.get("winner_voice", ""),
                "loser_voice": metadata.get("loser_voice", ""),
                "text": metadata.get("text", text_sample),
                "comment": metadata.get("comment", ""),
                "locale": locale,
                "falcon_won": metadata.get("falcon_won", False),
                "user_choice": metadata.get("user_choice", ""),
                "winner_config": metadata.get("winner_config", {}),
                "loser_config": metadata.get("loser_config", {}),
                "timestamp": timestamp,
                "comparison_num": metadata.get("comparison_num", len(results) + 1)
            }
            results.append(result)
        except Exception as e:
            # Fallback for old format - try to extract locale from text_sample or skip
            continue
    
    # Extract locale and calculate win rates per locale
    locale_stats = {}
    
    for result in results:
        # Use stored locale from metadata, or extract from voice
        locale = result.get("locale", "Unknown")
        if locale == "Unknown":
            winner_voice = result.get("winner_voice", "")
            if winner_voice:
                voice_parts = winner_voice.split("-")
                if len(voice_parts) >= 2:
                    locale = "-".join(voice_parts[:2])
        
        # Initialize locale stats if not exists
        if locale not in locale_stats:
            locale_stats[locale] = {
                "falcon_wins": 0,
                "zeroshot_wins": 0,
                "total": 0
            }
        
        # Count wins
        locale_stats[locale]["total"] += 1
        if result["winner"] == "murf_falcon_oct23":
            locale_stats[locale]["falcon_wins"] += 1
        elif result["winner"] == "murf_zeroshot":
            locale_stats[locale]["zeroshot_wins"] += 1
    
    # Create summary table
    locale_data = []
    for locale, stats in sorted(locale_stats.items()):
        total = stats["total"]
        falcon_wins = stats["falcon_wins"]
        zeroshot_wins = stats["zeroshot_wins"]
        falcon_win_rate = (falcon_wins / total * 100) if total > 0 else 0
        zeroshot_win_rate = (zeroshot_wins / total * 100) if total > 0 else 0
        
        locale_data.append({
            "Locale": locale,
            "Falcon Wins": falcon_wins,
            "Zeroshot Wins": zeroshot_wins,
            "Total Comparisons": total,
            "Falcon Win Rate": f"{falcon_win_rate:.1f}%",
            "Zeroshot Win Rate": f"{zeroshot_win_rate:.1f}%"
        })
    
    if locale_data:
        # Create DataFrame for visualizations
        locale_df = pd.DataFrame(locale_data)
        
        # Prepare data for bar charts
        locales = [item["Locale"] for item in locale_data]
        falcon_wins_list = [item["Falcon Wins"] for item in locale_data]
        zeroshot_wins_list = [item["Zeroshot Wins"] for item in locale_data]
        falcon_win_rates = [float(item["Falcon Win Rate"].replace("%", "")) for item in locale_data]
        zeroshot_win_rates = [float(item["Zeroshot Win Rate"].replace("%", "")) for item in locale_data]
        
        # Prepare data for grouped bar chart with win rates
        wins_chart_data = []
        for i, locale in enumerate(locales):
            wins_chart_data.append({
                "Locale": locale, 
                "Model": "Falcon", 
                "Wins": falcon_wins_list[i],
                "Win Rate": falcon_win_rates[i]
            })
            wins_chart_data.append({
                "Locale": locale, 
                "Model": "Zeroshot", 
                "Wins": zeroshot_wins_list[i],
                "Win Rate": zeroshot_win_rates[i]
            })
        
        wins_chart_df = pd.DataFrame(wins_chart_data)
        
        # Bar chart: Total Wins by Locale with Win Rate labels
        fig_wins = px.bar(
            wins_chart_df,
            x="Locale",
            y="Wins",
            color="Model",
            barmode='group',
            labels={'Wins': 'Number of Wins', 'Locale': 'Locale'},
            title='Total Wins by Locale (Falcon vs Zeroshot)',
            color_discrete_map={"Falcon": '#6642B3', "Zeroshot": '#42B366'},
            text=[f"{row['Wins']} ({row['Win Rate']:.1f}%)" for _, row in wins_chart_df.iterrows()]
        )
        fig_wins.update_traces(textposition='outside')
        fig_wins.update_layout(
            legend=dict(title="Model"),
            xaxis_title="Locale",
            yaxis_title="Number of Wins",
            height=400
        )
        st.plotly_chart(fig_wins, use_container_width=True)
        
        # Data table
        st.markdown("#### Detailed Data")
        st.dataframe(locale_df, use_container_width=True, hide_index=True)
    else:
        st.info("No locale data available")
    
    # Overall summary metrics
    st.markdown("---")
    st.markdown("### Overall Summary")
    falcon_wins = sum(1 for r in results if r["winner"] == "murf_falcon_oct23")
    zeroshot_wins = sum(1 for r in results if r["winner"] == "murf_zeroshot")
    total = len(results)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Murf Falcon Wins", falcon_wins, f"{falcon_wins/total*100:.1f}%")
    with col2:
        st.metric("Murf Zeroshot Wins", zeroshot_wins, f"{zeroshot_wins/total*100:.1f}%")


def ranked_blind_comments_page():
    """Comments persisted with ranked blind-test votes (Gen2 vs NewModel)."""
    st.header("Comments")
    st.markdown("Notes you entered on each blind vote (stored with ELO updates).")
    votes = db.get_ranked_blind_test_votes()
    if not votes:
        st.info("No blind A/B votes in the database yet.")
        return
    rows = []
    for winner, loser, timestamp, metadata_json in votes:
        try:
            meta = json.loads(metadata_json) if metadata_json else {}
            comment = (meta.get("comment") or "").strip()
            if not comment:
                continue
            locale = meta.get("locale")
            wn = get_provider_name_for_de_anonymization(winner)
            ln = get_provider_name_for_de_anonymization(loser)
            text_full = meta.get("text") or ""
            rows.append({
                "Time": timestamp,
                "Locale": locale or "—",
                "Winner": wn,
                "Loser": ln,
                "Winner voice": meta.get("winner_voice", meta.get("winner_config", {}).get("voice", "")),
                "Loser voice": meta.get("loser_voice", meta.get("loser_config", {}).get("voice", "")),
                "Sample text": text_full[:120] + ("…" if len(text_full) > 120 else ""),
                "Comment": comment,
            })
        except Exception:
            continue
    if not rows:
        st.info("Votes exist but no comments yet. Add an optional comment before each vote to see them here.")
        return
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True, height="content")


def fvs_comments_page():
    """Display all comments from Falcon vs Zeroshot tests, grouped by locale, with ChatGPT summaries"""
    st.header("Comments (Falcon vs 0shot)")
    st.markdown("All comments from Falcon vs Zeroshot tests, organized by locale")
    
    # Get results from database
    votes = db.get_fvs_votes()
    
    if not votes:
        st.info("No comments available. Complete a Falcon vs Zeroshot test to see comments.")
        return
    
    # Parse votes and group comments by locale
    locale_comments = {}
    
    for winner, loser, text_sample, timestamp, metadata_json in votes:
        try:
            metadata = json.loads(metadata_json) if metadata_json else {}
            comment = metadata.get("comment", "").strip()
            
            # Skip if no comment
            if not comment:
                continue
            
            # Extract locale from metadata or voice
            locale = metadata.get("locale", "Unknown")
            if locale == "Unknown" and metadata.get("winner_voice"):
                voice_parts = metadata.get("winner_voice", "").split("-")
                if len(voice_parts) >= 2:
                    locale = "-".join(voice_parts[:2])
            
            # Initialize locale if not exists
            if locale not in locale_comments:
                locale_comments[locale] = []
            
            # Add comment with metadata
            locale_comments[locale].append({
                "comment": comment,
                "text": metadata.get("text", text_sample),
                "winner": "Murf Falcon" if winner == "murf_falcon_oct23" else "Murf Zeroshot",
                "loser": "Murf Falcon" if loser == "murf_falcon_oct23" else "Murf Zeroshot",
                "timestamp": timestamp,
                "winner_voice": metadata.get("winner_voice", ""),
                "loser_voice": metadata.get("loser_voice", "")
            })
        except Exception as e:
            continue
    
    if not locale_comments:
        st.info("No comments found in the database.")
        return
    
    # Sort locales alphabetically
    sorted_locales = sorted(locale_comments.keys())
    
    # Initialize selected locale in session state
    if "fvs_selected_locale" not in st.session_state:
        st.session_state.fvs_selected_locale = sorted_locales[0] if sorted_locales else None
    
    # Simple dropdown to select locale
    # Create dropdown options with comment counts
    locale_options = {f"{locale} ({len(locale_comments[locale])})": locale for locale in sorted_locales}
    
    selected_label = None
    for label, locale in locale_options.items():
        if locale == st.session_state.fvs_selected_locale:
            selected_label = label
            break
    
    selected_label = st.selectbox(
        "Select Locale:",
        options=list(locale_options.keys()),
        index=list(locale_options.keys()).index(selected_label) if selected_label else 0,
        key="fvs_locale_dropdown"
    )
    
    selected_locale = locale_options[selected_label]
    st.session_state.fvs_selected_locale = selected_locale
    
    # Display table for selected locale
    if selected_locale and selected_locale in locale_comments:
        comments = locale_comments[selected_locale]
        st.markdown(f"### {selected_locale} - {len(comments)} comment{'s' if len(comments) != 1 else ''}")
        
        # Prepare data for table
        table_data = []
        for comment_idx, comment_data in enumerate(comments, 1):
            timestamp_str = comment_data.get('timestamp', '')
            if timestamp_str:
                if isinstance(timestamp_str, str):
                    formatted_timestamp = timestamp_str
                else:
                    formatted_timestamp = timestamp_str.strftime('%Y-%m-%d %H:%M:%S')
            else:
                formatted_timestamp = '-'
            
            table_data.append({
                "#": comment_idx,
                "Winner": comment_data['winner'],
                "Loser": comment_data['loser'],
                "Winner Voice": comment_data.get('winner_voice', '-'),
                "Loser Voice": comment_data.get('loser_voice', '-'),
                "Text": comment_data['text'][:100] + "..." if len(comment_data['text']) > 100 else comment_data['text'],
                "Comment": comment_data['comment'],
                "Timestamp": formatted_timestamp
            })
        
        # Display as table
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Summary section using ChatGPT
    st.markdown("---")
    
    st.markdown("### Summary of Comments by LLM")
    st.markdown("AI-generated summary of comments for the selected locale (auto-updates when new comments are added)")
    
    # Check if OpenAI is available
    if openai is None:
        st.warning("OpenAI library is not installed. Please install it to use the summary feature.")
        return
    
    # Check for API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.warning("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        return
    
    # Initialize OpenAI client
    try:
        client = openai.OpenAI(api_key=openai_api_key)
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {e}")
        return
    
    # Generate summary for selected locale only
    if selected_locale and selected_locale in locale_comments:
        comments = locale_comments[selected_locale]
        comments_text = [c['comment'] for c in comments]
        
        if not comments_text:
            st.info(f"No comments available for {selected_locale} to summarize.")
        else:
            comment_count = len(comments_text)
            
            # Check if summary exists in database
            stored_summary = db.get_locale_summary(selected_locale)
            should_regenerate = True
            summary = None
            summary_metadata = {}
            
            if stored_summary:
                stored_count = stored_summary["comment_count"]
                # Check if comment count matches (if it changed, regenerate automatically)
                if stored_count == comment_count:
                    summary = stored_summary["summary"]
                    should_regenerate = False
                    summary_metadata = stored_summary
                    st.success(f"Using stored summary ({stored_count} comments analyzed on {stored_summary['updated_at'][:10]})")
                else:
                    # Comment count changed - auto-regenerate
                    st.info(f"New comments detected ({stored_count} → {comment_count}). Regenerating summary...")
                    should_regenerate = True
            
            if should_regenerate:
                # Calculate win/loss statistics for context
                falcon_wins = sum(1 for c in locale_comments[selected_locale] if c.get('winner') == 'Murf Falcon')
                zeroshot_wins = sum(1 for c in locale_comments[selected_locale] if c.get('winner') == 'Murf Zeroshot')
                total_comparisons = len(locale_comments[selected_locale])
                
                # Create prompt for concise, actionable summarization - CRITICAL ISSUES ONLY
                comments_list = "\n".join([f"{idx + 1}. {comment}" for idx, comment in enumerate(comments_text)])
                prompt = f"""Analyze user feedback comparing Murf Falcon vs Murf Zeroshot for locale {selected_locale}.

Context: {total_comparisons} comparisons | Falcon wins: {falcon_wins} | Zeroshot wins: {zeroshot_wins}

Comments:
{comments_list}

Provide a BRIEF summary (max 200 words) focusing ONLY on CRITICAL issues:

1. **Most Critical Issues ({selected_locale})** (ranked by impact, not just frequency):
   - Only include issues that significantly impact quality or user experience
   - Filter out minor nitpicks and edge cases
   - Focus on what Zeroshot needs to improve over Falcon
   - Note frequency only if it's a critical issue (e.g., "Critical: Pronunciation errors (5x)")

2. **Key Action Items ({selected_locale})** (top 3-5 only):
   - What should Zeroshot prioritize improving?
   - Rank by criticality first, then frequency

3. **Overall Pattern ({selected_locale})** (one sentence):
   - Clear preference trend or balanced?

IMPORTANT: Ignore minor issues, nitpicks, and one-off edge cases. Only report issues that meaningfully impact quality. Be extremely selective - quality over quantity. Always include the locale name ({selected_locale}) in section headers."""
                
                # Generate summary with loading indicator
                with st.spinner(f"Generating summary for {selected_locale}..."):
                    try:
                        response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {"role": "system", "content": "You are a critical issue filter. You ONLY report the most critical, impactful issues that significantly affect quality. You ignore minor nitpicks, edge cases, and one-off complaints. Be extremely selective - focus on issues that matter. Keep it brief (max 200 words). Use bullet points."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.2,
                            max_tokens=500
                        )
                        
                        summary = response.choices[0].message.content
                        
                        # Save summary to database
                        db.save_locale_summary(selected_locale, summary, comment_count, "gpt-4o")
                        st.success("Summary generated and saved!")
                        
                    except Exception as e:
                        st.error(f"Error generating summary for {selected_locale}: {e}")
                        summary = None
            
            # Display summary
            if summary:
                st.markdown(summary)
                
                # Download and Refresh buttons
                st.markdown("---")
                col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
                with col2:
                    # Refresh button - white/secondary style
                    refresh_key = f"refresh_summary_{selected_locale}" if selected_locale else "refresh_summary"
                    if st.button("Refresh Summary", key=refresh_key, use_container_width=True, type="secondary"):
                        # Force regeneration by deleting stored summary
                        if selected_locale:
                            db.delete_locale_summary(selected_locale)
                            # Clear any cached state and rerun
                            st.session_state.pop(refresh_key, None)
                            st.rerun()
                
                # Download buttons for different formats
                base_filename = f"summary_{selected_locale.replace(' ', '_').replace('/', '_')}_{datetime.now().strftime('%Y%m%d')}"
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                with col3:
                    # DOC format (formatted text) - structured with locale, model, comments, then summary
                    doc_content = f"""SUMMARY OF COMMENTS

{'='*80}
LOCALE INFORMATION
{'='*80}

Locale: {selected_locale}
Generated: {timestamp}
Comment Count: {comment_count}
Model Used: gpt-4o

{'='*80}
COMMENTS PRESENT FOR THIS LOCALE ({comment_count} total)
{'='*80}

{chr(10).join([f"{idx + 1}. {comment}" for idx, comment in enumerate(comments_text)])}

{'='*80}
AI-GENERATED SUMMARY
{'='*80}

{summary}

{'='*80}
END OF SUMMARY
{'='*80}
"""
                    st.download_button(
                        label="📄 Download DOC",
                        data=doc_content,
                        file_name=f"{base_filename}.doc",
                        mime="application/msword",
                        use_container_width=True,
                        type="secondary"
                    )
                
                with col4:
                    # CSV format
                    csv_output = io.StringIO()
                    csv_writer = csv.writer(csv_output)
                    
                    # Write header
                    csv_writer.writerow(["Field", "Value"])
                    csv_writer.writerow(["Locale", selected_locale])
                    csv_writer.writerow(["Generated", timestamp])
                    csv_writer.writerow(["Comment Count", comment_count])
                    csv_writer.writerow(["Model Used", "gpt-4o"])
                    csv_writer.writerow([])  # Empty row
                    csv_writer.writerow(["Summary"])
                    csv_writer.writerow([summary])
                    csv_writer.writerow([])  # Empty row
                    csv_writer.writerow(["Comment #", "Comment"])
                    
                    # Write comments
                    for idx, comment in enumerate(comments_text, 1):
                        csv_writer.writerow([idx, comment])
                    
                    csv_content = csv_output.getvalue()
                    csv_output.close()
                    
                    st.download_button(
                        label="📊 Download CSV",
                        data=csv_content,
                        file_name=f"{base_filename}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        type="secondary"
                    )

def display_blind_test_2_setup(configured_providers: List[str]):
    """Display the blind test 2 setup page"""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**1. Comparison**")
        st.caption("Head-to-head: Murf Gen 2 (streaming API, model GEN2) vs your NewModel `/tts` deployment.")
        pool = []
        if "murf_gen2" in configured_providers:
            pool.append("murf_gen2")
        if "omni_tts" in configured_providers:
            pool.append("omni_tts")
        st.session_state.blind_test_2_selected_competitors = pool
        if len(pool) < 2:
            st.warning(
                "Configure **both** providers: `MURF_API_KEY`, `OMNI_API_KEY`, and `OMNI_HOST` or `OMNI_BASE_URL`. "
                "Voice mapping is built into the app (Murf voiceId → NewModel `source_voice_id`)."
            )
        else:
            st.success(f"**{TTS_PROVIDERS['murf_gen2'].name}** vs **{TTS_PROVIDERS['omni_tts'].name}**")
    
    with col2:
        st.markdown("**2. Test parameters**")
        max_comparisons = st.slider(
            "Number of comparisons:",
            min_value=5,
            max_value=50,
            value=25,
            step=5,
            help="How many head-to-head comparisons to run",
            key="max_comparisons_2"
        )
        st.session_state.blind_test_2_max_comparisons = max_comparisons
        
        st.markdown("**3. Voice gender**")
        if "blind_test_2_gender_filter" not in st.session_state:
            st.session_state.blind_test_2_gender_filter = "female"
        
        selected_gender = st.radio(
            "Gender for both systems:",
            ["Male", "Female"],
            index=0 if st.session_state.blind_test_2_gender_filter == "male" else 1,
            horizontal=True,
            key="gender_radio_setup_2"
        )
        st.session_state.blind_test_2_gender_filter = selected_gender.lower()
        
        st.markdown("**4. Locale / language**")
        if "blind_test_2_locale_filter" not in st.session_state:
            st.session_state.blind_test_2_locale_filter = "US"
        
        locale_labels = {
            "US": "US English (en-US)",
            "IN": "English India (en-IN)",
            "UK": "UK English (en-UK)",
            "HI": "Hindi (hi-IN)",
            "BN": "Bangla (bn-IN)",
            "TA": "Tamil (ta-IN)",
        }
        locale_order = ["US", "IN", "UK", "HI", "BN", "TA"]
        current = st.session_state.blind_test_2_locale_filter
        if current not in locale_order:
            current = "US"
        selected_locale_display = st.selectbox(
            "Locale (voices must exist for Murf Gen2 + NewModel list):",
            [locale_labels[k] for k in locale_order],
            index=locale_order.index(current),
            key="locale_select_setup_2"
        )
        rev = {v: k for k, v in locale_labels.items()}
        st.session_state.blind_test_2_locale_filter = rev[selected_locale_display]
        st.caption("Voices are filtered by **gender + locale** on the shared Murf/NewModel list (same `voiceId` per round for en-US, en-IN, en-UK, hi-IN, bn-IN, ta-IN).")
    
    st.divider()
    
    # Sentence upload section - full width
    st.markdown("**5. Upload test sentences** (one random line per round)")
    
    _vb_loc = st.session_state.blind_test_2_locale_filter
    _stamp = "_voice_battle_bundle_text_locale"
    _TEXT_KEY = "blind_test_2_sentence_draft"
    _BACK_KEY = "blind_test_2_sentence_draft_backup"

    # One-time migration from older widget-only key (lost when leaving this page).
    if _TEXT_KEY not in st.session_state and "sentences_text_2" in st.session_state:
        legacy = str(st.session_state.pop("sentences_text_2", "") or "")
        if legacy.strip():
            st.session_state[_TEXT_KEY] = legacy
            st.session_state[_BACK_KEY] = legacy
            if _stamp not in st.session_state:
                st.session_state[_stamp] = _vb_loc

    if st.session_state.get(_stamp) != _vb_loc:
        bun = bundled_default_sentences_voice_battle(_vb_loc)
        st.session_state[_TEXT_KEY] = bun
        st.session_state[_BACK_KEY] = bun
        st.session_state[_stamp] = _vb_loc
    elif _TEXT_KEY not in st.session_state:
        if _BACK_KEY in st.session_state and str(st.session_state.get(_BACK_KEY, "")).strip():
            st.session_state[_TEXT_KEY] = st.session_state[_BACK_KEY]
        else:
            bun = bundled_default_sentences_voice_battle(_vb_loc)
            st.session_state[_TEXT_KEY] = bun
            st.session_state[_BACK_KEY] = bun

    sentences_text = st.text_area(
        "Enter sentences (one per line):",
        height=200,
        help="Enter multiple sentences, one per line. The system will randomly select sentences for each test.",
        key=_TEXT_KEY,
    )
    st.session_state[_BACK_KEY] = sentences_text
    
    sentences = [s.strip() for s in sentences_text.strip().split('\n') if s.strip()]
    
    # Check if sentences have changed - if so, clear current pair to force regeneration
    if "blind_test_2_sentences_hash" not in st.session_state:
        st.session_state.blind_test_2_sentences_hash = None
    
    import hashlib
    sentences_hash = hashlib.md5(str(sorted(sentences)).encode()).hexdigest()
    
    # If sentences changed and there's a current pair, clear it
    if (st.session_state.blind_test_2_sentences_hash is not None and 
        st.session_state.blind_test_2_sentences_hash != sentences_hash and
        st.session_state.blind_test_2_current_pair is not None):
        st.session_state.blind_test_2_current_pair = None
        st.session_state.blind_test_2_comparison_count = 0
        st.session_state.blind_test_2_results_history = []
    
    st.session_state.blind_test_2_sentences = sentences
    st.session_state.blind_test_2_sentences_hash = sentences_hash
    st.caption(f"📝 {len(sentences)} sentences loaded")
    
    can_start = (
        len(st.session_state.get("blind_test_2_selected_competitors", [])) >= 2 and 
        len(st.session_state.blind_test_2_sentences) >= 1
    )
    
    # Center the button and make it smaller
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Start Voice Battle", type="primary", disabled=not can_start, key="start_battle_2"):
            # Use ALL selected competitors (not just 2)
            selected_pool = st.session_state.blind_test_2_selected_competitors
            if len(selected_pool) >= 2:
                import itertools
                import random
                
                # Store ALL selected competitors (not just 2)
                st.session_state.blind_test_2_final_competitors = selected_pool
                
                # Generate all pairwise combinations
                all_pairs = list(itertools.combinations(selected_pool, 2))
                # Shuffle the pairs for variety
                random.shuffle(all_pairs)
                
                # Store the pairs list for cycling through
                st.session_state.blind_test_2_comparison_pairs = all_pairs
                st.session_state.blind_test_2_pair_index = 0
                
                # Show which models were selected
                comp_names = [TTS_PROVIDERS[c].name for c in selected_pool]
                num_pairs = len(all_pairs)
                st.info(f"🎯 {len(selected_pool)} models selected: {', '.join(comp_names)}. Will test {num_pairs} pairwise comparisons.")
            
            st.session_state.blind_test_2_setup_complete = True
            st.session_state.blind_test_2_comparison_count = 0
            st.session_state.blind_test_2_results_history = []
            st.session_state.blind_test_2_current_pair = None  # Clear any existing pair
            st.rerun()
    
    if not can_start:
        pass

def display_blind_test_setup(configured_providers: List[str]):
    """Display the blind test setup page"""
    from config import get_voices_by_gender, get_voice_gender, get_voices_by_gender_and_locale
    import random
    
    # Get Murf providers and other providers
    murf_providers = [p for p in configured_providers if "murf" in p.lower()]
    other_providers = [p for p in configured_providers if "murf" not in p.lower()]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**1. Select Competitor**")
        
        # Create dropdown options for competitors (including Murf providers)
        competitor_options = []
        competitor_ids = []
        
        # Add first non-Murf provider (if any)
        if other_providers:
            first_provider_id = other_providers[0]
            provider_name = TTS_PROVIDERS[first_provider_id].name
            model_name = TTS_PROVIDERS[first_provider_id].model_name
            competitor_options.append(f"{provider_name} ({model_name})")
            competitor_ids.append(first_provider_id)
        
        # Add Murf providers (Falcon and Zeroshot) as 2nd and 3rd options
        for provider_id in murf_providers:
            provider_name = TTS_PROVIDERS[provider_id].name
            model_name = TTS_PROVIDERS[provider_id].model_name
            # Show "Murf Falcon" instead of just "Murf"
            if provider_id == "murf_falcon_oct23":
                display_name = "Murf Falcon"
            elif provider_id == "murf_zeroshot":
                display_name = "Murf Zeroshot"
            else:
                display_name = provider_name
            competitor_options.append(f"{display_name} ({model_name})")
            competitor_ids.append(provider_id)
        
        # Add remaining non-Murf providers (skip first one already added)
        for provider_id in other_providers[1:]:
            provider_name = TTS_PROVIDERS[provider_id].name
            model_name = TTS_PROVIDERS[provider_id].model_name
            competitor_options.append(f"{provider_name} ({model_name})")
            competitor_ids.append(provider_id)
        
        if competitor_options:
            selected_competitor_display = st.selectbox(
                "Test Murf against:",
                competitor_options,
                key="competitor_select"
            )
            
            # Map back to provider ID
            selected_idx = competitor_options.index(selected_competitor_display)
            selected_competitor_id = competitor_ids[selected_idx]
            st.session_state.blind_test_selected_competitors = [selected_competitor_id]
            
            # Show competitor voice info
            competitor_voice_info = TTS_PROVIDERS[selected_competitor_id].voice_info
            female_count = sum(1 for v in competitor_voice_info.values() if v.gender == "female")
            male_count = sum(1 for v in competitor_voice_info.values() if v.gender == "male")
            st.caption(f"Available voices: {female_count} female, {male_count} male")
        else:
            st.warning("No competitors configured")
        
        # Test Parameters below competitor dropdown (uses empty space)
        st.markdown("**Test Parameters**")
        max_comparisons = st.slider(
            "Number of comparisons:",
            min_value=5,
            max_value=50,
            value=25,
            step=5,
            help="How many head-to-head comparisons to run"
        )
        st.session_state.blind_test_max_comparisons = max_comparisons
    
    with col2:
        st.markdown("**2. Select Murf Voice**")
        
        # Murf provider dropdown
        if murf_providers:
            murf_options = []
            murf_ids = []
            for p in murf_providers:
                murf_options.append(f"{TTS_PROVIDERS[p].name} ({TTS_PROVIDERS[p].model_name})")
                murf_ids.append(p)
            
            selected_murf_display = st.selectbox(
                "Provider:",
                murf_options,
                key="murf_provider_select"
            )
            
            selected_murf_idx = murf_options.index(selected_murf_display)
            murf_provider = murf_ids[selected_murf_idx]
            st.session_state.blind_test_murf_provider = murf_provider
            
            murf_voice_info = TTS_PROVIDERS[murf_provider].voice_info
            
            # Initialize gender filter if not set
            if "blind_test_gender_filter" not in st.session_state:
                st.session_state.blind_test_gender_filter = "female"
            
            # Gender selection - text labels side by side using radio
            selected_gender_radio = st.radio(
                "**Gender:**",
                ["Male", "Female"],
                index=0 if st.session_state.blind_test_gender_filter == "female" else 1,
                horizontal=True,
                key="gender_radio"
            )
            
            new_gender = selected_gender_radio.lower()
            selected_gender = st.session_state.blind_test_gender_filter
            
            # Only update if gender actually changed (not on every rerun)
            if new_gender != selected_gender:
                st.session_state.blind_test_gender_filter = new_gender
                # Reset voice to first voice of new gender only when gender changes
                filtered_voices_new = [(v, info) for v, info in murf_voice_info.items() if info.gender == new_gender]
                if filtered_voices_new:
                    st.session_state.blind_test_murf_voice = filtered_voices_new[0][0]
                selected_gender = new_gender
            
            # Filter voices by selected gender only (show ALL locales in voice selector)
            # Locale filtering will only apply during comparison generation, not in the selector
            filtered_voices = [(v, info) for v, info in murf_voice_info.items() if info.gender == selected_gender]
            
            # Initialize voice if not set or if current voice is not valid for selected gender
            if "blind_test_murf_voice" not in st.session_state:
                if filtered_voices:
                    st.session_state.blind_test_murf_voice = filtered_voices[0][0]
            elif st.session_state.blind_test_murf_voice not in [v for v, _ in filtered_voices]:
                # Current voice doesn't match gender/locale, reset to first voice of current gender/locale
                if filtered_voices:
                    st.session_state.blind_test_murf_voice = filtered_voices[0][0]
            
            # Voice multiselect (up to 4 voices)
            voice_options = [f"{info.name} ({info.accent})" for v, info in filtered_voices]
            voice_ids = [v for v, info in filtered_voices]
            
            if voice_options:
                # Initialize selected voices if not set
                if "blind_test_murf_voices" not in st.session_state or not st.session_state.blind_test_murf_voices:
                    # Default to first voice if none selected
                    st.session_state.blind_test_murf_voices = [voice_ids[0]] if voice_ids else []
                
                # Find currently selected voice indices for multiselect
                current_selected_indices = []
                for voice_id in st.session_state.blind_test_murf_voices:
                    if voice_id in voice_ids:
                        current_selected_indices.append(voice_ids.index(voice_id))
                
                # Use multiselect to allow selecting up to 4 voices
                selected_voice_displays = st.multiselect(
                    "Select MURF voices (up to 4, will shuffle during comparisons):",
                    voice_options,
                    default=[voice_options[i] for i in current_selected_indices if i < len(voice_options)],
                    max_selections=4,
                    key="murf_voices_multiselect"
                )
                
                # Update session state with selected voice IDs
                selected_voice_ids = []
                for display in selected_voice_displays:
                    idx = voice_options.index(display)
                    selected_voice_ids.append(voice_ids[idx])
                
                st.session_state.blind_test_murf_voices = selected_voice_ids
                
                # Keep backward compatibility: set single voice to first selected for legacy code
                if selected_voice_ids:
                    st.session_state.blind_test_murf_voice = selected_voice_ids[0]
                else:
                    # If no voices selected, default to first voice
                    if voice_ids:
                        st.session_state.blind_test_murf_voices = [voice_ids[0]]
                        st.session_state.blind_test_murf_voice = voice_ids[0]
            
            # Show selected voices info
            if st.session_state.blind_test_murf_voices:
                selected_names = []
                for voice_id in st.session_state.blind_test_murf_voices:
                    voice_info = murf_voice_info.get(voice_id)
                    if voice_info:
                        selected_names.append(voice_info.name)
                if selected_names:
                    st.caption(f"Selected: **{', '.join(selected_names)}** • Will shuffle during comparisons and compare with {selected_gender} voices")
        else:
            st.warning("No Murf provider configured")
    
    st.divider()
    
    # Sentence upload section - full width
    st.markdown("**3. Upload Test Sentences** (System will pick randomly)")
    
    sentences_text = st.text_area(
        "Enter sentences (one per line):",
        value="""The quick brown fox jumps over the lazy dog.
The wine glass fills again and laughter breaks through the pressure that had been building quietly for hours.
Just to confirm, the co-applicant's name is spelled M-A-R-I-S-A, correct?
Scientists have made a groundbreaking discovery that could revolutionize renewable energy.
Hello, how can I assist you today with your account inquiry?""",
        height=200,
        help="Enter multiple sentences, one per line. The system will randomly select sentences for each test."
    )
    
    sentences = [s.strip() for s in sentences_text.strip().split('\n') if s.strip()]
    
    # Check if sentences have changed - if so, clear current pair to force regeneration
    if "blind_test_sentences_hash" not in st.session_state:
        st.session_state.blind_test_sentences_hash = None
    
    import hashlib
    sentences_hash = hashlib.md5(str(sorted(sentences)).encode()).hexdigest()
    
    # If sentences changed and there's a current pair, clear it
    if (st.session_state.blind_test_sentences_hash is not None and 
        st.session_state.blind_test_sentences_hash != sentences_hash and
        st.session_state.blind_test_current_pair is not None):
        st.session_state.blind_test_current_pair = None
        st.session_state.blind_test_comparison_count = 0
        st.session_state.blind_test_results_history = []
    
    st.session_state.blind_test_sentences = sentences
    st.session_state.blind_test_sentences_hash = sentences_hash
    st.caption(f"📝 {len(sentences)} sentences loaded")
    
    can_start = (
        len(st.session_state.get("blind_test_selected_competitors", [])) >= 1 and 
        (st.session_state.blind_test_murf_voice or 
         (st.session_state.blind_test_murf_voices and len(st.session_state.blind_test_murf_voices) > 0)) and 
        len(st.session_state.blind_test_sentences) >= 1
    )
    
    # Center the button and make it smaller
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Start Voice Battle", type="primary", disabled=not can_start):
            st.session_state.blind_test_setup_complete = True
            st.session_state.blind_test_comparison_count = 0
            st.session_state.blind_test_results_history = []
            st.session_state.blind_test_current_pair = None  # Clear any existing pair
            st.rerun()
    
    if not can_start:
        st.caption("Select a competitor and ensure sentences are loaded")

def display_blind_test_2_comparison():
    """Display the active blind test 2 comparison"""
    from config import get_voices_by_gender, get_voice_gender
    import random
    
    # Check if we should show final results (either completed or user clicked End Test)
    if st.session_state.get("show_final_results_2", False):
        # Clear the comparison view first to ensure full-width display
        st.session_state.blind_test_2_current_pair = None
        # Display full-width final results
        display_final_results_2()
        return
    
    # Check if we need to generate a new pair
    force_regen = st.session_state.get("force_regenerate_2", False)
    if st.session_state.blind_test_2_current_pair is None or force_regen:
        # Clear the force flag
        st.session_state.force_regenerate_2 = False
        # Force clear any cached audio state before generating new comparison
        if "blind_test_2_audio_played" in st.session_state:
            st.session_state.blind_test_2_audio_played = {"A": 0, "B": 0}
        # CRITICAL: Clear the pair again to ensure no stale data
        st.session_state.blind_test_2_current_pair = None
        print(f"[DEBUG] Generating new comparison (force_regen={force_regen})")
        generate_next_comparison_2()
        return
    
    pair = st.session_state.blind_test_2_current_pair
    
    # Validate that the current pair's text is still in the sentences list
    # If sentences were changed, the pair might have old text - regenerate it
    if pair and pair.get("text"):
        current_text = pair.get("text")
        sentences = st.session_state.get("blind_test_2_sentences", [])
        if sentences and current_text not in sentences:
            # Current pair has text that's no longer in sentences - regenerate
            st.session_state.blind_test_2_current_pair = None
            generate_next_comparison_2()
            return
        
        # Additional validation: Check if this pair was generated with a different comparison count
        # This ensures we don't show stale audio from a previous comparison
        pair_comparison_id = pair.get("comparison_id", "")
        expected_comparison_id = f"{st.session_state.blind_test_2_comparison_count}_"
        if pair_comparison_id and not pair_comparison_id.startswith(expected_comparison_id):
            # Pair is from a different comparison - regenerate
            st.session_state.blind_test_2_current_pair = None
            generate_next_comparison_2()
            return
    
    if pair is None or pair.get("error"):
        error_msg = pair.get("message", "Failed to generate comparison.") if pair else "Failed to generate comparison."
        st.error(f"⚠️ {error_msg}")
        if st.button("Retry", type="primary", key="retry_2"):
            st.session_state.blind_test_2_current_pair = None
            st.rerun()
        return
    
    # Create unique key for this comparison using generation timestamp
    generated_at = pair.get("generated_at", 0)
    comparison_key = f"{st.session_state.blind_test_2_comparison_count}_{int(generated_at)}"
    
    # CRITICAL DEBUG: Log what we're about to display
    print(f"[DISPLAY DEBUG] Displaying pair:")
    print(f"  - Text: '{pair['text'][:60]}...'")
    print(f"  - Comparison key: {comparison_key}")
    print(f"  - Generated at: {generated_at}")
    if pair.get('sample_a') and hasattr(pair['sample_a'], 'text'):
        print(f"  - Sample A text: '{pair['sample_a'].text[:60] if pair['sample_a'].text else 'N/A'}...'")
    if pair.get('sample_b') and hasattr(pair['sample_b'], 'text'):
        print(f"  - Sample B text: '{pair['sample_b'].text[:60] if pair['sample_b'].text else 'N/A'}...'")
    
    # Progress indicator
    progress = st.session_state.blind_test_2_comparison_count / st.session_state.blind_test_2_max_comparisons
    st.progress(progress)
    st.caption(f"Comparison {st.session_state.blind_test_2_comparison_count + 1} of {st.session_state.blind_test_2_max_comparisons}")
    
    # Display the prompt/sentence - sleek gray design (no purple border)
    st.markdown(f"""
    <div style="background: #f5f5f5; padding: 12px 16px; border-radius: 8px; margin: 8px 0;">
        <span style="color: #666; font-size: 0.85em; font-weight: 500;">PROMPT</span>
        <p style="color: #333; font-size: 1em; margin: 4px 0 0 0; line-height: 1.5;">{pair['text']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='color: #888; font-size: 0.9em; margin: 16px 0 8px 0;'>Vote to reveal your model preference</p>", unsafe_allow_html=True)
    
    # Audio players side by side with unique keys
    col1, col2 = st.columns(2)
    
    with col1:
        display_audio_player(pair['sample_a'], "A", "left", comparison_key)
        # Add spacing and center the button
        st.markdown('<div style="margin-top: 16px;"></div>', unsafe_allow_html=True)
        button_col1, button_col2, button_col3 = st.columns([1, 2, 1])
        with button_col2:
            if st.button("Vote A", type="primary", key="vote_a_2", use_container_width=True):
                handle_vote_2("A", pair)
    
    with col2:
        # Validate audio text matches displayed text before displaying
        sample_b = pair['sample_b']
        if sample_b and hasattr(sample_b, 'metadata') and sample_b.metadata:
            audio_text = sample_b.metadata.get('generated_text', '')
            if audio_text and audio_text != pair['text']:
                st.error(f"⚠️ Audio text mismatch detected! Regenerating...")
                st.session_state.blind_test_2_current_pair = None
                st.rerun()
                return
        display_audio_player(pair['sample_b'], "B", "right", comparison_key)
        # Add spacing and center the button
        st.markdown('<div style="margin-top: 16px;"></div>', unsafe_allow_html=True)
        button_col1, button_col2, button_col3 = st.columns([1, 2, 1])
        with button_col2:
            if st.button("Vote B", type="primary", key="vote_b_2", use_container_width=True):
                handle_vote_2("B", pair)
    
    st.divider()
    
    # Comment input field (optional)
    comment_key = f"comment_2_{st.session_state.blind_test_2_comparison_count}"
    if comment_key not in st.session_state:
        st.session_state[comment_key] = ""
    
    comment = st.text_area(
        "Add a comment (optional)",
        value=st.session_state.get(comment_key, ""),
        key=comment_key,
        placeholder="e.g., Pronunciation issue on 'A', unnatural pause, quality difference...",
        height=80,
        help="Add notes about this comparison to help identify actionable feedback"
    )
    
    # Action button - End Test only (centered, medium size)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("End Test", use_container_width=True, type="secondary", key="end_test_2"):
            # Set flag to show final results, but keep setup_complete True so we stay in comparison view
            st.session_state.show_final_results_2 = True
            st.session_state.blind_test_2_current_pair = None
            st.rerun()
            return


def display_blind_test_comparison():
    """Display the active blind test comparison"""
    from config import get_voices_by_gender, get_voice_gender
    import random
    
    # Check if we should show final results (either completed or user clicked End Test)
    if st.session_state.get("show_final_results", False):
        # Clear the comparison view first to ensure full-width display
        st.session_state.blind_test_current_pair = None
        # Display full-width final results
        display_final_results()
        return
    
    # Check if we need to generate a new pair
    force_regen = st.session_state.get("force_regenerate", False)
    if st.session_state.blind_test_current_pair is None or force_regen:
        # Clear the force flag
        st.session_state.force_regenerate = False
        # Force clear any cached audio state before generating new comparison
        if "blind_test_audio_played" in st.session_state:
            st.session_state.blind_test_audio_played = {"A": 0, "B": 0}
        # CRITICAL: Clear the pair again to ensure no stale data
        st.session_state.blind_test_current_pair = None
        print(f"[DEBUG] Generating new comparison (force_regen={force_regen})")
        generate_next_comparison()
        return
    
    pair = st.session_state.blind_test_current_pair
    
    # Validate that the current pair's text is still in the sentences list
    # If sentences were changed, the pair might have old text - regenerate it
    if pair and pair.get("text"):
        current_text = pair.get("text")
        sentences = st.session_state.get("blind_test_sentences", [])
        if sentences and current_text not in sentences:
            # Current pair has text that's no longer in sentences - regenerate
            st.session_state.blind_test_current_pair = None
            generate_next_comparison()
            return
        
        # Additional validation: Check if this pair was generated with a different comparison count
        # This ensures we don't show stale audio from a previous comparison
        pair_comparison_id = pair.get("comparison_id", "")
        expected_comparison_id = f"{st.session_state.blind_test_comparison_count}_"
        if pair_comparison_id and not pair_comparison_id.startswith(expected_comparison_id):
            # Pair is from a different comparison - regenerate
            st.session_state.blind_test_current_pair = None
            generate_next_comparison()
            return
    
    if pair is None or pair.get("error"):
        error_msg = pair.get("message", "Failed to generate comparison.") if pair else "Failed to generate comparison."
        st.error(f"⚠️ {error_msg}")
        if st.button("Retry", type="primary"):
            st.session_state.blind_test_current_pair = None
            st.rerun()
        return
    
    # Create unique key for this comparison using generation timestamp
    generated_at = pair.get("generated_at", 0)
    comparison_key = f"{st.session_state.blind_test_comparison_count}_{int(generated_at)}"
    
    # CRITICAL DEBUG: Log what we're about to display
    print(f"[DISPLAY DEBUG] Displaying pair:")
    print(f"  - Text: '{pair['text'][:60]}...'")
    print(f"  - Comparison key: {comparison_key}")
    print(f"  - Generated at: {generated_at}")
    if pair.get('sample_a') and hasattr(pair['sample_a'], 'text'):
        print(f"  - Sample A text: '{pair['sample_a'].text[:60] if pair['sample_a'].text else 'N/A'}...'")
    if pair.get('sample_b') and hasattr(pair['sample_b'], 'text'):
        print(f"  - Sample B text: '{pair['sample_b'].text[:60] if pair['sample_b'].text else 'N/A'}...'")
    
    # Progress indicator
    progress = st.session_state.blind_test_comparison_count / st.session_state.blind_test_max_comparisons
    st.progress(progress)
    st.caption(f"Comparison {st.session_state.blind_test_comparison_count + 1} of {st.session_state.blind_test_max_comparisons}")
    
    # Display the prompt/sentence - sleek gray design (no purple border)
    st.markdown(f"""
    <div style="background: #f5f5f5; padding: 12px 16px; border-radius: 8px; margin: 8px 0;">
        <span style="color: #666; font-size: 0.85em; font-weight: 500;">PROMPT</span>
        <p style="color: #333; font-size: 1em; margin: 4px 0 0 0; line-height: 1.5;">{pair['text']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='color: #888; font-size: 0.9em; margin: 16px 0 8px 0;'>Vote to reveal your model preference</p>", unsafe_allow_html=True)
    
    # Audio players side by side with unique keys
    col1, col2 = st.columns(2)
    
    with col1:
        display_audio_player(pair['sample_a'], "A", "left", comparison_key)
        # Add spacing and center the button
        st.markdown('<div style="margin-top: 16px;"></div>', unsafe_allow_html=True)
        button_col1, button_col2, button_col3 = st.columns([1, 2, 1])
        with button_col2:
            if st.button("Vote A", type="primary", key="vote_a", use_container_width=True):
                handle_vote("A", pair)
    
    with col2:
        # Validate audio text matches displayed text before displaying
        sample_b = pair['sample_b']
        if sample_b and hasattr(sample_b, 'metadata') and sample_b.metadata:
            audio_text = sample_b.metadata.get('generated_text', '')
            if audio_text and audio_text != pair['text']:
                st.error(f"⚠️ Audio text mismatch detected! Regenerating...")
                st.session_state.blind_test_current_pair = None
                st.rerun()
                return
        display_audio_player(pair['sample_b'], "B", "right", comparison_key)
        # Add spacing and center the button
        st.markdown('<div style="margin-top: 16px;"></div>', unsafe_allow_html=True)
        button_col1, button_col2, button_col3 = st.columns([1, 2, 1])
        with button_col2:
            if st.button("Vote B", type="primary", key="vote_b", use_container_width=True):
                handle_vote("B", pair)
    
    st.divider()
    
    # Comment input field (optional)
    comment_key = f"comment_{st.session_state.blind_test_comparison_count}"
    if comment_key not in st.session_state:
        st.session_state[comment_key] = ""
    
    comment = st.text_area(
        "Add a comment (optional)",
        value=st.session_state.get(comment_key, ""),
        key=comment_key,
        placeholder="e.g., Pronunciation issue on 'X', unnatural pause, quality difference...",
        height=80,
        help="Add notes about this comparison to help identify actionable feedback"
    )
    
    # Action button - End Test only (centered, medium size)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("End Test", use_container_width=True, type="secondary"):
            # Set flag to show final results, but keep setup_complete True so we stay in comparison view
            st.session_state.show_final_results = True
            st.session_state.blind_test_current_pair = None
            st.rerun()
            return


def display_audio_player(result, label: str, side: str, unique_key: str = ""):
    """Display an audio player using Streamlit's native audio component"""
    import time
    
    has_audio = (result and 
                 hasattr(result, 'audio_data') and 
                 result.audio_data and 
                 hasattr(result, 'success') and 
                 result.success)
    
    if not has_audio:
        error_msg = ""
        if result and hasattr(result, 'error_message') and result.error_message:
            error_msg = f": {result.error_message}"
        st.error(f"Sample {label} failed to generate{error_msg}")
        return
    
    # CRITICAL DEBUG: Log audio data info
    audio_size = len(result.audio_data) if result.audio_data else 0
    text_in_result = result.text if hasattr(result, 'text') else 'N/A'
    print(f"[AUDIO DEBUG] Sample {label}: size={audio_size} bytes, text='{text_in_result[:50] if text_in_result != 'N/A' else 'N/A'}...'")
    
    # Display label
    st.markdown(f"**Sample {label}**")
    
    fmt = "audio/mp3"
    if getattr(result, "metadata", None) and result.metadata.get("format") == "wav":
        fmt = "audio/wav"
    
    # Use Streamlit's native audio component - this avoids browser caching issues
    st.audio(result.audio_data, format=fmt)


def generate_next_comparison_2():
    """Generate the next comparison pair for Blind Test 2 - Randomly selects 2 competitors from selected pool"""
    from config import get_voices_by_gender, get_voice_gender
    import random
    import time
    
    print(f"[GENERATE DEBUG] Starting generate_next_comparison_2 for comparison #{st.session_state.blind_test_2_comparison_count}")
    
    # Check if we've reached max comparisons
    if st.session_state.blind_test_2_comparison_count >= st.session_state.blind_test_2_max_comparisons:
        st.session_state.show_final_results_2 = True
        st.rerun()
        return
    
    # Get gender from setup (selected once at start)
    gender_filter = st.session_state.blind_test_2_gender_filter
    comparison_index = st.session_state.blind_test_2_comparison_count
    
    # FORCE CLEAR any existing pair first to prevent stale audio
    st.session_state.blind_test_2_current_pair = None
    
    # Generate a unique generation ID for this comparison
    generation_id = f"gen_{int(time.time() * 1000)}_{st.session_state.blind_test_2_comparison_count}"
    print(f"[GENERATE DEBUG] Generation ID: {generation_id}")
    
    # Get a random sentence
    sentences = st.session_state.blind_test_2_sentences
    if not sentences:
        st.error("No sentences available")
        return
    
    # CRITICAL: Track which sentences have been used to ensure variety
    if "used_sentences_2" not in st.session_state:
        st.session_state.used_sentences_2 = []
    
    # Get available sentences (ones not used yet, or all if all have been used)
    available_sentences = [s for s in sentences if s not in st.session_state.used_sentences_2]
    
    # If all sentences have been used, reset and start fresh
    if not available_sentences:
        st.session_state.used_sentences_2 = []
        available_sentences = sentences
    
    # Select a random sentence from available ones
    text = random.choice(available_sentences)
    
    # Mark this sentence as used
    st.session_state.used_sentences_2.append(text)
    
    # CRITICAL DEBUG: Log the selected text
    print(f"[CRITICAL DEBUG] Selected sentence #{len(st.session_state.used_sentences_2)}: '{text[:80]}...'")
    print(f"[CRITICAL DEBUG] Available sentences: {len(available_sentences)}, Total: {len(sentences)}")
    
    # Get the pre-generated comparison pairs
    comparison_pairs = st.session_state.get("blind_test_2_comparison_pairs")
    pair_index = st.session_state.get("blind_test_2_pair_index", 0)
    
    if not comparison_pairs or len(comparison_pairs) == 0:
        st.error("Comparison pairs not generated. Please restart the test.")
        return
    
    # Cycle through pairs - if we've used all pairs, shuffle and restart
    if pair_index >= len(comparison_pairs):
        random.shuffle(comparison_pairs)
        pair_index = 0
        st.session_state.blind_test_2_pair_index = 0
    
    # Get the current pair
    competitor_a_id, competitor_b_id = comparison_pairs[pair_index]
    
    # Update pair index for next comparison
    st.session_state.blind_test_2_pair_index = pair_index + 1
    
    print(f"[COMPETITOR SELECTION] Using pair {pair_index + 1}/{len(comparison_pairs)}: {competitor_a_id} vs {competitor_b_id} (Comparison #{comparison_index + 1})")
    
    # Get locale filter from setup
    locale_filter = st.session_state.get("blind_test_2_locale_filter", "US")
    
    # Murf + NewModel share the same Murf voiceId catalog — locale must filter NewModel too (old code only filtered Murf).
    apply_locale_a = ("murf" in competitor_a_id.lower()) or competitor_a_id == "omni_tts"
    apply_locale_b = ("murf" in competitor_b_id.lower()) or competitor_b_id == "omni_tts"
    
    # Get voices for competitor A matching gender AND locale when this provider uses locale in blind test
    from config import get_voices_by_gender_and_locale, get_voices_by_gender, voice_matches_blind_locale
    if apply_locale_a:
        competitor_a_voices = get_voices_by_gender_and_locale(competitor_a_id, gender_filter, locale_filter)
    else:
        competitor_a_voices = get_voices_by_gender(competitor_a_id, gender_filter)
    if competitor_a_id in TTS_PROVIDERS:
        supported_voices_set = set(TTS_PROVIDERS[competitor_a_id].supported_voices)
        competitor_a_voices = [v for v in competitor_a_voices if v in supported_voices_set]
    
    # CRITICAL: For Cartesia, ALWAYS get fresh list and verify it's in voice_id_map
    if competitor_a_id in ["cartesia_sonic2", "cartesia_turbo", "cartesia_sonic3"]:
        try:
            provider_obj = TTSProviderFactory.create_provider(competitor_a_id)
            if hasattr(provider_obj, 'voice_id_map'):
                # Get completely fresh gender-filtered list (locale only if Murf)
                if apply_locale_a:
                    fresh_a = get_voices_by_gender_and_locale(competitor_a_id, gender_filter, locale_filter)
                else:
                    fresh_a = get_voices_by_gender(competitor_a_id, gender_filter)
                if competitor_a_id in TTS_PROVIDERS:
                    supported_set = set(TTS_PROVIDERS[competitor_a_id].supported_voices)
                    # Only use voices that are: 1) in voice_id_map, 2) supported, 3) correct gender
                    competitor_a_voices = [
                        v for v in fresh_a 
                        if v in supported_set and 
                        v in provider_obj.voice_id_map and
                        TTS_PROVIDERS[competitor_a_id].voice_info.get(v) and
                        TTS_PROVIDERS[competitor_a_id].voice_info[v].gender == gender_filter
                    ]
                    # Additional locale check only if Murf
                    if apply_locale_a:
                        _vi = TTS_PROVIDERS[competitor_a_id].voice_info
                        competitor_a_voices = [
                            v for v in competitor_a_voices
                            if _vi.get(v) and voice_matches_blind_locale(v, _vi[v], locale_filter)
                        ]
                    print(f"[CARTESIA PRE-FILTER] Competitor A: Filtered to {len(competitor_a_voices)} {gender_filter} {' ' + locale_filter if apply_locale_a else ''} voices")
        except Exception as e:
            print(f"Warning: Could not pre-filter Cartesia voices for A: {e}")
    
    # If no voices found, try from voice_info
    if not competitor_a_voices and competitor_a_id in TTS_PROVIDERS:
        competitor_a_voice_info = TTS_PROVIDERS[competitor_a_id].voice_info
        supported_voices_set = set(TTS_PROVIDERS[competitor_a_id].supported_voices)
        if apply_locale_a:
            competitor_a_voices = [
                v for v, info in competitor_a_voice_info.items() 
                if info.gender == gender_filter and 
                v in supported_voices_set and
                voice_matches_blind_locale(v, info, locale_filter)
            ]
        else:
            competitor_a_voices = [
                v for v, info in competitor_a_voice_info.items() 
                if info.gender == gender_filter and 
                v in supported_voices_set
            ]
    
    if not competitor_a_voices:
        locale_display = {"US": "US English", "TA": "Tamil (ta-IN)", "IN": "Indian English (en-IN)", "UK": "UK English (en-UK)", "HI": "Hindi", "BN": "Bangla (bn-IN)"}.get(locale_filter, locale_filter)
        st.error(f"Competitor {TTS_PROVIDERS[competitor_a_id].name} doesn't have any {gender_filter} {locale_display} voices available.")
        st.session_state.blind_test_2_current_pair = None
        return
    
    # Get voices for competitor B matching gender AND locale (only if Murf)
    if apply_locale_b:
        competitor_b_voices = get_voices_by_gender_and_locale(competitor_b_id, gender_filter, locale_filter)
    else:
        competitor_b_voices = get_voices_by_gender(competitor_b_id, gender_filter)
    if competitor_b_id in TTS_PROVIDERS:
        supported_voices_set = set(TTS_PROVIDERS[competitor_b_id].supported_voices)
        competitor_b_voices = [v for v in competitor_b_voices if v in supported_voices_set]
    
    # CRITICAL: For Cartesia, ALWAYS get fresh list and verify it's in voice_id_map
    if competitor_b_id in ["cartesia_sonic2", "cartesia_turbo", "cartesia_sonic3"]:
        try:
            provider_obj = TTSProviderFactory.create_provider(competitor_b_id)
            if hasattr(provider_obj, 'voice_id_map'):
                # Get completely fresh gender-filtered list (locale only if Murf)
                if apply_locale_b:
                    fresh_b = get_voices_by_gender_and_locale(competitor_b_id, gender_filter, locale_filter)
                else:
                    fresh_b = get_voices_by_gender(competitor_b_id, gender_filter)
                if competitor_b_id in TTS_PROVIDERS:
                    supported_set = set(TTS_PROVIDERS[competitor_b_id].supported_voices)
                    # Only use voices that are: 1) in voice_id_map, 2) supported, 3) correct gender
                    competitor_b_voices = [
                        v for v in fresh_b 
                        if v in supported_set and 
                        v in provider_obj.voice_id_map and
                        TTS_PROVIDERS[competitor_b_id].voice_info.get(v) and
                        TTS_PROVIDERS[competitor_b_id].voice_info[v].gender == gender_filter
                    ]
                    # Additional locale check only if Murf
                    if apply_locale_b:
                        _vb = TTS_PROVIDERS[competitor_b_id].voice_info
                        competitor_b_voices = [
                            v for v in competitor_b_voices
                            if _vb.get(v) and voice_matches_blind_locale(v, _vb[v], locale_filter)
                        ]
                    print(f"[CARTESIA PRE-FILTER] Competitor B: Filtered to {len(competitor_b_voices)} {gender_filter} {' ' + locale_filter if apply_locale_b else ''} voices")
        except Exception as e:
            print(f"Warning: Could not pre-filter Cartesia voices for B: {e}")
    
    # If no voices found, try from voice_info
    if not competitor_b_voices and competitor_b_id in TTS_PROVIDERS:
        competitor_b_voice_info = TTS_PROVIDERS[competitor_b_id].voice_info
        supported_voices_set = set(TTS_PROVIDERS[competitor_b_id].supported_voices)
        if apply_locale_b:
            competitor_b_voices = [
                v for v, info in competitor_b_voice_info.items() 
                if info.gender == gender_filter and 
                v in supported_voices_set and
                voice_matches_blind_locale(v, info, locale_filter)
            ]
        else:
            competitor_b_voices = [
                v for v, info in competitor_b_voice_info.items() 
                if info.gender == gender_filter and 
                v in supported_voices_set
            ]
    
    if not competitor_b_voices:
        locale_display = {"US": "US English", "TA": "Tamil (ta-IN)", "IN": "Indian English (en-IN)", "UK": "UK English (en-UK)", "HI": "Hindi", "BN": "Bangla (bn-IN)"}.get(locale_filter, locale_filter)
        st.error(f"Competitor {TTS_PROVIDERS[competitor_b_id].name} doesn't have any {gender_filter} {locale_display} voices available.")
        st.session_state.blind_test_2_current_pair = None
        return
    
    # CRITICAL: Re-verify voice lists contain ONLY correct gender AND locale voices before selection
    # This is especially important for Cartesia
    # Apply locale filter only if competitor is Murf
    _ainfo = TTS_PROVIDERS[competitor_a_id].voice_info
    competitor_a_voices = [
        v for v in competitor_a_voices
        if _ainfo.get(v) and _ainfo[v].gender == gender_filter
        and (not apply_locale_a or voice_matches_blind_locale(v, _ainfo[v], locale_filter))
    ]
    _binfo = TTS_PROVIDERS[competitor_b_id].voice_info
    competitor_b_voices = [
        v for v in competitor_b_voices
        if _binfo.get(v) and _binfo[v].gender == gender_filter
        and (not apply_locale_b or voice_matches_blind_locale(v, _binfo[v], locale_filter))
    ]
    
    if not competitor_a_voices:
        locale_display = {"US": "US English", "TA": "Tamil (ta-IN)", "IN": "Indian English (en-IN)", "UK": "UK English (en-UK)", "HI": "Hindi", "BN": "Bangla (bn-IN)"}.get(locale_filter, locale_filter)
        st.error(f"CRITICAL: Competitor A ({TTS_PROVIDERS[competitor_a_id].name}) has no {gender_filter} {locale_display} voices after filtering")
        st.session_state.blind_test_2_current_pair = None
        return
    
    if not competitor_b_voices:
        locale_display = {"US": "US English", "TA": "Tamil (ta-IN)", "IN": "Indian English (en-IN)", "UK": "UK English (en-UK)", "HI": "Hindi", "BN": "Bangla (bn-IN)"}.get(locale_filter, locale_filter)
        st.error(f"CRITICAL: Competitor B ({TTS_PROVIDERS[competitor_b_id].name}) has no {gender_filter} {locale_display} voices after filtering")
        st.session_state.blind_test_2_current_pair = None
        return
    
    # CRITICAL: One final filter to ensure ONLY correct gender AND locale voices remain (only if Murf)
    competitor_a_voices = [
        v for v in competitor_a_voices
        if _ainfo.get(v) and _ainfo[v].gender == gender_filter
        and (not apply_locale_a or voice_matches_blind_locale(v, _ainfo[v], locale_filter))
    ]
    competitor_b_voices = [
        v for v in competitor_b_voices
        if _binfo.get(v) and _binfo[v].gender == gender_filter
        and (not apply_locale_b or voice_matches_blind_locale(v, _binfo[v], locale_filter))
    ]
    
    if not competitor_a_voices or not competitor_b_voices:
        locale_display = {"US": "US English", "TA": "Tamil (ta-IN)", "IN": "Indian English (en-IN)", "UK": "UK English (en-UK)", "HI": "Hindi", "BN": "Bangla (bn-IN)"}.get(locale_filter, locale_filter)
        st.error(f"CRITICAL: No {gender_filter} {locale_display} voices available after final filtering")
        st.session_state.blind_test_2_current_pair = None
        return
    
    # Final gender + locale pass before intersection
    competitor_a_voices = [
        v for v in competitor_a_voices
        if _ainfo.get(v) and _ainfo[v].gender == gender_filter
        and (not apply_locale_a or voice_matches_blind_locale(v, _ainfo[v], locale_filter))
    ]
    competitor_b_voices = [
        v for v in competitor_b_voices
        if _binfo.get(v) and _binfo[v].gender == gender_filter
        and (not apply_locale_b or voice_matches_blind_locale(v, _binfo[v], locale_filter))
    ]
    
    if not competitor_a_voices or not competitor_b_voices:
        st.error(f"CRITICAL: No {gender_filter} voices available after final gender check")
        st.session_state.blind_test_2_current_pair = None
        return
    
    # Same Murf voiceId for both providers each round (NewModel maps via MURF_TO_OMNI_VOICE).
    common_voices = sorted(set(competitor_a_voices) & set(competitor_b_voices))
    if not common_voices:
        locale_display = {"US": "US English", "TA": "Tamil (ta-IN)", "IN": "Indian English (en-IN)", "UK": "UK English (en-UK)", "HI": "Hindi", "BN": "Bangla (bn-IN)"}.get(locale_filter, locale_filter)
        st.error(
            f"No overlapping {gender_filter} {locale_display} voices between "
            f"{TTS_PROVIDERS[competitor_a_id].name} and {TTS_PROVIDERS[competitor_b_id].name}."
        )
        st.session_state.blind_test_2_current_pair = None
        return
    
    voice_index = comparison_index % len(common_voices)
    competitor_a_voice = common_voices[voice_index]
    competitor_b_voice = common_voices[voice_index]
    competitor_a_voice_index = voice_index
    competitor_b_voice_index = voice_index
    
    # IMMEDIATE VERIFICATION: Double-check selected voices match gender
    selected_a_info = TTS_PROVIDERS[competitor_a_id].voice_info.get(competitor_a_voice)
    selected_b_info = TTS_PROVIDERS[competitor_b_id].voice_info.get(competitor_b_voice)
    
    # If wrong, force use first voice from overlapping pool
    if not selected_a_info or selected_a_info.gender != gender_filter:
        print(f"[CRITICAL] Competitor A voice {competitor_a_voice} wrong gender! Using first correct voice.")
        competitor_a_voice = common_voices[0]
        selected_a_info = TTS_PROVIDERS[competitor_a_id].voice_info.get(competitor_a_voice)
        if not selected_a_info or selected_a_info.gender != gender_filter:
            st.error(f"CRITICAL: Cannot find {gender_filter} voice for Competitor A")
            st.session_state.blind_test_2_current_pair = None
            return
    
    if not selected_b_info or selected_b_info.gender != gender_filter:
        print(f"[CRITICAL] Competitor B voice {competitor_b_voice} wrong gender! Using first correct voice.")
        competitor_b_voice = common_voices[0]
        selected_b_info = TTS_PROVIDERS[competitor_b_id].voice_info.get(competitor_b_voice)
        if not selected_b_info or selected_b_info.gender != gender_filter:
            st.error(f"CRITICAL: Cannot find {gender_filter} voice for Competitor B")
            st.session_state.blind_test_2_current_pair = None
            return
    
    if competitor_a_voice != competitor_b_voice:
        competitor_b_voice = competitor_a_voice
    
    # IMMEDIATE VERIFICATION: Check voices right after selection (before any Cartesia handling)
    competitor_a_voice_info = TTS_PROVIDERS[competitor_a_id].voice_info.get(competitor_a_voice)
    competitor_b_voice_info = TTS_PROVIDERS[competitor_b_id].voice_info.get(competitor_b_voice)
    
    # CRITICAL: If wrong gender detected, this should NEVER happen after filtering, but verify anyway
    if not competitor_a_voice_info or competitor_a_voice_info.gender != gender_filter:
        print(f"[CRITICAL ERROR] Competitor A voice {competitor_a_voice} is {competitor_a_voice_info.gender if competitor_a_voice_info else 'unknown'}, expected {gender_filter}")
        # Get completely fresh list with locale filter
        fresh_a = get_voices_by_gender_and_locale(competitor_a_id, gender_filter, locale_filter)
        if competitor_a_id in TTS_PROVIDERS:
            supported_set = set(TTS_PROVIDERS[competitor_a_id].supported_voices)
            fresh_a = [v for v in fresh_a if v in supported_set]
        if fresh_a:
            competitor_a_voice = fresh_a[0]
            competitor_a_voice_info = TTS_PROVIDERS[competitor_a_id].voice_info.get(competitor_a_voice)
            print(f"[EMERGENCY FIX] Competitor A set to {competitor_a_voice} ({competitor_a_voice_info.gender if competitor_a_voice_info else 'unknown'})")
        else:
            st.error(f"CRITICAL: Cannot find {gender_filter} voice for Competitor A")
            st.session_state.blind_test_2_current_pair = None
            return
    
    if not competitor_b_voice_info or competitor_b_voice_info.gender != gender_filter:
        print(f"[CRITICAL ERROR] Competitor B voice {competitor_b_voice} is {competitor_b_voice_info.gender if competitor_b_voice_info else 'unknown'}, expected {gender_filter}")
        # Get completely fresh list with locale filter
        fresh_b = get_voices_by_gender_and_locale(competitor_b_id, gender_filter, locale_filter)
        if competitor_b_id in TTS_PROVIDERS:
            supported_set = set(TTS_PROVIDERS[competitor_b_id].supported_voices)
            fresh_b = [v for v in fresh_b if v in supported_set]
        if fresh_b:
            competitor_b_voice = fresh_b[0]
            competitor_b_voice_info = TTS_PROVIDERS[competitor_b_id].voice_info.get(competitor_b_voice)
            print(f"[EMERGENCY FIX] Competitor B set to {competitor_b_voice} ({competitor_b_voice_info.gender if competitor_b_voice_info else 'unknown'})")
        else:
            st.error(f"CRITICAL: Cannot find {gender_filter} voice for Competitor B")
            st.session_state.blind_test_2_current_pair = None
            return
    
    # CRITICAL: Verify gender matches for both voices before proceeding
    
    # Fix competitor A if gender doesn't match
    if not competitor_a_voice_info or competitor_a_voice_info.gender != gender_filter:
        print(f"[GENDER FIX] Competitor A voice {competitor_a_voice} is {competitor_a_voice_info.gender if competitor_a_voice_info else 'unknown'}, expected {gender_filter}")
        correct_a_voices = [v for v in competitor_a_voices if TTS_PROVIDERS[competitor_a_id].voice_info.get(v) and TTS_PROVIDERS[competitor_a_id].voice_info[v].gender == gender_filter]
        if correct_a_voices:
            competitor_a_voice = correct_a_voices[competitor_a_voice_index % len(correct_a_voices)]
            competitor_a_voice_info = TTS_PROVIDERS[competitor_a_id].voice_info.get(competitor_a_voice)
            print(f"[GENDER FIX] Fixed Competitor A to {competitor_a_voice} ({competitor_a_voice_info.gender})")
        else:
            st.error(f"Competitor A ({TTS_PROVIDERS[competitor_a_id].name}) has no {gender_filter} voices")
            st.session_state.blind_test_2_current_pair = None
            return
    
    # Fix competitor B if gender doesn't match
    if not competitor_b_voice_info or competitor_b_voice_info.gender != gender_filter:
        print(f"[GENDER FIX] Competitor B voice {competitor_b_voice} is {competitor_b_voice_info.gender if competitor_b_voice_info else 'unknown'}, expected {gender_filter}")
        correct_b_voices = [v for v in competitor_b_voices if TTS_PROVIDERS[competitor_b_id].voice_info.get(v) and TTS_PROVIDERS[competitor_b_id].voice_info[v].gender == gender_filter]
        if correct_b_voices:
            competitor_b_voice = correct_b_voices[competitor_b_voice_index % len(correct_b_voices)]
            competitor_b_voice_info = TTS_PROVIDERS[competitor_b_id].voice_info.get(competitor_b_voice)
            print(f"[GENDER FIX] Fixed Competitor B to {competitor_b_voice} ({competitor_b_voice_info.gender})")
        else:
            st.error(f"Competitor B ({TTS_PROVIDERS[competitor_b_id].name}) has no {gender_filter} voices")
            st.session_state.blind_test_2_current_pair = None
            return
    
    print(f"[VOICE SELECTION] Competitor A ({competitor_a_id}): {competitor_a_voice} ({competitor_a_voice_info.gender}) - voice {competitor_a_voice_index + 1} of {len(competitor_a_voices)}")
    print(f"[VOICE SELECTION] Competitor B ({competitor_b_id}): {competitor_b_voice} ({competitor_b_voice_info.gender}) - voice {competitor_b_voice_index + 1} of {len(competitor_b_voices)}")
    
    # Special handling for Cartesia providers - ensure voice is in map AND matches gender
    # CRITICAL: For Cartesia, ALWAYS get fresh gender-filtered list - don't trust any previous lists
    for comp_id, comp_voice in [(competitor_a_id, competitor_a_voice), (competitor_b_id, competitor_b_voice)]:
        if comp_id in ["cartesia_sonic2", "cartesia_turbo", "cartesia_sonic3"]:
            try:
                provider_obj = TTSProviderFactory.create_provider(comp_id)
                if hasattr(provider_obj, 'voice_id_map'):
                    # FOR CARTESIA: ALWAYS get fresh gender-filtered list and re-verify
                    # Don't trust any previous selection - start completely fresh every time
                    voice_info_check = TTS_PROVIDERS[comp_id].voice_info.get(comp_voice)
                    voice_in_map = comp_voice in provider_obj.voice_id_map
                    gender_matches = voice_info_check and voice_info_check.gender == gender_filter
                    
                    # ALWAYS get completely fresh list for Cartesia - don't trust any previous lists
                    fresh_cartesia_voices = get_voices_by_gender_and_locale(comp_id, gender_filter, locale_filter)
                    if comp_id in TTS_PROVIDERS:
                        supported_set = set(TTS_PROVIDERS[comp_id].supported_voices)
                        # Filter to only voices that are in voice_id_map AND supported AND match gender AND locale
                        fresh_cartesia_voices = [
                            v for v in fresh_cartesia_voices 
                            if v in supported_set and 
                            v in provider_obj.voice_id_map and
                            TTS_PROVIDERS[comp_id].voice_info.get(v) and
                            TTS_PROVIDERS[comp_id].voice_info[v].gender == gender_filter and
                            voice_matches_blind_locale(v, TTS_PROVIDERS[comp_id].voice_info[v], locale_filter)
                        ]
                    
                    # If current voice is not in fresh list or doesn't match, replace it
                    if not voice_in_map or not gender_matches or comp_voice not in fresh_cartesia_voices:
                        if fresh_cartesia_voices:
                            # Use index-based selection from fresh list
                            voice_index = competitor_a_voice_index if comp_id == competitor_a_id else competitor_b_voice_index
                            new_voice = fresh_cartesia_voices[voice_index % len(fresh_cartesia_voices)]
                            new_voice_info = TTS_PROVIDERS[comp_id].voice_info.get(new_voice)
                            
                            # FINAL VERIFY the new voice matches gender
                            if new_voice_info and new_voice_info.gender == gender_filter:
                                if comp_id == competitor_a_id:
                                    competitor_a_voice = new_voice
                                    competitor_a_voice_info = new_voice_info
                                else:
                                    competitor_b_voice = new_voice
                                    competitor_b_voice_info = new_voice_info
                                print(f"[CARTESIA FIX] Set {comp_id} to {new_voice} (gender: {new_voice_info.gender})")
                            else:
                                # Last resort: use first voice from fresh list
                                final_voice = fresh_cartesia_voices[0]
                                final_voice_info = TTS_PROVIDERS[comp_id].voice_info.get(final_voice)
                                if comp_id == competitor_a_id:
                                    competitor_a_voice = final_voice
                                    competitor_a_voice_info = final_voice_info
                                else:
                                    competitor_b_voice = final_voice
                                    competitor_b_voice_info = final_voice_info
                                print(f"[CARTESIA EMERGENCY] Set {comp_id} to {final_voice} (gender: {final_voice_info.gender if final_voice_info else 'unknown'})")
                        else:
                            st.error(f"No {gender_filter} voice found in Cartesia for {TTS_PROVIDERS[comp_id].name}")
                            st.session_state.blind_test_2_current_pair = None
                            return
                    else:
                        # Voice is valid, but verify one more time anyway
                        verify_info = TTS_PROVIDERS[comp_id].voice_info.get(comp_voice)
                        if not verify_info or verify_info.gender != gender_filter:
                            # Get fresh list and fix
                            if fresh_cartesia_voices:
                                final_voice = fresh_cartesia_voices[0]
                                final_voice_info = TTS_PROVIDERS[comp_id].voice_info.get(final_voice)
                                if comp_id == competitor_a_id:
                                    competitor_a_voice = final_voice
                                    competitor_a_voice_info = final_voice_info
                                else:
                                    competitor_b_voice = final_voice
                                    competitor_b_voice_info = final_voice_info
                                print(f"[CARTESIA FINAL FIX] Corrected {comp_id} to {final_voice} (gender: {final_voice_info.gender if final_voice_info else 'unknown'})")
            except Exception as e:
                print(f"Warning: Could not validate Cartesia voice: {e}")
    
    # POST-CARTESIA VERIFICATION: Immediately verify both voices after Cartesia handling
    post_cartesia_a_info = TTS_PROVIDERS[competitor_a_id].voice_info.get(competitor_a_voice)
    post_cartesia_b_info = TTS_PROVIDERS[competitor_b_id].voice_info.get(competitor_b_voice)
    
    if competitor_a_id in ["cartesia_sonic2", "cartesia_turbo", "cartesia_sonic3"]:
        if not post_cartesia_a_info or post_cartesia_a_info.gender != gender_filter:
            print(f"[POST-CARTESIA CHECK] Competitor A still wrong! Getting fresh list...")
            fresh_a = get_voices_by_gender(competitor_a_id, gender_filter)
            if competitor_a_id in TTS_PROVIDERS:
                supported_set = set(TTS_PROVIDERS[competitor_a_id].supported_voices)
                fresh_a = [v for v in fresh_a if v in supported_set]
            if fresh_a:
                competitor_a_voice = fresh_a[0]
                post_cartesia_a_info = TTS_PROVIDERS[competitor_a_id].voice_info.get(competitor_a_voice)
                print(f"[POST-CARTESIA FIX] Competitor A set to {competitor_a_voice} ({post_cartesia_a_info.gender if post_cartesia_a_info else 'unknown'})")
    
    if competitor_b_id in ["cartesia_sonic2", "cartesia_turbo", "cartesia_sonic3"]:
        if not post_cartesia_b_info or post_cartesia_b_info.gender != gender_filter:
            print(f"[POST-CARTESIA CHECK] Competitor B still wrong! Getting fresh list...")
            fresh_b = get_voices_by_gender(competitor_b_id, gender_filter)
            if competitor_b_id in TTS_PROVIDERS:
                supported_set = set(TTS_PROVIDERS[competitor_b_id].supported_voices)
                fresh_b = [v for v in fresh_b if v in supported_set]
            if fresh_b:
                competitor_b_voice = fresh_b[0]
                post_cartesia_b_info = TTS_PROVIDERS[competitor_b_id].voice_info.get(competitor_b_voice)
                print(f"[POST-CARTESIA FIX] Competitor B set to {competitor_b_voice} ({post_cartesia_b_info.gender if post_cartesia_b_info else 'unknown'})")
    
    # FINAL CRITICAL VERIFICATION: Double-check both voices match gender before generating audio
    # Use get_voices_by_gender to get FRESH list of correct gender voices (don't trust previous lists)
    final_a_info = TTS_PROVIDERS[competitor_a_id].voice_info.get(competitor_a_voice)
    final_b_info = TTS_PROVIDERS[competitor_b_id].voice_info.get(competitor_b_voice)
    
    # Verify Competitor A - get fresh list if needed
    if not final_a_info or final_a_info.gender != gender_filter:
        print(f"[FINAL GENDER CHECK] ERROR: Competitor A voice {competitor_a_voice} is {final_a_info.gender if final_a_info else 'unknown'}, expected {gender_filter}")
        # Get FRESH list of correct gender voices using get_voices_by_gender
        fresh_a_voices = get_voices_by_gender(competitor_a_id, gender_filter)
        if competitor_a_id in TTS_PROVIDERS:
            supported_set = set(TTS_PROVIDERS[competitor_a_id].supported_voices)
            fresh_a_voices = [v for v in fresh_a_voices if v in supported_set]
        if fresh_a_voices:
            # Use first valid voice of correct gender
            competitor_a_voice = fresh_a_voices[0]
            final_a_info = TTS_PROVIDERS[competitor_a_id].voice_info.get(competitor_a_voice)
            print(f"[FINAL GENDER CHECK] FORCED Competitor A to {competitor_a_voice} ({final_a_info.gender if final_a_info else 'unknown'})")
        else:
            st.error(f"CRITICAL: Competitor A ({TTS_PROVIDERS[competitor_a_id].name}) has no {gender_filter} voices available")
            st.session_state.blind_test_2_current_pair = None
            return
    
    # Verify Competitor B - get fresh list if needed
    if not final_b_info or final_b_info.gender != gender_filter:
        print(f"[FINAL GENDER CHECK] ERROR: Competitor B voice {competitor_b_voice} is {final_b_info.gender if final_b_info else 'unknown'}, expected {gender_filter}")
        # Get FRESH list of correct gender voices using get_voices_by_gender
        fresh_b_voices = get_voices_by_gender(competitor_b_id, gender_filter)
        if competitor_b_id in TTS_PROVIDERS:
            supported_set = set(TTS_PROVIDERS[competitor_b_id].supported_voices)
            fresh_b_voices = [v for v in fresh_b_voices if v in supported_set]
        if fresh_b_voices:
            # Use first valid voice of correct gender
            competitor_b_voice = fresh_b_voices[0]
            final_b_info = TTS_PROVIDERS[competitor_b_id].voice_info.get(competitor_b_voice)
            print(f"[FINAL GENDER CHECK] FORCED Competitor B to {competitor_b_voice} ({final_b_info.gender if final_b_info else 'unknown'})")
        else:
            st.error(f"CRITICAL: Competitor B ({TTS_PROVIDERS[competitor_b_id].name}) has no {gender_filter} voices available")
            st.session_state.blind_test_2_current_pair = None
            return
    
    # ABSOLUTE FINAL CHECK - abort if still wrong
    if not final_a_info or final_a_info.gender != gender_filter:
        st.error(f"CRITICAL: Competitor A voice {competitor_a_voice} still has wrong gender: {final_a_info.gender if final_a_info else 'unknown'}")
        st.session_state.blind_test_2_current_pair = None
        return
    
    if not final_b_info or final_b_info.gender != gender_filter:
        st.error(f"CRITICAL: Competitor B voice {competitor_b_voice} still has wrong gender: {final_b_info.gender if final_b_info else 'unknown'}")
        st.session_state.blind_test_2_current_pair = None
        return
    
    # ABSOLUTE FINAL VERIFICATION RIGHT BEFORE AUDIO GENERATION
    # CRITICAL: For Cartesia, we MUST verify voice is in voice_id_map AND matches gender
    # Re-check both voices one more time - if wrong, force correct from fresh gender-filtered list
    
    # Check Competitor A
    pre_gen_a_info = TTS_PROVIDERS[competitor_a_id].voice_info.get(competitor_a_voice)
    if competitor_a_id in ["cartesia_sonic2", "cartesia_turbo", "cartesia_sonic3"]:
        try:
            provider_obj_a = TTSProviderFactory.create_provider(competitor_a_id)
            if hasattr(provider_obj_a, 'voice_id_map'):
                # For Cartesia, voice MUST be in voice_id_map AND match gender
                if competitor_a_voice not in provider_obj_a.voice_id_map or not pre_gen_a_info or pre_gen_a_info.gender != gender_filter:
                    print(f"[CARTESIA PRE-GEN] Competitor A voice {competitor_a_voice} invalid! Getting fresh Cartesia list...")
                    fresh_a_cartesia = get_voices_by_gender(competitor_a_id, gender_filter)
                    if competitor_a_id in TTS_PROVIDERS:
                        supported_set = set(TTS_PROVIDERS[competitor_a_id].supported_voices)
                        # CRITICAL: Only use voices that are in voice_id_map AND match gender
                        fresh_a_cartesia = [
                            v for v in fresh_a_cartesia 
                            if v in supported_set and 
                            v in provider_obj_a.voice_id_map and
                            TTS_PROVIDERS[competitor_a_id].voice_info.get(v) and
                            TTS_PROVIDERS[competitor_a_id].voice_info[v].gender == gender_filter
                        ]
                    if fresh_a_cartesia:
                        competitor_a_voice = fresh_a_cartesia[comparison_index % len(fresh_a_cartesia)]
                        pre_gen_a_info = TTS_PROVIDERS[competitor_a_id].voice_info.get(competitor_a_voice)
                        print(f"[CARTESIA PRE-GEN FIX] Competitor A set to {competitor_a_voice} ({pre_gen_a_info.gender if pre_gen_a_info else 'unknown'})")
        except Exception as e:
            print(f"Warning: Could not validate Cartesia voice for A: {e}")
    
    if not pre_gen_a_info or pre_gen_a_info.gender != gender_filter:
        print(f"[PRE-GEN FIX] Competitor A voice {competitor_a_voice} wrong gender! Getting fresh list...")
        fresh_a_final = get_voices_by_gender(competitor_a_id, gender_filter)
        if competitor_a_id in TTS_PROVIDERS:
            supported_set = set(TTS_PROVIDERS[competitor_a_id].supported_voices)
            fresh_a_final = [v for v in fresh_a_final if v in supported_set]
        if fresh_a_final:
            competitor_a_voice = fresh_a_final[comparison_index % len(fresh_a_final)]
            pre_gen_a_info = TTS_PROVIDERS[competitor_a_id].voice_info.get(competitor_a_voice)
            print(f"[PRE-GEN FIX] Competitor A set to {competitor_a_voice} ({pre_gen_a_info.gender if pre_gen_a_info else 'unknown'})")
    
    # Check Competitor B
    pre_gen_b_info = TTS_PROVIDERS[competitor_b_id].voice_info.get(competitor_b_voice)
    if competitor_b_id in ["cartesia_sonic2", "cartesia_turbo", "cartesia_sonic3"]:
        try:
            provider_obj_b = TTSProviderFactory.create_provider(competitor_b_id)
            if hasattr(provider_obj_b, 'voice_id_map'):
                # For Cartesia, voice MUST be in voice_id_map AND match gender
                if competitor_b_voice not in provider_obj_b.voice_id_map or not pre_gen_b_info or pre_gen_b_info.gender != gender_filter:
                    print(f"[CARTESIA PRE-GEN] Competitor B voice {competitor_b_voice} invalid! Getting fresh Cartesia list...")
                    fresh_b_cartesia = get_voices_by_gender(competitor_b_id, gender_filter)
                    if competitor_b_id in TTS_PROVIDERS:
                        supported_set = set(TTS_PROVIDERS[competitor_b_id].supported_voices)
                        # CRITICAL: Only use voices that are in voice_id_map AND match gender
                        fresh_b_cartesia = [
                            v for v in fresh_b_cartesia 
                            if v in supported_set and 
                            v in provider_obj_b.voice_id_map and
                            TTS_PROVIDERS[competitor_b_id].voice_info.get(v) and
                            TTS_PROVIDERS[competitor_b_id].voice_info[v].gender == gender_filter
                        ]
                    if fresh_b_cartesia:
                        competitor_b_voice = fresh_b_cartesia[comparison_index % len(fresh_b_cartesia)]
                        pre_gen_b_info = TTS_PROVIDERS[competitor_b_id].voice_info.get(competitor_b_voice)
                        print(f"[CARTESIA PRE-GEN FIX] Competitor B set to {competitor_b_voice} ({pre_gen_b_info.gender if pre_gen_b_info else 'unknown'})")
        except Exception as e:
            print(f"Warning: Could not validate Cartesia voice for B: {e}")
    
    if not pre_gen_b_info or pre_gen_b_info.gender != gender_filter:
        print(f"[PRE-GEN FIX] Competitor B voice {competitor_b_voice} wrong gender! Getting fresh list...")
        fresh_b_final = get_voices_by_gender(competitor_b_id, gender_filter)
        if competitor_b_id in TTS_PROVIDERS:
            supported_set = set(TTS_PROVIDERS[competitor_b_id].supported_voices)
            fresh_b_final = [v for v in fresh_b_final if v in supported_set]
        if fresh_b_final:
            competitor_b_voice = fresh_b_final[comparison_index % len(fresh_b_final)]
            pre_gen_b_info = TTS_PROVIDERS[competitor_b_id].voice_info.get(competitor_b_voice)
            print(f"[PRE-GEN FIX] Competitor B set to {competitor_b_voice} ({pre_gen_b_info.gender if pre_gen_b_info else 'unknown'})")
    
    # Final abort check - if still wrong, don't generate
    if not pre_gen_a_info or pre_gen_a_info.gender != gender_filter:
        st.error(f"CRITICAL: Cannot find {gender_filter} voice for Competitor A. Aborting.")
        st.session_state.blind_test_2_current_pair = None
        return
    
    if not pre_gen_b_info or pre_gen_b_info.gender != gender_filter:
        st.error(f"CRITICAL: Cannot find {gender_filter} voice for Competitor B. Aborting.")
        st.session_state.blind_test_2_current_pair = None
        return
    
    # Murf Gen2 + NewModel must use the same Murf voiceId per round (emergency Cartesia-style fixes can desync).
    if {competitor_a_id, competitor_b_id} == {"murf_gen2", "omni_tts"}:
        shared = competitor_a_voice if competitor_a_id == "murf_gen2" else competitor_b_voice
        competitor_a_voice = shared
        competitor_b_voice = shared
        pre_gen_a_info = TTS_PROVIDERS[competitor_a_id].voice_info.get(shared)
        pre_gen_b_info = TTS_PROVIDERS[competitor_b_id].voice_info.get(shared)
    
    for _pid, _vid in ((competitor_a_id, competitor_a_voice), (competitor_b_id, competitor_b_voice)):
        _li = TTS_PROVIDERS[_pid].voice_info.get(_vid)
        if (
            not _li
            or _li.gender != gender_filter
            or not voice_matches_blind_locale(_vid, _li, locale_filter)
        ):
            st.error(
                f"Voice `{_vid}` does not match **{gender_filter}** + **{locale_filter}** for this test. "
                "Reload the app and start the test again."
            )
            st.session_state.blind_test_2_current_pair = None
            return
    
    print(f"[FINAL VERIFICATION] ✓ Both voices confirmed: A={competitor_a_voice} ({pre_gen_a_info.gender}), B={competitor_b_voice} ({pre_gen_b_info.gender})")
    
    # Show loading placeholder while generating
    loading_placeholder = st.empty()
    loading_placeholder.info(f"Generating audio samples... This may take a few seconds.")
    
    # Generate samples (runs in parallel for speed)
    print(f"[DEBUG] Generating audio for text: '{text[:50]}...' (Comparison #{st.session_state.blind_test_2_comparison_count})")
    
    try:
        sample_a, sample_b = asyncio.run(generate_comparison_samples(
            text, competitor_a_id, competitor_a_voice, competitor_b_id, competitor_b_voice
        ))
        
        # Validate that samples were generated with the correct text
        if sample_a and hasattr(sample_a, 'metadata') and sample_a.metadata:
            generated_text_a = sample_a.metadata.get('text', '')
            if generated_text_a and generated_text_a != text:
                print(f"[WARNING] Sample A text mismatch! Expected: '{text[:50]}', Got: '{generated_text_a[:50]}'")
        
        if sample_b and hasattr(sample_b, 'metadata') and sample_b.metadata:
            generated_text_b = sample_b.metadata.get('text', '')
            if generated_text_b and generated_text_b != text:
                print(f"[WARNING] Sample B text mismatch! Expected: '{text[:50]}', Got: '{generated_text_b[:50]}'")
                
    except Exception as e:
        loading_placeholder.error(f"Error generating samples: {str(e)}")
        sample_a, sample_b = None, None
    
    # Clear loading message
    loading_placeholder.empty()
    
    # Check if either sample failed
    if sample_a is None or sample_b is None or not (sample_a.success if sample_a else False) or not (sample_b.success if sample_b else False):
        # Show detailed error and allow retry
        error_msg = "Both samples failed to generate."
        error_details = []
        
        if sample_a:
            if not sample_a.success:
                error_details.append(f"Sample A ({TTS_PROVIDERS[competitor_a_id].name}): {sample_a.error_message if hasattr(sample_a, 'error_message') and sample_a.error_message else 'Unknown error'}")
        else:
            error_details.append(f"Sample A ({TTS_PROVIDERS[competitor_a_id].name}): Failed to generate")
            
        if sample_b:
            if not sample_b.success:
                error_details.append(f"Sample B ({TTS_PROVIDERS[competitor_b_id].name}): {sample_b.error_message if hasattr(sample_b, 'error_message') and sample_b.error_message else 'Unknown error'}")
        else:
            error_details.append(f"Sample B ({TTS_PROVIDERS[competitor_b_id].name}): Failed to generate")
        
        if error_details:
            error_msg = " | ".join(error_details)
        
        st.session_state.blind_test_2_current_pair = {"error": True, "message": error_msg}
        st.rerun()
        return
    
    # CRITICAL FIX: Validate text matches before storing
    if sample_a and sample_a.success:
        if not hasattr(sample_a, 'metadata') or sample_a.metadata is None:
            sample_a.metadata = {}
        sample_a.metadata['generated_text'] = text
        sample_a.metadata['comparison_num'] = st.session_state.blind_test_2_comparison_count
    
    if sample_b and sample_b.success:
        if not hasattr(sample_b, 'metadata') or sample_b.metadata is None:
            sample_b.metadata = {}
        sample_b.metadata['generated_text'] = text
        sample_b.metadata['comparison_num'] = st.session_state.blind_test_2_comparison_count
    
    # Randomize order (50/50 chance A or B)
    a_is_first = random.random() > 0.5
    
    # Generate a unique timestamp for this comparison to prevent caching
    unique_timestamp = time.time()
    comparison_id = f"{st.session_state.blind_test_2_comparison_count}_{unique_timestamp}"
    
    # Log for debugging
    print(f"[DEBUG] Storing pair - Text: '{text[:60]}...', Comparison #{st.session_state.blind_test_2_comparison_count}")
    
    if a_is_first:
        st.session_state.blind_test_2_current_pair = {
            "sample_a": sample_a, "sample_b": sample_b,
            "provider_a": competitor_a_id, "provider_b": competitor_b_id,
            "voice_a": competitor_a_voice, "voice_b": competitor_b_voice,
            "text": text, "generated_at": unique_timestamp,
            "comparison_id": comparison_id
        }
    else:
        st.session_state.blind_test_2_current_pair = {
            "sample_a": sample_b, "sample_b": sample_a,
            "provider_a": competitor_b_id, "provider_b": competitor_a_id,
            "voice_a": competitor_b_voice, "voice_b": competitor_a_voice,
            "text": text, "generated_at": unique_timestamp,
            "comparison_id": comparison_id
        }
    
    st.session_state.blind_test_2_audio_played = {"A": 0, "B": 0}
    st.rerun()


def generate_next_comparison():
    """Generate the next comparison pair - ALWAYS generates fresh audio"""
    from config import get_voices_by_gender, get_voice_gender, voice_matches_blind_locale
    import random
    import time
    
    print(f"[GENERATE DEBUG] Starting generate_next_comparison for comparison #{st.session_state.blind_test_comparison_count}")
    
    # Check if we've reached max comparisons
    if st.session_state.blind_test_comparison_count >= st.session_state.blind_test_max_comparisons:
        st.session_state.show_final_results = True
        st.rerun()
        return
    
    # FORCE CLEAR any existing pair first to prevent stale audio
    st.session_state.blind_test_current_pair = None
    
    # Generate a unique generation ID for this comparison
    generation_id = f"gen_{int(time.time() * 1000)}_{st.session_state.blind_test_comparison_count}"
    print(f"[GENERATE DEBUG] Generation ID: {generation_id}")
    
    # Get a random sentence
    sentences = st.session_state.blind_test_sentences
    if not sentences:
        st.error("No sentences available")
        return
    
    # CRITICAL: Track which sentences have been used to ensure variety
    if "used_sentences" not in st.session_state:
        st.session_state.used_sentences = []
    
    # Get available sentences (ones not used yet, or all if all have been used)
    available_sentences = [s for s in sentences if s not in st.session_state.used_sentences]
    
    # If all sentences have been used, reset and start fresh
    if not available_sentences:
        st.session_state.used_sentences = []
        available_sentences = sentences
    
    # Select a random sentence from available ones
    text = random.choice(available_sentences)
    
    # Mark this sentence as used
    st.session_state.used_sentences.append(text)
    
    # CRITICAL DEBUG: Log the selected text
    print(f"[CRITICAL DEBUG] Selected sentence #{len(st.session_state.used_sentences)}: '{text[:80]}...'")
    print(f"[CRITICAL DEBUG] Available sentences: {len(available_sentences)}, Total: {len(sentences)}")
    
    # Get configured Murf provider - use selected provider from setup, or fallback to falcon
    config_status = check_configuration()
    murf_providers = [
        p for p, status in config_status["providers"].items() 
        if status["configured"] and "murf" in p.lower()
    ]
    
    # Use the provider selected in setup screen, or fallback to falcon if available
    if "blind_test_murf_provider" in st.session_state and st.session_state.blind_test_murf_provider in murf_providers:
        murf_provider_id = st.session_state.blind_test_murf_provider
    elif "murf_gen2" in murf_providers:
        murf_provider_id = "murf_gen2"
    elif murf_providers:
        murf_provider_id = murf_providers[0]
    else:
        murf_provider_id = None
    
    if not murf_provider_id:
        st.error("No Murf provider configured. Please set MURF_API_KEY.")
        return
    
    # Get Murf voice - SHUFFLE through selected voices
    gender_filter = st.session_state.blind_test_gender_filter
    comparison_index = st.session_state.blind_test_comparison_count
    
    # Get selected MURF voices (up to 4)
    selected_murf_voices = st.session_state.blind_test_murf_voices if st.session_state.blind_test_murf_voices else []
    
    # Check if Murf is selected as competitor - if so, extract locale from selected voices
    competitor_id = st.session_state.blind_test_selected_competitors[0] if st.session_state.blind_test_selected_competitors else None
    is_murf_competitor = competitor_id and "murf" in competitor_id.lower()
    
    # Extract locale from selected Murf voices if Murf is competitor
    locale_filter = None
    if is_murf_competitor and selected_murf_voices:
        # Extract locale from first selected voice
        first_voice = selected_murf_voices[0]
        if "en-US" in first_voice or "en-us" in first_voice.lower():
            locale_filter = "US"
        elif "en-IN" in first_voice or "en-in" in first_voice.lower():
            locale_filter = "IN"
        elif "en-UK" in first_voice or "en-uk" in first_voice.lower():
            locale_filter = "UK"
        elif "hi-IN" in first_voice or "hi-in" in first_voice.lower():
            locale_filter = "HI"
        elif "bn-IN" in first_voice or "bn-in" in first_voice.lower():
            locale_filter = "BN"
        elif "ta-IN" in first_voice or "ta-in" in first_voice.lower():
            locale_filter = "TA"
    
    # If no voices selected, fall back to single voice or get first voice of selected gender
    if not selected_murf_voices:
        murf_voice = st.session_state.blind_test_murf_voice
        if not murf_voice or murf_voice not in TTS_PROVIDERS[murf_provider_id].supported_voices:
            murf_voice_info = TTS_PROVIDERS[murf_provider_id].voice_info
            filtered_voices = [(v, info) for v, info in murf_voice_info.items() if info.gender == gender_filter]
            # Apply locale filter if Murf is competitor and locale is extracted
            if is_murf_competitor and locale_filter:
                filtered_voices = [
                    (v, info) for v, info in filtered_voices
                    if voice_matches_blind_locale(v, info, locale_filter)
                ]
            if filtered_voices:
                murf_voice = filtered_voices[0][0]
                st.session_state.blind_test_murf_voice = murf_voice
                selected_murf_voices = [murf_voice]
        # Filter selected voices by locale if Murf is competitor and locale is extracted
        if is_murf_competitor and selected_murf_voices and locale_filter:
            murf_voice_info = TTS_PROVIDERS[murf_provider_id].voice_info
            selected_murf_voices = [
                v for v in selected_murf_voices
                if v in TTS_PROVIDERS[murf_provider_id].supported_voices and
                murf_voice_info.get(v) and
                murf_voice_info[v].gender == gender_filter and
                voice_matches_blind_locale(v, murf_voice_info[v], locale_filter)
            ]
            if selected_murf_voices:
                murf_voice = selected_murf_voices[0]
                st.session_state.blind_test_murf_voice = murf_voice
        voice_index = 0  # Only one voice selected
    else:
        # Filter selected voices by locale if Murf is competitor and locale is extracted
        if is_murf_competitor and locale_filter:
            murf_voice_info = TTS_PROVIDERS[murf_provider_id].voice_info
            selected_murf_voices = [
                v for v in selected_murf_voices
                if v in TTS_PROVIDERS[murf_provider_id].supported_voices and
                murf_voice_info.get(v) and
                murf_voice_info[v].gender == gender_filter and
                voice_matches_blind_locale(v, murf_voice_info[v], locale_filter)
            ]
            # Update session state with filtered voices
            if selected_murf_voices:
                st.session_state.blind_test_murf_voices = selected_murf_voices
            else:
                # If no voices match locale, get first voice matching gender and locale
                murf_voice_info = TTS_PROVIDERS[murf_provider_id].voice_info
                filtered_voices = [
                    (v, info) for v, info in murf_voice_info.items()
                    if info.gender == gender_filter and voice_matches_blind_locale(v, info, locale_filter)
                ]
                if filtered_voices:
                    selected_murf_voices = [filtered_voices[0][0]]
                    st.session_state.blind_test_murf_voices = selected_murf_voices
        
        # Cycle through selected voices based on comparison count
        # Use modulo to cycle through the list
        if selected_murf_voices:
            voice_index = comparison_index % len(selected_murf_voices)
            murf_voice = selected_murf_voices[voice_index]
        else:
            # Fallback if no voices match
            murf_voice_info = TTS_PROVIDERS[murf_provider_id].voice_info
            filtered_voices = [(v, info) for v, info in murf_voice_info.items() if info.gender == gender_filter]
            if is_murf_competitor and locale_filter:
                filtered_voices = [
                    (v, info) for v, info in filtered_voices
                    if voice_matches_blind_locale(v, info, locale_filter)
                ]
            if filtered_voices:
                murf_voice = filtered_voices[0][0]
                selected_murf_voices = [murf_voice]
                st.session_state.blind_test_murf_voices = selected_murf_voices
                voice_index = 0
            else:
                st.error(f"No {gender_filter} voices available for Murf with locale {locale_filter}")
                st.session_state.blind_test_current_pair = None
                return
        
        # Ensure selected voice is still valid
        if murf_voice not in TTS_PROVIDERS[murf_provider_id].supported_voices:
            # Voice is invalid, filter and use first valid one
            murf_voice_info = TTS_PROVIDERS[murf_provider_id].voice_info
            filtered_voices = [(v, info) for v, info in murf_voice_info.items() if info.gender == gender_filter]
            if is_murf_competitor and locale_filter:
                filtered_voices = [
                    (v, info) for v, info in filtered_voices
                    if voice_matches_blind_locale(v, info, locale_filter)
                ]
            valid_voice_ids = [v for v in selected_murf_voices if v in TTS_PROVIDERS[murf_provider_id].supported_voices]
            if valid_voice_ids:
                voice_index = comparison_index % len(valid_voice_ids)
                murf_voice = valid_voice_ids[voice_index]
            elif filtered_voices:
                murf_voice = filtered_voices[0][0]
                selected_murf_voices = [murf_voice]
                st.session_state.blind_test_murf_voices = selected_murf_voices
                voice_index = 0
    
    print(f"[MURF VOICE DEBUG] Comparison #{comparison_index + 1}: Using MURF voice: {murf_voice} (voice {voice_index + 1} of {len(selected_murf_voices)} selected)")
    
    # Get the selected competitor (single selection now)
    competitors = st.session_state.blind_test_selected_competitors
    if not competitors:
        st.error("No competitor selected")
        return
    
    competitor_id = competitors[0]  # Single competitor selected
    
    # Check if competitor is Murf
    is_murf_competitor = "murf" in competitor_id.lower()
    
    # Get a voice with matching gender from competitor - MUST MATCH GENDER
    # Get the actual gender of the selected Murf voice to ensure perfect match
    murf_voice_info = TTS_PROVIDERS[murf_provider_id].voice_info.get(murf_voice)
    if murf_voice_info:
        actual_gender = murf_voice_info.gender
        # Use the actual gender from the voice, not just the filter
        gender_filter = actual_gender
        print(f"[GENDER DEBUG] Murf voice: {murf_voice} is {actual_gender}")
    else:
        print(f"[GENDER DEBUG] WARNING: Could not find Murf voice info for {murf_voice}, using filter: {gender_filter}")
    
    # Get competitor voices matching gender - apply locale filter if competitor is Murf and locale is extracted
    if is_murf_competitor and locale_filter:
        from config import get_voices_by_gender_and_locale
        competitor_voices = get_voices_by_gender_and_locale(competitor_id, gender_filter, locale_filter)
        print(f"[GENDER DEBUG] Competitor {competitor_id} voices matching '{gender_filter}' and locale '{locale_filter}': {competitor_voices}")
    else:
        competitor_voices = get_voices_by_gender(competitor_id, gender_filter)
        print(f"[GENDER DEBUG] Competitor {competitor_id} voices matching '{gender_filter}': {competitor_voices}")
    
    # Additional validation: ensure voices are in supported_voices list
    if competitor_id in TTS_PROVIDERS:
        supported_voices_set = set(TTS_PROVIDERS[competitor_id].supported_voices)
        competitor_voices = [v for v in competitor_voices if v in supported_voices_set]
    
    # CRITICAL: For Cartesia, ALWAYS get fresh list and verify it's in voice_id_map BEFORE selection
    if competitor_id in ["cartesia_sonic2", "cartesia_turbo", "cartesia_sonic3"]:
        try:
            provider_obj = TTSProviderFactory.create_provider(competitor_id)
            if hasattr(provider_obj, 'voice_id_map'):
                # Get completely fresh gender-filtered list
                fresh_voices = get_voices_by_gender(competitor_id, gender_filter)
                if competitor_id in TTS_PROVIDERS:
                    supported_set = set(TTS_PROVIDERS[competitor_id].supported_voices)
                    # Only use voices that are: 1) in voice_id_map, 2) supported, 3) correct gender
                    competitor_voices = [
                        v for v in fresh_voices 
                        if v in supported_set and 
                        v in provider_obj.voice_id_map and
                        TTS_PROVIDERS[competitor_id].voice_info.get(v) and
                        TTS_PROVIDERS[competitor_id].voice_info[v].gender == gender_filter
                    ]
                    print(f"[CARTESIA PRE-FILTER] Unranked Blind Test: Filtered to {len(competitor_voices)} {gender_filter} voices")
        except Exception as e:
            print(f"Warning: Could not pre-filter Cartesia voices: {e}")
    
    # If no voices found for this gender, try to find any voice with matching gender from voice_info
    if not competitor_voices and competitor_id in TTS_PROVIDERS:
        competitor_voice_info = TTS_PROVIDERS[competitor_id].voice_info
        supported_voices_set = set(TTS_PROVIDERS[competitor_id].supported_voices)
        competitor_voices = [
            v for v, info in competitor_voice_info.items() 
            if info.gender == gender_filter and v in supported_voices_set
        ]
    
    # If both competitor and Murf provider are Murf, use same selected voices for both (skip random selection)
    if is_murf_competitor and murf_provider_id and "murf" in murf_provider_id.lower():
        # Use the same selected voices for competitor (same as Murf provider)
        # Cycle through selected voices using comparison index
        if selected_murf_voices:
            competitor_voice_index = comparison_index % len(selected_murf_voices)
            competitor_voice = selected_murf_voices[competitor_voice_index]
            print(f"[MURF VS MURF] Using same voice for both: {competitor_voice} (voice {competitor_voice_index + 1} of {len(selected_murf_voices)})")
            
            # Verify the voice is valid for competitor
            if competitor_voice not in TTS_PROVIDERS[competitor_id].supported_voices:
                st.error(f"Selected voice {competitor_voice} not available for {TTS_PROVIDERS[competitor_id].name}")
                st.session_state.blind_test_current_pair = None
                return
        else:
            st.error("No Murf voices selected")
            st.session_state.blind_test_current_pair = None
            return
    elif not competitor_voices:
        # Only show error if not Murf vs Murf (we already handled that case above)
        st.error(f"Competitor {TTS_PROVIDERS[competitor_id].name} doesn't have any {gender_filter} voices available. Please select a different competitor.")
        st.session_state.blind_test_current_pair = None
        return
    
    # If competitor is NOT Murf, use existing random selection logic
    if not (is_murf_competitor and murf_provider_id and "murf" in murf_provider_id.lower()):
        # CRITICAL: Final filter to ensure ONLY correct gender voices remain - do this MULTIPLE times
        # Also apply locale filter if competitor is Murf and locale is extracted
        for _ in range(3):  # Triple filter to be absolutely sure
            _cvi = TTS_PROVIDERS[competitor_id].voice_info
            competitor_voices = [
                v for v in competitor_voices 
                if _cvi.get(v) and 
                _cvi[v].gender == gender_filter and
                (not is_murf_competitor or not locale_filter or voice_matches_blind_locale(v, _cvi[v], locale_filter))
            ]
        
        if not competitor_voices:
            st.error(f"CRITICAL: No {gender_filter} voices available after final filtering for {TTS_PROVIDERS[competitor_id].name}")
            st.session_state.blind_test_current_pair = None
            return
        
        # Select from matching gender voices only
        # If only 1 voice available, use it; otherwise randomly pick one
        if len(competitor_voices) == 1:
            competitor_voice = competitor_voices[0]
            print(f"[GENDER DEBUG] Only 1 {gender_filter} voice available for competitor: {competitor_voice}")
        else:
            competitor_voice = random.choice(competitor_voices)
            print(f"[GENDER DEBUG] Selected {gender_filter} voice for competitor: {competitor_voice} (from {len(competitor_voices)} options)")
    
    # IMMEDIATE VERIFICATION: Double-check selected voice matches gender - if wrong, force correct
    selected_voice_check = TTS_PROVIDERS[competitor_id].voice_info.get(competitor_voice)
    if not selected_voice_check or selected_voice_check.gender != gender_filter:
        print(f"[CRITICAL] Selected voice {competitor_voice} wrong gender! Getting fresh list and using first correct voice.")
        # Get completely fresh list
        fresh_final = get_voices_by_gender(competitor_id, gender_filter)
        if competitor_id in TTS_PROVIDERS:
            supported_set = set(TTS_PROVIDERS[competitor_id].supported_voices)
            fresh_final = [v for v in fresh_final if v in supported_set]
            # Filter again by gender
            fresh_final = [
                v for v in fresh_final 
                if TTS_PROVIDERS[competitor_id].voice_info.get(v) and 
                TTS_PROVIDERS[competitor_id].voice_info[v].gender == gender_filter
            ]
        if fresh_final:
            competitor_voice = fresh_final[0]
            selected_voice_check = TTS_PROVIDERS[competitor_id].voice_info.get(competitor_voice)
            print(f"[CRITICAL FIX] Set competitor to {competitor_voice} ({selected_voice_check.gender if selected_voice_check else 'unknown'})")
        else:
            st.error(f"CRITICAL: Cannot find {gender_filter} voice for {TTS_PROVIDERS[competitor_id].name}")
            st.session_state.blind_test_current_pair = None
            return
    
    # Final verification - if still wrong, abort
    if not selected_voice_check or selected_voice_check.gender != gender_filter:
        st.error(f"CRITICAL: Voice {competitor_voice} still has wrong gender. Aborting.")
        st.session_state.blind_test_current_pair = None
        return
    
    # Special handling for Cartesia providers - ensure voice is in map AND matches gender
    if competitor_id in ["cartesia_sonic2", "cartesia_turbo", "cartesia_sonic3"]:
        try:
            provider_obj = TTSProviderFactory.create_provider(competitor_id)
            if hasattr(provider_obj, 'voice_id_map'):
                voice_info_check = TTS_PROVIDERS[competitor_id].voice_info.get(competitor_voice)
                voice_in_map = competitor_voice in provider_obj.voice_id_map
                gender_matches = voice_info_check and voice_info_check.gender == gender_filter
                
                if not voice_in_map or not gender_matches:
                    # Find valid voice that is: 1) in voice_id_map, 2) matches gender, 3) in supported_voices
                    valid_voices = [
                        v for v in competitor_voices 
                        if v in provider_obj.voice_id_map and
                        TTS_PROVIDERS[competitor_id].voice_info.get(v) and
                        TTS_PROVIDERS[competitor_id].voice_info[v].gender == gender_filter
                    ]
                    if valid_voices:
                        # Use index-based selection to maintain consistency (not random)
                        voice_index = competitor_voices.index(competitor_voice) if competitor_voice in competitor_voices else 0
                        competitor_voice = valid_voices[voice_index % len(valid_voices)]
                        voice_info_check = TTS_PROVIDERS[competitor_id].voice_info.get(competitor_voice)
                        print(f"[CARTESIA FIX] Fixed {competitor_id} voice to {competitor_voice} (gender: {gender_filter})")
                    else:
                        st.error(f"No {gender_filter} voice found in Cartesia voice mapping for {TTS_PROVIDERS[competitor_id].name}")
                        st.session_state.blind_test_current_pair = None
                        return
                else:
                    print(f"[CARTESIA DEBUG] Voice {competitor_voice} is valid (in map, gender: {gender_filter})")
        except Exception as e:
            print(f"Warning: Could not validate Cartesia voice: {e}")
    
    # FINAL CRITICAL VERIFICATION: Double-check voice matches gender before generating audio
    final_competitor_info = TTS_PROVIDERS[competitor_id].voice_info.get(competitor_voice)
    if not final_competitor_info or final_competitor_info.gender != gender_filter:
        print(f"[FINAL GENDER CHECK] ERROR: Competitor voice {competitor_voice} is {final_competitor_info.gender if final_competitor_info else 'unknown'}, expected {gender_filter}")
        # Force correct gender voice
        correct_voices = [v for v in competitor_voices if TTS_PROVIDERS[competitor_id].voice_info.get(v) and TTS_PROVIDERS[competitor_id].voice_info[v].gender == gender_filter]
        if correct_voices:
            competitor_voice = correct_voices[0]
            final_competitor_info = TTS_PROVIDERS[competitor_id].voice_info.get(competitor_voice)
            print(f"[FINAL GENDER CHECK] FORCED Competitor to {competitor_voice} ({final_competitor_info.gender})")
        else:
            st.error(f"CRITICAL: Competitor has no {gender_filter} voices available")
            st.session_state.blind_test_current_pair = None
            return
    
    print(f"[FINAL VERIFICATION] ✓ Competitor voice confirmed: {competitor_voice} ({final_competitor_info.gender})")
    
    # Special handling for Sarvam - ensure voice matches gender
    if competitor_id == "sarvam" or competitor_id == "sarvam_bulbul_v3":
        voice_info = TTS_PROVIDERS[competitor_id].voice_info.get(competitor_voice)
        if not voice_info or voice_info.gender != gender_filter:
            print(f"[SARVAM DEBUG] Gender check: voice={competitor_voice}, expected={gender_filter}, got={voice_info.gender if voice_info else 'None'}")
            # Find correct gender voice
            correct_voices = [
                v for v, info in TTS_PROVIDERS[competitor_id].voice_info.items()
                if info.gender == gender_filter and v in TTS_PROVIDERS[competitor_id].supported_voices
            ]
            if correct_voices:
                competitor_voice = random.choice(correct_voices)
                print(f"[SARVAM DEBUG] Fixed gender mismatch, using {competitor_voice}")
            else:
                st.error(f"No {gender_filter} voice available for Sarvam AI")
                st.session_state.blind_test_current_pair = None
                return
    
    # FINAL STRICT GENDER VERIFICATION - Ensure competitor voice gender matches Murf voice gender
    if competitor_id in TTS_PROVIDERS:
        selected_voice_info = TTS_PROVIDERS[competitor_id].voice_info.get(competitor_voice)
        
        # Verify gender match
        if selected_voice_info and selected_voice_info.gender != gender_filter:
            print(f"[GENDER DEBUG] ERROR: Gender mismatch! Expected {gender_filter}, got {selected_voice_info.gender}")
            # Force find a voice with correct gender
            competitor_voices = [
                v for v, info in TTS_PROVIDERS[competitor_id].voice_info.items() 
                if info.gender == gender_filter and v in TTS_PROVIDERS[competitor_id].supported_voices
            ]
            if competitor_voices:
                competitor_voice = competitor_voices[0]
                selected_voice_info = TTS_PROVIDERS[competitor_id].voice_info.get(competitor_voice)
                print(f"[GENDER DEBUG] Fixed! Using {competitor_voice} ({selected_voice_info.gender if selected_voice_info else 'unknown'})")
            else:
                st.error(f"No {gender_filter} voice available for {TTS_PROVIDERS[competitor_id].name}")
                return
        elif selected_voice_info:
            print(f"[GENDER DEBUG] ✓ Verified: Competitor voice {competitor_voice} is {selected_voice_info.gender} (matches Murf's {gender_filter})")
        
        if not selected_voice_info:
            # Voice not found in voice_info, find a valid one
            competitor_voices = [
                v for v, info in TTS_PROVIDERS[competitor_id].voice_info.items() 
                if info.gender == gender_filter and v in TTS_PROVIDERS[competitor_id].supported_voices
            ]
            if competitor_voices:
                competitor_voice = random.choice(competitor_voices)
            else:
                st.error(f"Voice validation error: Could not find valid {gender_filter} voice for {TTS_PROVIDERS[competitor_id].name}")
                return
        elif selected_voice_info.gender != gender_filter:
            # Gender mismatch - this should never happen, but fix it if it does
            competitor_voices = [
                v for v, info in TTS_PROVIDERS[competitor_id].voice_info.items() 
                if info.gender == gender_filter and v in TTS_PROVIDERS[competitor_id].supported_voices
            ]
            if competitor_voices:
                competitor_voice = random.choice(competitor_voices)
            else:
                st.error(f"Gender mismatch error: Selected voice doesn't match {gender_filter} gender")
                return
    
    # IMPORTANT: Murf voice stays FIXED - use the one from session state, don't change it
    # The murf_voice is already set from the setup page and should never change during comparisons
    
    # FINAL GENDER CHECK LOG - CRITICAL VERIFICATION BEFORE GENERATION
    murf_final_info = TTS_PROVIDERS[murf_provider_id].voice_info.get(murf_voice)
    comp_final_info = TTS_PROVIDERS[competitor_id].voice_info.get(competitor_voice) if competitor_id in TTS_PROVIDERS else None
    
    # ABSOLUTE FINAL CHECK - if genders don't match, force fix it
    if murf_final_info and comp_final_info and murf_final_info.gender != comp_final_info.gender:
        print(f"[GENDER FINAL] CRITICAL ERROR: Gender mismatch detected! Murf: {murf_final_info.gender}, Competitor: {comp_final_info.gender}")
        # Force find correct gender voice
        correct_gender_voices = [
            v for v, info in TTS_PROVIDERS[competitor_id].voice_info.items()
            if info.gender == murf_final_info.gender and v in TTS_PROVIDERS[competitor_id].supported_voices
        ]
        if correct_gender_voices:
            competitor_voice = random.choice(correct_gender_voices)
            comp_final_info = TTS_PROVIDERS[competitor_id].voice_info.get(competitor_voice)
            print(f"[GENDER FINAL] FORCED FIX: Changed competitor voice to {competitor_voice} (gender: {comp_final_info.gender if comp_final_info else '?'})")
        else:
            st.error(f"CRITICAL: Cannot find {murf_final_info.gender} voice for {TTS_PROVIDERS[competitor_id].name}. Cannot proceed.")
            st.session_state.blind_test_current_pair = None
            return
    
    # ABSOLUTE FINAL VERIFICATION RIGHT BEFORE AUDIO GENERATION (for Unranked Blind Test)
    # CRITICAL: For Cartesia, we MUST verify voice is in voice_id_map AND matches gender
    pre_gen_comp_info_vs = TTS_PROVIDERS[competitor_id].voice_info.get(competitor_voice)
    
    if competitor_id in ["cartesia_sonic2", "cartesia_turbo", "cartesia_sonic3"]:
        try:
            provider_obj_comp_vs = TTSProviderFactory.create_provider(competitor_id)
            if hasattr(provider_obj_comp_vs, 'voice_id_map'):
                # For Cartesia, voice MUST be in voice_id_map AND match gender
                if competitor_voice not in provider_obj_comp_vs.voice_id_map or not pre_gen_comp_info_vs or pre_gen_comp_info_vs.gender != gender_filter:
                    print(f"[CARTESIA PRE-GEN] Competitor voice {competitor_voice} invalid! Getting fresh Cartesia list...")
                    fresh_comp_cartesia_vs = get_voices_by_gender(competitor_id, gender_filter)
                    if competitor_id in TTS_PROVIDERS:
                        supported_set = set(TTS_PROVIDERS[competitor_id].supported_voices)
                        # CRITICAL: Only use voices that are in voice_id_map AND match gender
                        fresh_comp_cartesia_vs = [
                            v for v in fresh_comp_cartesia_vs 
                            if v in supported_set and 
                            v in provider_obj_comp_vs.voice_id_map and
                            TTS_PROVIDERS[competitor_id].voice_info.get(v) and
                            TTS_PROVIDERS[competitor_id].voice_info[v].gender == gender_filter
                        ]
                    if fresh_comp_cartesia_vs:
                        competitor_voice = fresh_comp_cartesia_vs[comparison_index % len(fresh_comp_cartesia_vs)]
                        pre_gen_comp_info_vs = TTS_PROVIDERS[competitor_id].voice_info.get(competitor_voice)
                        print(f"[CARTESIA PRE-GEN FIX] Competitor set to {competitor_voice} ({pre_gen_comp_info_vs.gender if pre_gen_comp_info_vs else 'unknown'})")
        except Exception as e:
            print(f"Warning: Could not validate Cartesia voice: {e}")
    
    if not pre_gen_comp_info_vs or pre_gen_comp_info_vs.gender != gender_filter:
        print(f"[PRE-GEN FIX] Competitor voice {competitor_voice} wrong gender! Getting fresh list...")
        fresh_comp_final_vs = get_voices_by_gender(competitor_id, gender_filter)
        if competitor_id in TTS_PROVIDERS:
            supported_set = set(TTS_PROVIDERS[competitor_id].supported_voices)
            fresh_comp_final_vs = [v for v in fresh_comp_final_vs if v in supported_set]
        if fresh_comp_final_vs:
            competitor_voice = fresh_comp_final_vs[comparison_index % len(fresh_comp_final_vs)]
            pre_gen_comp_info_vs = TTS_PROVIDERS[competitor_id].voice_info.get(competitor_voice)
            print(f"[PRE-GEN FIX] Competitor set to {competitor_voice} ({pre_gen_comp_info_vs.gender if pre_gen_comp_info_vs else 'unknown'})")
    
    # Final abort check - if still wrong, don't generate
    if not pre_gen_comp_info_vs or pre_gen_comp_info_vs.gender != gender_filter:
        st.error(f"CRITICAL: Cannot find {gender_filter} voice for Competitor. Aborting.")
        st.session_state.blind_test_current_pair = None
        return
    
    print(f"[GENDER FINAL] ✓ Verified: Murf: {murf_voice} ({murf_final_info.gender if murf_final_info else '?'}) vs Competitor: {competitor_voice} ({pre_gen_comp_info_vs.gender if pre_gen_comp_info_vs else '?'})")
    
    # Additional safety check - this should never trigger if above logic works correctly
    if murf_final_info and pre_gen_comp_info_vs and murf_final_info.gender != pre_gen_comp_info_vs.gender:
        print(f"[GENDER FINAL] ❌ CRITICAL ERROR: Gender mismatch detected! Aborting.")
        st.error(f"Gender mismatch: Murf ({murf_final_info.gender}) vs Competitor ({pre_gen_comp_info_vs.gender})")
        st.session_state.blind_test_current_pair = None
        return
    else:
        print(f"[GENDER FINAL] ✓ Gender match confirmed: {murf_final_info.gender if murf_final_info else gender_filter}")
    
    # Show loading placeholder while generating
    loading_placeholder = st.empty()
    loading_placeholder.info(f"Generating audio samples... This may take a few seconds.")
    
    # Generate samples (runs in parallel for speed)
    # IMPORTANT: Log the text being used to ensure it's correct
    print(f"[DEBUG] Generating audio for text: '{text[:50]}...' (Comparison #{st.session_state.blind_test_comparison_count})")
    
    try:
        sample_a, sample_b = asyncio.run(generate_comparison_samples(
            text, murf_provider_id, murf_voice, competitor_id, competitor_voice
        ))
        
        # Validate that samples were generated with the correct text
        if sample_a and hasattr(sample_a, 'metadata') and sample_a.metadata:
            generated_text_a = sample_a.metadata.get('text', '')
            if generated_text_a and generated_text_a != text:
                print(f"[WARNING] Sample A text mismatch! Expected: '{text[:50]}', Got: '{generated_text_a[:50]}'")
        
        if sample_b and hasattr(sample_b, 'metadata') and sample_b.metadata:
            generated_text_b = sample_b.metadata.get('text', '')
            if generated_text_b and generated_text_b != text:
                print(f"[WARNING] Sample B text mismatch! Expected: '{text[:50]}', Got: '{generated_text_b[:50]}'")
                
    except Exception as e:
        loading_placeholder.error(f"Error generating samples: {str(e)}")
        sample_a, sample_b = None, None
    
    # Clear loading message
    loading_placeholder.empty()
    
    # Check if either sample failed
    if sample_a is None or sample_b is None or not (sample_a.success if sample_a else False) or not (sample_b.success if sample_b else False):
        # Show detailed error and allow retry
        error_msg = "Both samples failed to generate."
        error_details = []
        
        if sample_a:
            if not sample_a.success:
                error_details.append(f"Sample A ({TTS_PROVIDERS[murf_provider_id].name}): {sample_a.error_message if hasattr(sample_a, 'error_message') and sample_a.error_message else 'Unknown error'}")
        else:
            error_details.append(f"Sample A ({TTS_PROVIDERS[murf_provider_id].name}): Failed to generate")
            
        if sample_b:
            if not sample_b.success:
                error_details.append(f"Sample B ({TTS_PROVIDERS[competitor_id].name}): {sample_b.error_message if hasattr(sample_b, 'error_message') and sample_b.error_message else 'Unknown error'}")
        else:
            error_details.append(f"Sample B ({TTS_PROVIDERS[competitor_id].name}): Failed to generate")
        
        if error_details:
            error_msg = " | ".join(error_details)
        
        st.session_state.blind_test_current_pair = {"error": True, "message": error_msg}
        st.rerun()
        return
    
    # CRITICAL FIX: Validate text matches before storing
    # Ensure samples were generated with the correct text
    if sample_a and sample_a.success:
        # Store text in metadata for validation
        if not hasattr(sample_a, 'metadata') or sample_a.metadata is None:
            sample_a.metadata = {}
        sample_a.metadata['generated_text'] = text
        sample_a.metadata['comparison_num'] = st.session_state.blind_test_comparison_count
    
    if sample_b and sample_b.success:
        # Store text in metadata for validation
        if not hasattr(sample_b, 'metadata') or sample_b.metadata is None:
            sample_b.metadata = {}
        sample_b.metadata['generated_text'] = text
        sample_b.metadata['comparison_num'] = st.session_state.blind_test_comparison_count
    
    # Randomize order (50/50 chance Murf is A or B)
    murf_is_a = random.random() > 0.5
    
    # Generate a unique timestamp for this comparison to prevent caching
    import time
    unique_timestamp = time.time()
    comparison_id = f"{st.session_state.blind_test_comparison_count}_{unique_timestamp}"
    
    # Log for debugging
    print(f"[DEBUG] Storing pair - Text: '{text[:60]}...', Comparison #{st.session_state.blind_test_comparison_count}")
    
    if murf_is_a:
        st.session_state.blind_test_current_pair = {
            "sample_a": sample_a, "sample_b": sample_b,
            "provider_a": murf_provider_id, "provider_b": competitor_id,
            "voice_a": murf_voice, "voice_b": competitor_voice,
            "text": text, "murf_is": "A", "generated_at": unique_timestamp,
            "comparison_id": comparison_id
        }
    else:
        st.session_state.blind_test_current_pair = {
            "sample_a": sample_b, "sample_b": sample_a,
            "provider_a": competitor_id, "provider_b": murf_provider_id,
            "voice_a": competitor_voice, "voice_b": murf_voice,
            "text": text, "murf_is": "B", "generated_at": unique_timestamp,
            "comparison_id": comparison_id
        }
    
    st.session_state.blind_test_audio_played = {"A": 0, "B": 0}
    st.rerun()


async def generate_comparison_samples(text: str, provider_a: str, voice_a: str, provider_b: str, voice_b: str):
    """Generate audio samples for comparison - runs both in parallel for speed"""
    
    # Validate voices exist in supported voices
    if provider_a in TTS_PROVIDERS and voice_a not in TTS_PROVIDERS[provider_a].supported_voices:
        print(f"Warning: Voice '{voice_a}' not in supported voices for {provider_a}. Available: {TTS_PROVIDERS[provider_a].supported_voices}")
    
    if provider_b in TTS_PROVIDERS and voice_b not in TTS_PROVIDERS[provider_b].supported_voices:
        print(f"Warning: Voice '{voice_b}' not in supported voices for {provider_b}. Available: {TTS_PROVIDERS[provider_b].supported_voices}")
    
    # CRITICAL: Create TestSample with unique ID and ensure text is set correctly
    import time
    unique_sample_id = f"blind_comparison_{int(time.time() * 1000)}"
    
    sample = TestSample(
        id=unique_sample_id,
        text=text,  # CRITICAL: This text MUST be used for generation
        word_count=len(text.split()),
        category="blind_test",
        length_category="custom",
        complexity_score=0.5
    )
    
    # CRITICAL DEBUG: Verify text is correct
    print(f"[CRITICAL DEBUG] TestSample created with text: '{sample.text[:80]}...'")
    assert sample.text == text, f"Text mismatch! Expected '{text[:50]}', got '{sample.text[:50]}'"
    
    async def generate_sample(provider_id: str, voice: str):
        """Generate a single sample"""
        try:
            # CRITICAL DEBUG: Log the text being used for generation
            print(f"[CRITICAL DEBUG] Generating for {provider_id} with text: '{sample.text[:60]}...'")
            
            provider_obj = TTSProviderFactory.create_provider(provider_id)
            result = await st.session_state.benchmark_engine.run_single_test(
                provider_obj, sample, voice
            )
            
            # CRITICAL: Store the text in result metadata immediately
            if result:
                if not hasattr(result, 'metadata') or result.metadata is None:
                    result.metadata = {}
                result.metadata['text'] = sample.text
                result.metadata['generated_text'] = sample.text
                print(f"[CRITICAL DEBUG] Generated result for {provider_id} - text in metadata: '{result.metadata.get('text', '')[:60]}...'")
            
            if not result.success:
                print(f"Sample generation failed for {provider_id} with voice {voice}: {result.error_message if hasattr(result, 'error_message') else 'Unknown error'}")
            return result
        except Exception as e:
            print(f"Error generating sample ({provider_id} with voice {voice}): {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # Run both generations in parallel for faster loading
    results = await asyncio.gather(
        generate_sample(provider_a, voice_a),
        generate_sample(provider_b, voice_b),
        return_exceptions=True
    )
    
    result_a = results[0] if not isinstance(results[0], Exception) else None
    result_b = results[1] if not isinstance(results[1], Exception) else None
    
    return result_a, result_b

def display_fvs_setup():
    """Display the Falcon vs Zeroshot setup page"""
    from config import get_voices_by_gender, get_voice_gender
    from dataset import DatasetGenerator
    import random
    
    # Initialize filters if not set
    if "fvs_gender_filter" not in st.session_state:
        st.session_state.fvs_gender_filter = "female"
    if "fvs_locale_filter" not in st.session_state:
        st.session_state.fvs_locale_filter = None
    
    # Extract all available locales from Murf voices
    all_locales = set()
    for voice_id in TTS_PROVIDERS["murf_falcon_oct23"].supported_voices:
        parts = voice_id.split("-")
        if len(parts) >= 2:
            locale = f"{parts[0]}-{parts[1]}"
            all_locales.add(locale)
    
    locale_options = ["All Locales"] + sorted(list(all_locales))
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**1. Select Locale**")
        selected_locale_display = st.selectbox(
            "Choose locale:",
            locale_options,
            index=0 if st.session_state.fvs_locale_filter is None else (locale_options.index(st.session_state.fvs_locale_filter) if st.session_state.fvs_locale_filter in locale_options else 0),
            key="fvs_locale_select"
        )
        
        if selected_locale_display == "All Locales":
            st.session_state.fvs_locale_filter = None
        else:
            st.session_state.fvs_locale_filter = selected_locale_display
        
        # Show available voices count for selected locale and gender
        falcon_voice_info = TTS_PROVIDERS["murf_falcon_oct23"].voice_info
        zeroshot_voice_info = TTS_PROVIDERS["murf_zeroshot"].voice_info
        gender = st.session_state.fvs_gender_filter
        locale = st.session_state.fvs_locale_filter
        
        falcon_count = 0
        zeroshot_count = 0
        for v, info in falcon_voice_info.items():
            if info.gender == gender:
                if locale is None:
                    falcon_count += 1
                else:
                    voice_locale = "-".join(v.split("-")[:2])
                    if voice_locale == locale:
                        falcon_count += 1
        
        for v, info in zeroshot_voice_info.items():
            if info.gender == gender:
                if locale is None:
                    zeroshot_count += 1
                else:
                    voice_locale = "-".join(v.split("-")[:2])
                    if voice_locale == locale:
                        zeroshot_count += 1
        
        st.caption(f"Available voices: Falcon ({falcon_count}), Zeroshot ({zeroshot_count})")
    
    with col2:
        st.markdown("**2. Select Gender**")
        selected_gender_radio = st.radio(
            "**Gender:**",
            ["Male", "Female"],
            index=0 if st.session_state.fvs_gender_filter == "female" else 1,
            horizontal=True,
            key="fvs_gender_radio"
        )
        
        new_gender = selected_gender_radio.lower()
        if new_gender != st.session_state.fvs_gender_filter:
            st.session_state.fvs_gender_filter = new_gender
            st.rerun()
    
    with col3:
        st.markdown("**3. Number of Comparisons**")
        # Use a different key for the slider to avoid conflicts, then sync to session state
        # This pattern matches ranked blind test and prevents reset on gender change
        max_comparisons = st.slider(
            "Comparisons:",
            min_value=5,
            max_value=50,
            value=st.session_state.fvs_max_comparisons,
            step=5,
            help="How many head-to-head comparisons to run",
            key="fvs_max_comparisons_slider"
        )
        # Sync the slider value to session state (preserves value across reruns)
        st.session_state.fvs_max_comparisons = max_comparisons
        st.caption(f"Will run {st.session_state.fvs_max_comparisons} comparisons")
    
    st.divider()
    
    # Sentence selection
    st.markdown("**3. Upload Test Sentences** (System will pick randomly)")
    
    sentences_text = st.text_area(
        "Enter sentences (one per line):",
        value="""The quick brown fox jumps over the lazy dog.
The wine glass fills again and laughter breaks through the pressure that had been building quietly for hours.
Just to confirm, the co-applicant's name is spelled M-A-R-I-S-A, correct?
Scientists have made a groundbreaking discovery that could revolutionize renewable energy.
Hello, how can I assist you today with your account inquiry?""",
        height=200,
        help="Enter multiple sentences, one per line. The system will randomly select sentences for each test. Supports all languages including Hindi, Chinese, etc.",
        key="fvs_sentences_text"
    )
    
    # Parse sentences - handle Unicode properly (supports Hindi, Chinese, etc.)
    # Split by newlines and filter out empty lines
    # Streamlit text_area returns strings, so we just need to ensure proper Unicode handling
    sentences = [s.strip() for s in sentences_text.strip().split('\n') if s.strip()]
    
    # Debug: Log sentence count and first sentence (for troubleshooting)
    if sentences:
        print(f"[FVS DEBUG] Parsed {len(sentences)} sentences. First sentence: '{sentences[0][:50]}...'")
    
    # Check if sentences have changed - if so, clear current pair to force regeneration
    if "fvs_sentences_hash" not in st.session_state:
        st.session_state.fvs_sentences_hash = None
    
    import hashlib
    # Hash sentences with proper UTF-8 encoding to handle all languages (Hindi, Chinese, etc.)
    sentences_hash = hashlib.md5(str(sorted(sentences)).encode('utf-8')).hexdigest()
    
    # If sentences changed and there's a current pair, clear it
    if (st.session_state.fvs_sentences_hash is not None and 
        st.session_state.fvs_sentences_hash != sentences_hash and
        st.session_state.fvs_current_pair is not None):
        st.session_state.fvs_current_pair = None
        st.session_state.fvs_comparison_count = 0
        st.session_state.fvs_results_history = []
        st.session_state.used_sentences_fvs = []  # Reset used sentences when sentences change
    
    st.session_state.fvs_sentences = sentences
    st.session_state.fvs_sentences_hash = sentences_hash
    
    if sentences:
        st.caption(f"📝 {len(sentences)} sentences loaded")
    else:
        st.warning("Please enter at least one sentence")
    
    can_start = len(st.session_state.fvs_sentences) >= 1
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Start Comparison", type="primary", disabled=not can_start, key="fvs_start"):
            # The slider with key="fvs_max_comparisons" automatically updates session state
            # No need to manually set it - just use the value that's already there
            st.session_state.fvs_setup_complete = True
            st.session_state.fvs_comparison_count = 0
            st.session_state.fvs_results_history = []
            st.session_state.fvs_current_pair = None
            st.session_state.used_sentences_fvs = []  # Reset used sentences for new test
            st.session_state.fvs_test_start_time = datetime.now()  # Store test start timestamp
            st.rerun()
    
    if not can_start:
        st.caption("Ensure sentences are loaded")

def display_fvs_comparison():
    """Display the Falcon vs Zeroshot comparison"""
    # Check if we should show final results
    if st.session_state.get("fvs_show_final_results", False):
        st.session_state.fvs_current_pair = None
        display_fvs_final_results()
        return
    
    # Check if we need to generate a new pair
    force_regen = st.session_state.get("force_regenerate_fvs", False)
    if st.session_state.fvs_current_pair is None or force_regen:
        st.session_state.force_regenerate_fvs = False
        if "fvs_audio_played" in st.session_state:
            st.session_state.fvs_audio_played = {"A": 0, "B": 0}
        st.session_state.fvs_current_pair = None
        generate_fvs_comparison()
        return
    
    pair = st.session_state.fvs_current_pair
    
    if pair is None or pair.get("error"):
        error_msg = pair.get("message", "Failed to generate comparison.") if pair else "Failed to generate comparison."
        st.error(f"⚠️ {error_msg}")
        if st.button("Retry", type="primary", key="fvs_retry"):
            st.session_state.fvs_current_pair = None
            st.rerun()
        return
    
    # Progress indicator
    progress = st.session_state.fvs_comparison_count / st.session_state.fvs_max_comparisons
    st.progress(progress)
    st.caption(f"Comparison {st.session_state.fvs_comparison_count + 1} of {st.session_state.fvs_max_comparisons}")
    
    # Display the prompt/sentence - sleek gray design (same as ranked blind test)
    st.markdown(f"""
    <div style="background: #f5f5f5; padding: 12px 16px; border-radius: 8px; margin: 8px 0;">
        <span style="color: #666; font-size: 0.85em; font-weight: 500;">PROMPT</span>
        <p style="color: #333; font-size: 1em; margin: 4px 0 0 0; line-height: 1.5;">{pair['text']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='color: #888; font-size: 0.9em; margin: 16px 0 8px 0;'>Vote to reveal your model preference</p>", unsafe_allow_html=True)
    
    sample_a = pair["sample_a"]
    sample_b = pair["sample_b"]
    
    # Generate unique comparison key for audio players
    comparison_key = f"fvs_{st.session_state.fvs_comparison_count}_{pair.get('comparison_id', '')}"
    
    # Audio players side by side
    col1, col2 = st.columns(2)
    
    with col1:
        display_audio_player(sample_a, "A", "left", comparison_key)
        # Add spacing and center the button
        st.markdown('<div style="margin-top: 16px;"></div>', unsafe_allow_html=True)
        button_col1, button_col2, button_col3 = st.columns([1, 2, 1])
        with button_col2:
            if st.button("Vote A", type="primary", key="fvs_vote_a", use_container_width=True):
                handle_fvs_vote("A", pair)
    
    with col2:
        display_audio_player(sample_b, "B", "right", comparison_key)
        # Add spacing and center the button
        st.markdown('<div style="margin-top: 16px;"></div>', unsafe_allow_html=True)
        button_col1, button_col2, button_col3 = st.columns([1, 2, 1])
        with button_col2:
            if st.button("Vote B", type="primary", key="fvs_vote_b", use_container_width=True):
                handle_fvs_vote("B", pair)
    
    st.divider()
    
    # Comment input field (optional)
    comment_key = f"comment_fvs_{st.session_state.fvs_comparison_count}"
    if comment_key not in st.session_state:
        st.session_state[comment_key] = ""
    
    comment = st.text_area(
        "Add a comment (optional)",
        value=st.session_state.get(comment_key, ""),
        key=comment_key,
        placeholder="e.g., Pronunciation issue on 'X', unnatural pause, quality difference...",
        height=80,
        help="Add notes about this comparison to help identify actionable feedback"
    )
    
    # Action button - End Test only (centered, medium size)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("End Test", key="fvs_end_test", use_container_width=True, type="secondary"):
            st.session_state.fvs_show_final_results = True
            st.rerun()

def generate_fvs_comparison():
    """Generate the next Falcon vs Zeroshot comparison"""
    from config import get_voices_by_gender
    import random
    import time
    
    # Check if we've reached max comparisons
    if st.session_state.fvs_comparison_count >= st.session_state.fvs_max_comparisons:
        st.session_state.fvs_show_final_results = True
        st.rerun()
        return
    
    # Get gender filter
    gender_filter = st.session_state.fvs_gender_filter
    comparison_index = st.session_state.fvs_comparison_count
    locale_filter = st.session_state.fvs_locale_filter
    
    # Get a random sentence
    sentences = st.session_state.fvs_sentences
    if not sentences:
        st.error("No sentences available")
        return
    
    if "used_sentences_fvs" not in st.session_state:
        st.session_state.used_sentences_fvs = []
    
    available_sentences = [s for s in sentences if s not in st.session_state.used_sentences_fvs]
    if not available_sentences:
        st.session_state.used_sentences_fvs = []
        available_sentences = sentences
    
    text = random.choice(available_sentences)
    st.session_state.used_sentences_fvs.append(text)
    
    # Get all voices matching locale and gender - use same voice for both providers
    falcon_voice_info = TTS_PROVIDERS["murf_falcon_oct23"].voice_info
    
    # Get all voices matching gender and locale (same for both Falcon and Zeroshot)
    filtered_voices = []
    for v, info in falcon_voice_info.items():
        if info.gender == gender_filter:
            if locale_filter is None:
                filtered_voices.append(v)
            else:
                # Check if voice matches locale
                voice_locale = "-".join(v.split("-")[:2])
                if voice_locale == locale_filter:
                    filtered_voices.append(v)
    
    if not filtered_voices:
        st.error(f"No voices found for {gender_filter} gender and locale {locale_filter if locale_filter else 'All'}")
        return
    
    # Shuffle voices for variety (only once since both providers use same voices)
    random.shuffle(filtered_voices)
    
    # Cycle through voices using comparison index - use same voice for both providers
    voice_index = comparison_index % len(filtered_voices)
    
    # Use the same voice ID for both Falcon and Zeroshot
    falcon_voice = filtered_voices[voice_index]
    zeroshot_voice = filtered_voices[voice_index]
    
    # Generate samples
    sample_a, sample_b = asyncio.run(generate_comparison_samples(
        text, "murf_falcon_oct23", falcon_voice, "murf_zeroshot", zeroshot_voice
    ))
    
    if not sample_a or not sample_a.success:
        st.session_state.fvs_current_pair = {
            "error": True,
            "message": f"Failed to generate Falcon sample: {sample_a.error_message if sample_a else 'Unknown error'}"
        }
        return
    
    if not sample_b or not sample_b.success:
        st.session_state.fvs_current_pair = {
            "error": True,
            "message": f"Failed to generate Zeroshot sample: {sample_b.error_message if sample_b else 'Unknown error'}"
        }
        return
    
    # Randomize order
    falcon_is_a = random.random() > 0.5
    unique_timestamp = time.time()
    comparison_id = f"{st.session_state.fvs_comparison_count}_{unique_timestamp}"
    
    if falcon_is_a:
        st.session_state.fvs_current_pair = {
            "sample_a": sample_a, "sample_b": sample_b,
            "provider_a": "murf_falcon_oct23", "provider_b": "murf_zeroshot",
            "voice_a": falcon_voice, "voice_b": zeroshot_voice,
            "text": text, "falcon_is": "A", "generated_at": unique_timestamp,
            "comparison_id": comparison_id
        }
    else:
        st.session_state.fvs_current_pair = {
            "sample_a": sample_b, "sample_b": sample_a,
            "provider_a": "murf_zeroshot", "provider_b": "murf_falcon_oct23",
            "voice_a": zeroshot_voice, "voice_b": falcon_voice,
            "text": text, "falcon_is": "B", "generated_at": unique_timestamp,
            "comparison_id": comparison_id
        }
    
    st.session_state.fvs_audio_played = {"A": 0, "B": 0}
    st.rerun()

def handle_fvs_vote(choice: str, pair: dict):
    """Handle a vote for Falcon vs Zeroshot - does NOT update ELO/leaderboard"""
    # Prevent double voting
    current_comparison = st.session_state.fvs_comparison_count
    if st.session_state.get("last_voted_comparison_fvs") == current_comparison:
        return
    
    st.session_state.last_voted_comparison_fvs = current_comparison
    st.session_state.fvs_current_pair = None
    st.session_state.force_regenerate_fvs = True
    
    # Determine winner and loser
    if choice == "A":
        winner_provider = pair["provider_a"]
        loser_provider = pair["provider_b"]
        winner_voice = pair["voice_a"]
        loser_voice = pair["voice_b"]
    else:
        winner_provider = pair["provider_b"]
        loser_provider = pair["provider_a"]
        winner_voice = pair["voice_b"]
        loser_voice = pair["voice_a"]
    
    print(f"FVS Vote: {choice} | Winner: {winner_provider} | Loser: {loser_provider}")
    
    # Get comment for this comparison
    comment_key = f"comment_fvs_{current_comparison}"
    comment = st.session_state.get(comment_key, "")
    
    # De-anonymize comment: replace "Sample A" and "Sample B" with actual provider names
    provider_a_id = pair.get("provider_a", "")
    provider_b_id = pair.get("provider_b", "")
    comment = de_anonymize_comment(comment, provider_a_id, provider_b_id)
    
    # Extract API configuration from samples
    sample_a = pair.get("sample_a")
    sample_b = pair.get("sample_b")
    
    # Get API configs from metadata - map to provider_a and provider_b
    config_provider_a = {}
    config_provider_b = {}
    
    if sample_a and hasattr(sample_a, 'metadata') and sample_a.metadata:
        # Get endpoint URL from config for Murf providers
        endpoint_url_a = ""
        if provider_a_id in TTS_PROVIDERS:
            endpoint_url_a = TTS_PROVIDERS[provider_a_id].base_url
        
        config_provider_a = {
            "provider": provider_a_id,
            "voice": pair.get("voice_a", ""),
            "model": sample_a.metadata.get("model", ""),
            "format": sample_a.metadata.get("format", ""),
            "sample_rate": sample_a.metadata.get("sample_rate", ""),
            "endpoint_url": endpoint_url_a
        }
    
    if sample_b and hasattr(sample_b, 'metadata') and sample_b.metadata:
        # Get endpoint URL from config for Murf providers
        endpoint_url_b = ""
        if provider_b_id in TTS_PROVIDERS:
            endpoint_url_b = TTS_PROVIDERS[provider_b_id].base_url
        
        config_provider_b = {
            "provider": provider_b_id,
            "voice": pair.get("voice_b", ""),
            "model": sample_b.metadata.get("model", ""),
            "format": sample_b.metadata.get("format", ""),
            "sample_rate": sample_b.metadata.get("sample_rate", ""),
            "endpoint_url": endpoint_url_b
        }
    
    # Determine winner and loser configs based on choice
    if choice == "A":
        winner_config = config_provider_a
        loser_config = config_provider_b
    else:
        winner_config = config_provider_b
        loser_config = config_provider_a
    
    # Get test start timestamp (store it when test starts if not already stored)
    if "fvs_test_start_time" not in st.session_state:
        st.session_state.fvs_test_start_time = datetime.now()
    test_timestamp = st.session_state.fvs_test_start_time
    
    # Extract locale from voice names
    winner_voice_parts = winner_voice.split("-") if winner_voice else []
    locale = "-".join(winner_voice_parts[:2]) if len(winner_voice_parts) >= 2 else "Unknown"
    
    # Save to database for persistence (like leaderboard)
    metadata = {
        "vote_source": "falcon_vs_zeroshot",
        "winner_voice": winner_voice,
        "loser_voice": loser_voice,
        "locale": locale,
        "text": pair["text"],
        "comment": comment,
        "winner_config": winner_config,
        "loser_config": loser_config,
        "falcon_won": (pair["falcon_is"] == choice),
        "user_choice": choice,
        "comparison_num": current_comparison + 1
    }
    
    db.save_user_vote(
        winner_provider,
        loser_provider,
        pair["text"][:100],  # text_sample field
        session_id=f"fvs_{current_comparison}",
        vote_type="falcon_vs_zeroshot",
        metadata=metadata
    )
    
    # Record result (NO ELO UPDATE - this is the key difference)
    result_record = {
        "comparison_num": current_comparison + 1,
        "winner": winner_provider,
        "winner_voice": winner_voice,
        "loser": loser_provider,
        "loser_voice": loser_voice,
        "text": pair["text"],
        "falcon_won": (pair["falcon_is"] == choice),
        "user_choice": choice,
        "comment": comment,
        "winner_config": winner_config,
        "loser_config": loser_config,
        "test_timestamp": test_timestamp,
        "locale": locale
    }
    st.session_state.fvs_results_history.append(result_record)
    
    # Clear comment from session state after saving
    if comment_key in st.session_state:
        del st.session_state[comment_key]
    
    # Move to next comparison
    st.session_state.fvs_comparison_count += 1
    
    # Check if done
    if st.session_state.fvs_comparison_count >= st.session_state.fvs_max_comparisons:
        st.session_state.fvs_show_final_results = True
        st.session_state.fvs_current_pair = None
    else:
        st.session_state.fvs_current_pair = None
        if "fvs_audio_played" in st.session_state:
            st.session_state.fvs_audio_played = {"A": 0, "B": 0}
    
    st.rerun()

def display_fvs_final_results():
    """Display final results for Falcon vs Zeroshot"""
    results = st.session_state.fvs_results_history
    
    if not results:
        st.info("No results to display")
        return
    
    st.subheader("Results")
    
    falcon_wins = sum(1 for r in results if r["winner"] == "murf_falcon_oct23")
    zeroshot_wins = sum(1 for r in results if r["winner"] == "murf_zeroshot")
    total = len(results)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Murf Falcon Wins", falcon_wins, f"{falcon_wins/total*100:.1f}%")
    with col2:
        st.metric("Murf Zeroshot Wins", zeroshot_wins, f"{zeroshot_wins/total*100:.1f}%")
    
    # Results table with comments, endpoint URLs, model names, and timestamp
    st.subheader("All Comparisons")
    comparison_data = []
    for result in results:
        # De-anonymize comment if needed (for backward compatibility)
        comment = de_anonymize_comment_from_result(result)
        
        # Get endpoint URLs and model names from configs
        winner_config = result.get("winner_config", {})
        loser_config = result.get("loser_config", {})
        
        winner_endpoint = winner_config.get("endpoint_url", "") or "-"
        winner_model = winner_config.get("model", "") or "-"
        loser_endpoint = loser_config.get("endpoint_url", "") or "-"
        loser_model = loser_config.get("model", "") or "-"
        
        # Format timestamp
        test_timestamp = result.get("test_timestamp")
        if test_timestamp:
            if isinstance(test_timestamp, str):
                timestamp_str = test_timestamp
            else:
                timestamp_str = test_timestamp.strftime("%Y-%m-%d %H:%M:%S")
        else:
            timestamp_str = "-"
        
        comparison_data.append({
            "Comparison": result["comparison_num"],
            "Winner": "Murf Falcon" if result["winner"] == "murf_falcon_oct23" else "Murf Zeroshot",
            "Winner Voice": result["winner_voice"],
            "Loser": "Murf Falcon" if result["loser"] == "murf_falcon_oct23" else "Murf Zeroshot",
            "Loser Voice": result["loser_voice"],
            "Text": result["text"],
            "Comment": comment if comment else '-',
            "Winner Endpoint": winner_endpoint,
            "Winner Model": winner_model,
            "Loser Endpoint": loser_endpoint,
            "Loser Model": loser_model,
            "Test Timestamp": timestamp_str
        })
    
    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    if st.button("Start New Test", type="primary", key="fvs_new_test"):
        # Reset all state
        st.session_state.fvs_setup_complete = False
        st.session_state.fvs_comparison_count = 0
        st.session_state.fvs_results_history = []
        st.session_state.fvs_current_pair = None
        st.session_state.fvs_show_final_results = False
        st.session_state.used_sentences_fvs = []
        if "fvs_test_start_time" in st.session_state:
            del st.session_state.fvs_test_start_time
        st.rerun()


def handle_vote_2(choice: str, pair: dict):
    """Handle a user vote from blind test 2 and update ELO ratings
    
    IMPORTANT: ELO ratings should ONLY be updated from blind test votes (user preferences),
    NOT from quick test results (latency, TTFB, etc.). Quick test is for technical metrics only.
    """
    from database import db
    
    # Prevent double voting on same comparison
    current_comparison = st.session_state.blind_test_2_comparison_count
    if st.session_state.get("last_voted_comparison_2") == current_comparison:
        return  # Already voted on this comparison
    
    # Mark this comparison as voted
    st.session_state.last_voted_comparison_2 = current_comparison
    
    # CRITICAL FIX: Clear ALL audio-related state to force fresh generation
    st.session_state.blind_test_2_current_pair = None
    
    # Force the next comparison to generate fresh audio
    st.session_state.force_regenerate_2 = True
    
    print(f"[VOTE DEBUG] Vote recorded. Cleared pair. Next comparison will generate fresh audio.")
    
    # Determine winner and loser based on choice
    # IMPORTANT: choice "A" means sample A won, choice "B" means sample B won
    if choice == "A":
        winner_provider = pair["provider_a"]
        loser_provider = pair["provider_b"]
        winner_voice = pair["voice_a"]
        loser_voice = pair["voice_b"]
    else:  # choice == "B"
        winner_provider = pair["provider_b"]
        loser_provider = pair["provider_a"]
        winner_voice = pair["voice_b"]
        loser_voice = pair["voice_a"]
    
    # Debug: Print winner/loser to verify
    print(f"Vote: {choice} | Winner: {winner_provider} | Loser: {loser_provider}")
    
    comment_key = f"comment_2_{current_comparison}"
    comment_raw = st.session_state.get(comment_key, "")
    provider_a_id = pair.get("provider_a", "")
    provider_b_id = pair.get("provider_b", "")
    comment = de_anonymize_comment(comment_raw, provider_a_id, provider_b_id)
    
    sample_a = pair.get("sample_a")
    sample_b = pair.get("sample_b")
    config_provider_a = {}
    config_provider_b = {}
    if sample_a and hasattr(sample_a, 'metadata') and sample_a.metadata:
        config_provider_a = {
            "provider": pair.get("provider_a", ""),
            "voice": pair.get("voice_a", ""),
            "model": sample_a.metadata.get("model", ""),
            "format": sample_a.metadata.get("format", ""),
            "sample_rate": sample_a.metadata.get("sample_rate", "")
        }
    if sample_b and hasattr(sample_b, 'metadata') and sample_b.metadata:
        config_provider_b = {
            "provider": pair.get("provider_b", ""),
            "voice": pair.get("voice_b", ""),
            "model": sample_b.metadata.get("model", ""),
            "format": sample_b.metadata.get("format", ""),
            "sample_rate": sample_b.metadata.get("sample_rate", "")
        }
    if choice == "A":
        winner_config = config_provider_a
        loser_config = config_provider_b
    else:
        winner_config = config_provider_b
        loser_config = config_provider_a
    
    vote_metadata = {
        "ranked_blind_test": True,
        "comment": comment,
        "text": pair["text"],
        "winner_voice": winner_voice,
        "loser_voice": loser_voice,
        "user_choice": choice,
        "winner_config": winner_config,
        "loser_config": loser_config,
        "locale": st.session_state.get("blind_test_2_locale_filter"),
    }
    
    try:
        new_winner_rating, new_loser_rating = db.update_elo_ratings(
            winner_provider, loser_provider, k_factor=32
        )
        print(f"ELO Updated - Winner ({winner_provider}): {new_winner_rating:.1f}, Loser ({loser_provider}): {new_loser_rating:.1f}")
        db.save_user_vote(
            winner_provider,
            loser_provider,
            pair["text"][:100],
            session_id=f"blind_battle_2_{current_comparison}",
            metadata=vote_metadata,
        )
    except Exception as e:
        print(f"Error updating ratings: {e}")
    
    result_record = {
        "comparison_num": current_comparison + 1,
        "winner": winner_provider,
        "winner_voice": winner_voice,
        "loser": loser_provider,
        "loser_voice": loser_voice,
        "text": pair["text"],
        "user_choice": choice,
        "comment": comment,
        "winner_config": winner_config,
        "loser_config": loser_config
    }
    st.session_state.blind_test_2_results_history.append(result_record)
    
    if comment_key in st.session_state:
        del st.session_state[comment_key]
    
    st.session_state.blind_test_2_comparison_count += 1
    
    if st.session_state.blind_test_2_comparison_count >= st.session_state.blind_test_2_max_comparisons:
        st.session_state.show_final_results_2 = True
        st.session_state.blind_test_2_current_pair = None
    else:
        st.session_state.blind_test_2_current_pair = None
        if "blind_test_2_audio_played" in st.session_state:
            st.session_state.blind_test_2_audio_played = {"A": 0, "B": 0}
    
    st.rerun()

def handle_vote(choice: str, pair: dict):
    """Handle a user vote from blind test and update ELO ratings
    
    IMPORTANT: ELO ratings should ONLY be updated from blind test votes (user preferences),
    NOT from quick test results (latency, TTFB, etc.). Quick test is for technical metrics only.
    """
    from database import db
    
    # Prevent double voting on same comparison
    current_comparison = st.session_state.blind_test_comparison_count
    if st.session_state.get("last_voted_comparison") == current_comparison:
        return  # Already voted on this comparison
    
    # Mark this comparison as voted
    st.session_state.last_voted_comparison = current_comparison
    
    # CRITICAL FIX: Clear ALL audio-related state to force fresh generation
    st.session_state.blind_test_current_pair = None
    
    # Force the next comparison to generate fresh audio
    st.session_state.force_regenerate = True
    
    print(f"[VOTE DEBUG] Vote recorded. Cleared pair. Next comparison will generate fresh audio.")
    
    # Determine winner and loser based on choice
    # IMPORTANT: choice "A" means sample A won, choice "B" means sample B won
    if choice == "A":
        winner_provider = pair["provider_a"]
        loser_provider = pair["provider_b"]
        winner_voice = pair["voice_a"]
        loser_voice = pair["voice_b"]
    else:  # choice == "B"
        winner_provider = pair["provider_b"]
        loser_provider = pair["provider_a"]
        winner_voice = pair["voice_b"]
        loser_voice = pair["voice_a"]
    
    # Debug: Print winner/loser to verify
    print(f"Vote: {choice} | Winner: {winner_provider} | Loser: {loser_provider}")
    
    # Update ELO ratings - winner should get higher rating, loser should get lower
    try:
        new_winner_rating, new_loser_rating = db.update_elo_ratings(
            winner_provider, loser_provider, k_factor=32
        )
        print(f"ELO Updated - Winner ({winner_provider}): {new_winner_rating:.1f}, Loser ({loser_provider}): {new_loser_rating:.1f}")
        
        # Save vote
        db.save_user_vote(
            winner_provider,
            loser_provider,
            pair["text"][:100],
            session_id=f"blind_battle_{current_comparison}"
        )
    except Exception as e:
        print(f"Error updating ratings: {e}")
    
    # Get comment for this comparison
    comment_key = f"comment_{current_comparison}"
    comment = st.session_state.get(comment_key, "")
    
    # De-anonymize comment: replace "Sample A" and "Sample B" with actual provider names
    provider_a_id = pair.get("provider_a", "")
    provider_b_id = pair.get("provider_b", "")
    comment = de_anonymize_comment(comment, provider_a_id, provider_b_id)
    
    # Extract API configuration from samples
    sample_a = pair.get("sample_a")
    sample_b = pair.get("sample_b")
    
    # Get API configs from metadata - map to provider_a and provider_b
    config_provider_a = {}
    config_provider_b = {}
    
    if sample_a and hasattr(sample_a, 'metadata') and sample_a.metadata:
        config_provider_a = {
            "provider": pair.get("provider_a", ""),
            "voice": pair.get("voice_a", ""),
            "model": sample_a.metadata.get("model", ""),
            "format": sample_a.metadata.get("format", ""),
            "sample_rate": sample_a.metadata.get("sample_rate", "")
        }
    
    if sample_b and hasattr(sample_b, 'metadata') and sample_b.metadata:
        config_provider_b = {
            "provider": pair.get("provider_b", ""),
            "voice": pair.get("voice_b", ""),
            "model": sample_b.metadata.get("model", ""),
            "format": sample_b.metadata.get("format", ""),
            "sample_rate": sample_b.metadata.get("sample_rate", "")
        }
    
    # Determine winner and loser configs based on choice
    if choice == "A":
        winner_config = config_provider_a
        loser_config = config_provider_b
    else:
        winner_config = config_provider_b
        loser_config = config_provider_a
    
    # Record result with comment and API configs
    result_record = {
        "comparison_num": current_comparison + 1,
        "winner": winner_provider,
        "winner_voice": winner_voice,
        "loser": loser_provider,
        "loser_voice": loser_voice,
        "text": pair["text"],
        "murf_won": (pair["murf_is"] == choice),
        "user_choice": choice,
        "comment": comment,
        "winner_config": winner_config,
        "loser_config": loser_config
    }
    st.session_state.blind_test_results_history.append(result_record)
    
    # Clear comment from session state after saving
    if comment_key in st.session_state:
        del st.session_state[comment_key]
    
    # Move to next comparison
    st.session_state.blind_test_comparison_count += 1
    
    # Check if we're done
    if st.session_state.blind_test_comparison_count >= st.session_state.blind_test_max_comparisons:
        st.session_state.show_final_results = True
        st.session_state.blind_test_current_pair = None
    else:
        # Force clear the pair to ensure fresh audio generation
        st.session_state.blind_test_current_pair = None
        # Clear any cached audio state
        if "blind_test_audio_played" in st.session_state:
            st.session_state.blind_test_audio_played = {"A": 0, "B": 0}
    
    st.rerun()


def display_interim_results():
    """Display interim results during the blind test"""
    st.subheader("Current Results")
    
    results = st.session_state.blind_test_results_history
    
    if not results:
        st.info("No comparisons completed yet.")
        return
    
    # Calculate win rates
    provider_wins = {}
    provider_losses = {}
    
    for r in results:
        winner = r["winner"]
        loser = r["loser"]
        
        provider_wins[winner] = provider_wins.get(winner, 0) + 1
        provider_losses[loser] = provider_losses.get(loser, 0) + 1
    
    # Create summary table
    all_providers = set(provider_wins.keys()) | set(provider_losses.keys())
    summary_data = []
    
    for provider in all_providers:
        wins = provider_wins.get(provider, 0)
        losses = provider_losses.get(provider, 0)
        total = wins + losses
        win_rate = (wins / total * 100) if total > 0 else 0
        
        summary_data.append({
            "Provider": TTS_PROVIDERS.get(provider, {}).name if provider in TTS_PROVIDERS else provider.title(),
            "Model": get_model_name(provider),
            "Wins": wins,
            "Losses": losses,
            "Win Rate": f"{win_rate:.1f}%",
            "Samples": total
        })
    
    # Sort by win rate
    summary_data.sort(key=lambda x: float(x["Win Rate"].replace("%", "")), reverse=True)
    
    df = pd.DataFrame(summary_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Murf specific stats
    murf_wins = sum(1 for r in results if r.get("murf_won", False))
    murf_total = len(results)
    murf_win_rate = (murf_wins / murf_total * 100) if murf_total > 0 else 0
    
    st.metric(
        "Murf Win Rate",
        f"{murf_win_rate:.1f}%",
        delta=f"{murf_wins} wins / {murf_total} comparisons"
    )
    
    if st.button("Continue Testing", type="primary"):
        st.session_state.show_interim_results = False
        st.rerun()


def display_final_results_2():
    """Display final results after all comparisons for Blind Test 2"""
    st.markdown("---")
    st.header("Final Results")
    
    results = st.session_state.blind_test_2_results_history
    
    if not results:
        st.info("No comparisons completed.")
        if st.button("Start New Test", key="start_new_2"):
            reset_blind_test_2()
        return
    
    # Calculate comprehensive stats
    provider_wins = {}
    provider_losses = {}
    
    for r in results:
        winner = r["winner"]
        loser = r["loser"]
        
        provider_wins[winner] = provider_wins.get(winner, 0) + 1
        provider_losses[loser] = provider_losses.get(loser, 0) + 1
    
    # Calculate ELO for this test session only (starting from 1000 for all)
    # This ensures ELO reflects performance in this specific test, not cumulative history
    # Initialize ELO for all selected competitors
    final_competitors = st.session_state.get("blind_test_2_final_competitors", [])
    test_session_elo = {}
    if final_competitors:
        # Initialize all selected competitors
        for provider in final_competitors:
            test_session_elo[provider] = 1000.0  # Start all at 1000 for this test
    else:
        # Fallback: initialize providers that have wins or losses
        for provider in set(provider_wins.keys()) | set(provider_losses.keys()):
            test_session_elo[provider] = 1000.0  # Start all at 1000 for this test
    
    # Replay all comparisons to calculate ELO for this test session
    for r in results:
        winner = r["winner"]
        loser = r["loser"]
        
        # Skip if winner/loser not in ELO (shouldn't happen, but safety check)
        if winner not in test_session_elo or loser not in test_session_elo:
            continue
            
        winner_rating = test_session_elo[winner]
        loser_rating = test_session_elo[loser]
        
        # Calculate expected scores using EXACT standard ELO formula
        # E_X = 1 / (1 + 10^((R_Y - R_X) / 400))
        import math
        expected_winner = 1 / (1 + math.pow(10, (loser_rating - winner_rating) / 400))
        expected_loser = 1 / (1 + math.pow(10, (winner_rating - loser_rating) / 400))
        
        # Update ELO using EXACT formula: R'_X = R_X + K(S_X - E_X)
        # Winner: S_X = 1, Loser: S_X = 0
        k_factor = 32
        test_session_elo[winner] = winner_rating + k_factor * (1 - expected_winner)
        test_session_elo[loser] = loser_rating + k_factor * (0 - expected_loser)
    
    # Create leaderboard - show ALL competitors that were selected for this battle
    if final_competitors:
        # Show all selected competitors (even if they have 0 wins/losses)
        all_providers = set(final_competitors)
    else:
        # Fallback: show providers that have wins or losses
        all_providers = set(provider_wins.keys()) | set(provider_losses.keys())
    
    leaderboard_data = []
    
    for provider in all_providers:
        wins = provider_wins.get(provider, 0)
        losses = provider_losses.get(provider, 0)
        total = wins + losses
        win_rate = (wins / total * 100) if total > 0 else 0
        
        # Samples should be from current test only (wins + losses), not cumulative database data
        samples = total
        
        leaderboard_data.append({
            "Rank": 0,
            "Provider": TTS_PROVIDERS.get(provider, {}).name if provider in TTS_PROVIDERS else provider.title(),
            "Model": get_model_name(provider),
            "Wins": wins,
            "Losses": losses,
            "Win Rate": f"{win_rate:.1f}%",
            "Samples": samples
        })
    
    # Sort by Win Rate and assign ranks
    leaderboard_data.sort(key=lambda x: float(x["Win Rate"].replace("%", "")), reverse=True)
    for i, item in enumerate(leaderboard_data):
        item["Rank"] = i + 1
    
    df = pd.DataFrame(leaderboard_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Summary metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Comparisons", len(results))
    
    with col2:
        # Show all competitors with their rankings
        if len(leaderboard_data) >= 1:
            # Show winner first
            winner_data = leaderboard_data[0]
            winner_name = winner_data["Provider"]
            winner_win_rate = winner_data["Win Rate"]
            st.markdown(f"**Winner:** {winner_name} ({winner_win_rate})")
            
            # Show all other competitors
            if len(leaderboard_data) > 1:
                competitors_list = []
                for comp_data in leaderboard_data[1:]:
                    comp_name = comp_data["Provider"]
                    comp_win_rate = comp_data["Win Rate"]
                    comp_rank = comp_data["Rank"]
                    competitors_list.append(f"{comp_rank}. {comp_name} ({comp_win_rate})")
                
                competitors_text = "<br>".join(competitors_list)
                st.markdown(f"**All Competitors:**<br>{competitors_text}", unsafe_allow_html=True)
        else:
            st.metric("Providers Tested", len(all_providers))
    
    st.divider()
    
    # Results table with comments
    st.subheader("All Comparisons")
    comparison_data = []
    for result in results:
        # De-anonymize comment if needed (for backward compatibility)
        comment = de_anonymize_comment_from_result(result)
        comparison_data.append({
            "Comparison": result["comparison_num"],
            "Winner": get_provider_display_name(result["winner"]),
            "Winner Voice": result["winner_voice"],
            "Loser": get_provider_display_name(result["loser"]),
            "Loser Voice": result["loser_voice"],
            "Text": result["text"],
            "Comment": comment if comment else '-'
        })
    
    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Start New Test", type="primary", use_container_width=True, key="start_new_test_2"):
            reset_blind_test_2()
    
    with col2:
        if st.button("View Full Leaderboard", use_container_width=True, key="view_leaderboard_2"):
            st.session_state.current_page = "Leaderboard"
            st.rerun()

def reset_blind_test_2():
    """Reset all blind test 2 state"""
    st.session_state.blind_test_2_setup_complete = False
    st.session_state.blind_test_2_current_pair = None
    st.session_state.blind_test_2_comparison_count = 0
    st.session_state.blind_test_2_results_history = []
    st.session_state.show_interim_results_2 = False
    st.session_state.show_final_results_2 = False
    st.session_state.blind_test_2_final_competitors = None  # Clear final competitors
    st.session_state.blind_test_2_comparison_pairs = None  # Clear comparison pairs
    st.session_state.blind_test_2_pair_index = 0  # Reset pair index
    st.session_state.blind_test_2_locale_filter = "US"  # Reset to default locale
    for _k in (
        "_voice_battle_bundle_text_locale",
        "blind_test_2_sentence_draft",
        "blind_test_2_sentence_draft_backup",
        "sentences_text_2",
    ):
        if _k in st.session_state:
            del st.session_state[_k]
    st.rerun()

def display_final_results():
    """Display final results after all comparisons"""
    st.markdown("---")
    st.header("Final Results")
    
    results = st.session_state.blind_test_results_history
    
    if not results:
        st.info("No comparisons completed.")
        if st.button("Start New Test"):
            reset_blind_test()
        return
    
    # Calculate comprehensive stats
    provider_wins = {}
    provider_losses = {}
    
    for r in results:
        winner = r["winner"]
        loser = r["loser"]
        
        provider_wins[winner] = provider_wins.get(winner, 0) + 1
        provider_losses[loser] = provider_losses.get(loser, 0) + 1
    
    # Calculate ELO for this test session only (starting from 1000 for all)
    # This ensures ELO reflects performance in this specific test, not cumulative history
    test_session_elo = {}
    for provider in set(provider_wins.keys()) | set(provider_losses.keys()):
        test_session_elo[provider] = 1000.0  # Start all at 1000 for this test
    
    # Replay all comparisons to calculate ELO for this test session
    for r in results:
        winner = r["winner"]
        loser = r["loser"]
        
        winner_rating = test_session_elo[winner]
        loser_rating = test_session_elo[loser]
        
        # Calculate expected scores using EXACT standard ELO formula
        # E_X = 1 / (1 + 10^((R_Y - R_X) / 400))
        import math
        expected_winner = 1 / (1 + math.pow(10, (loser_rating - winner_rating) / 400))
        expected_loser = 1 / (1 + math.pow(10, (winner_rating - loser_rating) / 400))
        
        # Update ELO using EXACT formula: R'_X = R_X + K(S_X - E_X)
        # Winner: S_X = 1, Loser: S_X = 0
        k_factor = 32
        test_session_elo[winner] = winner_rating + k_factor * (1 - expected_winner)
        test_session_elo[loser] = loser_rating + k_factor * (0 - expected_loser)
    
    # Create leaderboard
    all_providers = set(provider_wins.keys()) | set(provider_losses.keys())
    leaderboard_data = []
    
    for provider in all_providers:
        wins = provider_wins.get(provider, 0)
        losses = provider_losses.get(provider, 0)
        total = wins + losses
        win_rate = (wins / total * 100) if total > 0 else 0
        
        # Samples should be from current test only (wins + losses), not cumulative database data
        samples = total
        
        leaderboard_data.append({
            "Rank": 0,
            "Provider": TTS_PROVIDERS.get(provider, {}).name if provider in TTS_PROVIDERS else provider.title(),
            "Model": get_model_name(provider),
            "Wins": wins,
            "Losses": losses,
            "Win Rate": f"{win_rate:.1f}%",
            "Samples": samples
        })
    
    # Sort by Win Rate and assign ranks
    leaderboard_data.sort(key=lambda x: float(x["Win Rate"].replace("%", "")), reverse=True)
    for i, item in enumerate(leaderboard_data):
        item["Rank"] = i + 1
    
    df = pd.DataFrame(leaderboard_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Summary metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Comparisons", len(results))
    
    with col2:
        # Show winner and competitor with win rate
        if len(leaderboard_data) >= 2:
            winner_data = leaderboard_data[0]  # Highest Win Rate is winner
            competitor_data = leaderboard_data[1]  # Second place is competitor
            
            winner_name = winner_data["Provider"]
            winner_win_rate = winner_data["Win Rate"]
            competitor_name = competitor_data["Provider"]
            competitor_win_rate = competitor_data["Win Rate"]
            
            st.markdown(f"**Winner:** {winner_name} ({winner_win_rate})")
            st.markdown(f"**Competitor:** {competitor_name} ({competitor_win_rate})")
        elif len(leaderboard_data) == 1:
            # Only one provider (shouldn't happen in 1v1, but handle it)
            st.markdown(f"**Winner:** {leaderboard_data[0]['Provider']}")
            st.markdown(f"**Win Rate:** {leaderboard_data[0]['Win Rate']}")
        else:
            total_providers = len(set([r["winner"] for r in results] + [r["loser"] for r in results]))
            st.metric("Providers Tested", total_providers)
    
    st.divider()
    
    # Results table with comments
    st.subheader("All Comparisons")
    comparison_data = []
    for result in results:
        # De-anonymize comment if needed (for backward compatibility)
        comment = de_anonymize_comment_from_result(result)
        comparison_data.append({
            "Comparison": result["comparison_num"],
            "Winner": get_provider_display_name(result["winner"]),
            "Winner Voice": result["winner_voice"],
            "Loser": get_provider_display_name(result["loser"]),
            "Loser Voice": result["loser_voice"],
            "Text": result["text"],
            "Comment": comment if comment else '-'
        })
    
    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Start New Test", type="primary", use_container_width=True):
            reset_blind_test()
    
    with col2:
        if st.button("View Full Leaderboard", use_container_width=True):
            st.session_state.current_page = "Leaderboard"
            st.rerun()


def reset_blind_test():
    """Reset all blind test state"""
    st.session_state.blind_test_setup_complete = False
    st.session_state.blind_test_current_pair = None
    st.session_state.blind_test_comparison_count = 0
    st.session_state.blind_test_results_history = []
    st.session_state.show_interim_results = False
    st.session_state.show_final_results = False
    st.rerun()


def generate_blind_test_samples(text: str, providers: List[str]):
    """Generate audio samples for blind testing (legacy function for backward compatibility)"""
    
    import random
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    
    async def test_provider(provider_id: str):
        try:
            provider = TTSProviderFactory.create_provider(provider_id)
            
            voices = TTS_PROVIDERS[provider_id].supported_voices
            voice = voices[0] if voices else "default"
            
            sample = TestSample(
                id="blind_test",
                text=text,
                word_count=len(text.split()),
                category="blind_test",
                length_category="custom",
                complexity_score=0.5
            )
            
            result = await st.session_state.benchmark_engine.run_single_test(
                provider, sample, voice
            )
            return result
            
        except Exception as e:
            st.error(f"Error testing provider: {str(e)}")
            return None
    
    # Run tests
    for i, provider_id in enumerate(providers):
        status_text.text(f"Generating sample {i+1}/{len(providers)}...")
        
        result = asyncio.run(test_provider(provider_id))
        
        if result and result.success:
            results.append(result)
        
        progress_bar.progress((i + 1) / len(providers))
    
    status_text.text("Samples generated!")
    
    if len(results) < 2:
        st.error("❌ Not enough successful samples generated. Please try again.")
        st.session_state.blind_test_samples = []
        return
    
    random.shuffle(results)
    
    labels = [chr(65 + i) for i in range(len(results))]
    for i, result in enumerate(results):
        result.blind_label = labels[i]
    
    st.session_state.blind_test_samples = results
    st.session_state.blind_test_voted = False
    st.session_state.blind_test_vote_choice = None
    
    st.success(f"✅ Generated {len(results)} blind test samples!")
    st.rerun()

def display_blind_test_samples():
    """Display blind test samples for voting"""
    
    samples = st.session_state.blind_test_samples
    
    if not st.session_state.blind_test_voted:
        st.subheader("🎧 Listen and Vote")
        st.markdown("**Listen to each sample and vote for the one with the best quality:**")
        
        for i in range(0, len(samples), 4):
            cols = st.columns(4)
            for j, result in enumerate(samples[i:i+4]):
                with cols[j]:
                    st.markdown(f"### Sample {result.blind_label}")
                    
                    if result.audio_data:
                        audio_base64 = base64.b64encode(result.audio_data).decode()
                        audio_html = f"""
                        <audio controls controlsList="nodownload" style="width: 100%;">
                            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mpeg">
                        </audio>
                        """
                        st.markdown(audio_html, unsafe_allow_html=True)
                        st.caption(f"Sample {result.blind_label}")
                        
                        st.download_button(
                            label="Download MP3",
                            data=result.audio_data,
                            file_name=f"sample_{result.blind_label}.mp3",
                            mime="audio/mpeg",
                            key=f"download_blind_{result.blind_label}_{i}_{j}"
                        )
        
        st.divider()
        
        st.markdown("### 🗳️ Cast Your Vote")
        
        vote_options = [f"Sample {r.blind_label}" for r in samples]
        selected_sample = st.radio(
            "Which sample sounds best to you?",
            vote_options,
            key="blind_vote_radio"
        )
        
        if st.button("Submit Vote", type="primary"):
            selected_label = selected_sample.split()[1]
            st.session_state.blind_test_vote_choice = selected_label
            st.session_state.blind_test_voted = True
            
            winner_result = next(r for r in samples if r.blind_label == selected_label)
            
            losers = [r for r in samples if r.blind_label != selected_label]
            if losers:
                for loser_result in losers:
                    handle_blind_test_vote(winner_result, loser_result, save_vote=False)
                
                handle_blind_test_vote(winner_result, losers[0], save_vote=True)
            
            st.rerun()
    
    else:
        st.subheader("🎉 Results Revealed!")
        
        voted_sample = next(r for r in samples if r.blind_label == st.session_state.blind_test_vote_choice)
        
        st.success(f"**You voted for Sample {st.session_state.blind_test_vote_choice}**")
        st.info(f"**Sample {st.session_state.blind_test_vote_choice} was generated by: {get_provider_display_name(voted_sample.provider)} ({voted_sample.model_name})**")
        
        st.divider()
        
        st.subheader("🔓 All Samples Revealed")
        
        comparison_data = []
        for result in sorted(samples, key=lambda r: r.blind_label):
            is_winner = result.blind_label == st.session_state.blind_test_vote_choice
            comparison_data.append({
                "Sample": result.blind_label,
                "Provider": get_provider_display_name(result.provider),
                "Model": result.model_name,
                "Location": get_location_display(result),
                "TTFB (ms)": f"{result.ttfb:.1f}" if result.ttfb > 0 else "N/A",
                "File Size (KB)": f"{result.file_size_bytes / 1024:.1f}",
                "Your Choice": "🏆 Winner" if is_winner else ""
            })
        
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.subheader("🎧 Listen Again (with provider names)")
        
        sorted_samples = sorted(samples, key=lambda r: r.blind_label)
        for i in range(0, len(sorted_samples), 4):
            cols = st.columns(4)
            for j, result in enumerate(sorted_samples[i:i+4]):
                with cols[j]:
                    is_winner = result.blind_label == st.session_state.blind_test_vote_choice
                    if is_winner:
                        st.markdown(f"### 🏆 Sample {result.blind_label}")
                    else:
                        st.markdown(f"### Sample {result.blind_label}")
                    
                    st.markdown(f"**{get_provider_display_name(result.provider)}**")
                    st.caption(result.model_name)
                    
                    if result.audio_data:
                        audio_base64 = base64.b64encode(result.audio_data).decode()
                        audio_html = f"""
                        <audio controls controlsList="nodownload" style="width: 100%;">
                            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mpeg">
                        </audio>
                        """
                        st.markdown(audio_html, unsafe_allow_html=True)
                        st.caption(f"TTFB: {result.ttfb:.1f}ms")
                        st.caption(f"Size: {result.file_size_bytes/1024:.1f} KB")
                        
                        st.download_button(
                            label="Download MP3",
                            data=result.audio_data,
                            file_name=f"{result.provider}_{result.blind_label}.mp3",
                            mime="audio/mpeg",
                            key=f"download_revealed_{result.blind_label}_{i}_{j}"
                        )
        
        st.divider()
        
    col1, col2 = st.columns(2)
        
    with col1:
            if st.button("Start New Blind Test", type="primary", use_container_width=True):
                st.session_state.blind_test_samples = []
                st.session_state.blind_test_voted = False
                st.session_state.blind_test_vote_choice = None
                st.rerun()
        
    with col2:
            if st.button("View Leaderboard", use_container_width=True):
                st.session_state.current_page = "Leaderboard"
                st.rerun()

def handle_blind_test_vote(winner_result: BenchmarkResult, loser_result: BenchmarkResult, save_vote: bool = True):
    """Handle blind test vote and update ELO ratings
    
    IMPORTANT: This is the ONLY way ELO should be updated for the leaderboard.
    Quick test results (latency, TTFB) should NOT affect ELO ratings.
    """
    
    from database import db
    
    try:
        winner_rating_before = db.get_elo_rating(winner_result.provider)
        loser_rating_before = db.get_elo_rating(loser_result.provider)
        
        new_winner_rating, new_loser_rating = db.update_elo_ratings(
            winner_result.provider, loser_result.provider, k_factor=32
        )
        
        if save_vote:
            db.save_user_vote(
                winner_result.provider, 
                loser_result.provider, 
                winner_result.text[:100] + "..." if len(winner_result.text) > 100 else winner_result.text,
                session_id="blind_test_session"
            )
        
    except Exception as e:
        st.error(f"Error updating ratings: {e}")

def leaderboard_page():
    """ELO leaderboard page - shows persistent Ranked Blind Test results from database"""
    
    st.header("Leaderboard")
    st.markdown("ELO-based rankings from the blind A/B test (Murf Gen 2 vs NewModel). Default rating is 1000.")
    
    # Get all configured providers - show ALL models with default ELO 1000
    config_status = check_configuration()
    configured_providers = [
        provider_id for provider_id, status in config_status["providers"].items() 
        if status["configured"]
    ]
    
    # Get Ranked Blind Test votes from database (persistent storage)
    # This reads fresh data from database every time the page loads
    votes = db.get_ranked_blind_test_votes()
    
    # Calculate stats from all Ranked Blind Test votes
    provider_wins = {}
    provider_losses = {}
    
    for winner, loser, timestamp, metadata in votes:
        provider_wins[winner] = provider_wins.get(winner, 0) + 1
        provider_losses[loser] = provider_losses.get(loser, 0) + 1
    
    # Initialize ELO for ALL configured providers at 1000 (default)
    test_session_elo = {}
    all_providers = set(configured_providers)  # Start with all configured providers
    # Also include any providers that have votes (in case they're not configured anymore)
    all_providers = all_providers | set(provider_wins.keys()) | set(provider_losses.keys())
    # Two-sided app: always show Murf Gen 2 + NewModel in leaderboard/chart when nothing else to list
    if not all_providers:
        all_providers = set(TTS_PROVIDERS.keys())
    
    # Initialize all providers at default ELO of 1000
    for provider in all_providers:
        test_session_elo[provider] = 1000.0  # Default ELO for all
    
    # Replay all votes to calculate cumulative ELO (only if votes exist)
    if votes:
        for winner, loser, timestamp, metadata in votes:
            winner_rating = test_session_elo.get(winner, 1000.0)
            loser_rating = test_session_elo.get(loser, 1000.0)
            
            # Calculate expected scores using ELO formula
            import math
            expected_winner = 1 / (1 + math.pow(10, (loser_rating - winner_rating) / 400))
            expected_loser = 1 / (1 + math.pow(10, (winner_rating - loser_rating) / 400))
            
            # Update ELO
            k_factor = 32
            test_session_elo[winner] = winner_rating + k_factor * (1 - expected_winner)
            test_session_elo[loser] = loser_rating + k_factor * (0 - expected_loser)
    
    # Prepare data for display - show ALL configured providers with default ELO 1000
    display_data = []
    for provider in all_providers:
        wins = provider_wins.get(provider, 0)
        losses = provider_losses.get(provider, 0)
        total = wins + losses
        win_rate = (wins / total * 100) if total > 0 else 0
        
        provider_config = TTS_PROVIDERS.get(provider)
        provider_name = provider_config.name if provider_config else provider.title()
        model_name = get_model_name(provider)
        
        display_data.append({
            "Rank": 0,  # Will be assigned after sorting
            "Provider": provider_name,
            "Model": model_name,
            "ELO": round(test_session_elo.get(provider, 1000)),  # Default to 1000 if not found
            "Samples": total,
            "Wins": wins,
            "Losses": losses,
            "Win Rate": f"{win_rate:.1f}%" if total > 0 else "0.0%"
        })
    
    # Sort by ELO and assign ranks
    display_data.sort(key=lambda x: x["ELO"], reverse=True)
    for i, item in enumerate(display_data):
        item["Rank"] = f"#{i + 1}"
    
    # Create DataFrame
    df = pd.DataFrame(display_data)
    
    # Add header for main leaderboard table
    st.subheader("ELO Rankings")
    st.caption("Bar chart uses the same ELO values as the table below.")

    if len(df) > 0:
        # Same rows as the table: ELO descending → best provider on the left (matches rank order)
        chart_df = df.sort_values("ELO", ascending=False).copy()
        provider_order = list(chart_df["Provider"])
        try:
            fig_elo = px.bar(
                chart_df,
                x="Provider",
                y="ELO",
                color="ELO",
                color_continuous_scale=["#ddd6ee", "#6642B3"],
                category_orders={"Provider": provider_order},
                hover_data={
                    "Rank": True,
                    "Model": True,
                    "Samples": True,
                    "Wins": True,
                    "Losses": True,
                    "Win Rate": True,
                },
            )
            fig_elo.update_traces(
                text=chart_df["ELO"].astype(int),
                texttemplate="%{text}",
                textposition="outside",
                cliponaxis=False,
            )
            fig_elo.update_layout(
                yaxis_title="ELO",
                xaxis_title="Provider",
                height=max(340, 56 + 28 * len(chart_df)),
                margin=dict(l=8, r=16, t=24, b=72 if len(chart_df) > 2 else 48),
                showlegend=False,
                coloraxis_showscale=False,
            )
            fig_elo.update_xaxes(tickangle=-28 if len(chart_df) > 1 else 0)
            st.plotly_chart(fig_elo, use_container_width=True)
        except Exception as e:
            st.warning(f"Interactive chart failed ({e}). Showing a simple bar chart instead.")
            try:
                st.bar_chart(
                    chart_df,
                    x="Provider",
                    y="ELO",
                    use_container_width=True,
                    sort=False,
                )
            except TypeError:
                st.bar_chart(
                    chart_df.set_index("Provider")[["ELO"]],
                    use_container_width=True,
                )
    
    # Add custom CSS for better styling - remove scroll, let page scroll instead
    st.markdown("""
    <style>
    .stDataFrame {
        border-radius: 8px;
        overflow: visible !important;
    }
    .stDataFrame > div {
        overflow: visible !important;
        max-height: none !important;
    }
    .stDataFrame > div > div {
        overflow: visible !important;
        max-height: none !important;
    }
    .stDataFrame table {
        width: 100%;
        border-collapse: collapse;
    }
    .stDataFrame thead th {
        background-color: #f8f9fa;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 12px;
        letter-spacing: 0.5px;
        padding: 12px 16px;
        border-bottom: 2px solid #dee2e6;
    }
    .stDataFrame tbody td {
        padding: 14px 16px;
        border-bottom: 1px solid #f1f3f5;
    }
    .stDataFrame tbody tr:hover {
        background-color: #f8f9fa;
    }
    /* Remove scrollbar from dataframe container */
    div[data-testid="stDataFrame"] > div {
        overflow: visible !important;
        max-height: none !important;
    }
    div[data-testid="stDataFrame"] > div > div {
        overflow: visible !important;
        max-height: none !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Display table without height limit - use 'content' to show all rows
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height="content",
        column_config={
            "Rank": st.column_config.TextColumn("Rank", width="small"),
            "Provider": st.column_config.TextColumn("Provider", width="medium"),
            "Model": st.column_config.TextColumn("Model", width="medium"),
            "ELO": st.column_config.NumberColumn("ELO", format="%d", width="small"),
            "Samples": st.column_config.NumberColumn("Samples", format="%d", width="small"),
            "Wins": st.column_config.NumberColumn("Wins", format="%d", width="small"),
            "Losses": st.column_config.NumberColumn("Losses", format="%d", width="small"),
            "Win Rate": st.column_config.TextColumn("Win Rate", width="small")
        }
    )

if __name__ == "__main__":
    main()