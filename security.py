"""
Security utilities for the TTS benchmarking tool
"""
import os
import hashlib
import secrets
import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from functools import wraps
import streamlit as st

@dataclass
class SecurityConfig:
    """Security configuration settings"""
    max_text_length: int = 1000
    max_requests_per_minute: int = 60
    api_key_min_length: int = 20
    enable_rate_limiting: bool = True
    enable_input_validation: bool = True

class RateLimiter:
    """Simple rate limiter for API requests"""
    
    def __init__(self, max_requests: int = 60, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: Dict[str, list] = {}
    
    def is_allowed(self, identifier: str) -> Tuple[bool, Optional[str]]:
        """Check if request is allowed for given identifier"""
        current_time = time.time()
        
        if identifier not in self.requests:
            self.requests[identifier] = []
        
        # Clean old requests
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if current_time - req_time < self.time_window
        ]
        
        # Check rate limit
        if len(self.requests[identifier]) >= self.max_requests:
            return False, f"Rate limit exceeded. Max {self.max_requests} requests per {self.time_window} seconds."
        
        # Add current request
        self.requests[identifier].append(current_time)
        return True, None

class InputValidator:
    """Validates user inputs for security"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
    
    def validate_text_input(self, text: str) -> Tuple[bool, Optional[str]]:
        """Validate text input for TTS generation"""
        
        if not text or not text.strip():
            return False, "Text input cannot be empty"
        
        if len(text) > self.config.max_text_length:
            return False, f"Text exceeds maximum length of {self.config.max_text_length} characters"
        
        # Check for potentially malicious content
        suspicious_patterns = [
            '<script', '</script>', 'javascript:', 'data:',
            'vbscript:', 'onload=', 'onerror=', 'onclick='
        ]
        
        text_lower = text.lower()
        for pattern in suspicious_patterns:
            if pattern in text_lower:
                return False, f"Text contains potentially unsafe content: {pattern}"
        
        # Check for excessive special characters (potential injection attempts)
        special_char_count = sum(1 for char in text if not char.isalnum() and not char.isspace())
        if special_char_count > len(text) * 0.3:  # More than 30% special characters
            return False, "Text contains excessive special characters"
        
        return True, None
    
    def validate_api_key(self, api_key: str, provider: str) -> Tuple[bool, Optional[str]]:
        """Validate API key format"""
        
        if not api_key:
            return False, f"API key for {provider} is required"
        
        if len(api_key) < self.config.api_key_min_length:
            return False, f"API key for {provider} appears to be too short"
        
        # Basic format validation (adjust based on provider requirements)
        if provider == "openai":
            if not api_key.startswith("sk-"):
                return False, "OpenAI API key should start with 'sk-'"
        
        return True, None
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe file operations"""
        
        # Remove or replace dangerous characters
        dangerous_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        sanitized = filename
        
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '_')
        
        # Remove leading/trailing dots and spaces
        sanitized = sanitized.strip('. ')
        
        # Limit length
        if len(sanitized) > 255:
            sanitized = sanitized[:255]
        
        # Ensure it's not empty
        if not sanitized:
            sanitized = f"file_{int(time.time())}"
        
        return sanitized

class SessionManager:
    """Manages user sessions and security state"""
    
    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.security_config = SecurityConfig()
        self.input_validator = InputValidator(self.security_config)
    
    def get_session_id(self) -> str:
        """Get or create session ID"""
        if 'session_id' not in st.session_state:
            st.session_state.session_id = secrets.token_hex(16)
        return st.session_state.session_id
    
    def check_rate_limit(self) -> Tuple[bool, Optional[str]]:
        """Check rate limit for current session"""
        session_id = self.get_session_id()
        return self.rate_limiter.is_allowed(session_id)
    
    def validate_request(self, text: str) -> Tuple[bool, Optional[str]]:
        """Validate a TTS request"""
        
        # Check rate limit
        allowed, error_msg = self.check_rate_limit()
        if not allowed:
            return False, error_msg
        
        # Validate input
        valid, error_msg = self.input_validator.validate_text_input(text)
        if not valid:
            return False, error_msg
        
        return True, None

def secure_api_key_input(provider_name: str, env_var: str) -> Optional[str]:
    """Secure API key input with validation"""
    
    # Try to get from environment first
    api_key = os.getenv(env_var)
    
    if api_key:
        # Mask the key for display
        masked_key = api_key[:8] + "*" * (len(api_key) - 12) + api_key[-4:]
        st.success(f"✅ {provider_name} API key loaded from environment: {masked_key}")
        return api_key
    else:
        # Show input field for manual entry
        st.warning(f"⚠️ {provider_name} API key not found in environment variable {env_var}")
        
        with st.expander(f"Enter {provider_name} API Key"):
            manual_key = st.text_input(
                f"{provider_name} API Key:",
                type="password",
                help=f"Enter your {provider_name} API key. This will not be stored permanently.",
                key=f"manual_{env_var}"
            )
            
            if manual_key:
                # Validate the key
                validator = InputValidator(SecurityConfig())
                valid, error_msg = validator.validate_api_key(manual_key, provider_name.lower())
                
                if valid:
                    st.success(f"✅ {provider_name} API key validated")
                    return manual_key
                else:
                    st.error(f"❌ {error_msg}")
                    return None
        
        return None

def rate_limit_decorator(max_requests: int = 10, time_window: int = 60):
    """Decorator for rate limiting functions"""
    
    rate_limiter = RateLimiter(max_requests, time_window)
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Use session ID as identifier
            session_id = st.session_state.get('session_id', 'anonymous')
            
            allowed, error_msg = rate_limiter.is_allowed(session_id)
            if not allowed:
                st.error(f"🚫 {error_msg}")
                return None
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def log_security_event(event_type: str, details: Dict):
    """Log security-related events"""
    
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    session_id = st.session_state.get('session_id', 'unknown')
    
    log_entry = {
        "timestamp": timestamp,
        "session_id": session_id,
        "event_type": event_type,
        "details": details
    }
    
    # In production, you might want to send this to a logging service
    print(f"SECURITY_LOG: {log_entry}")

def create_security_dashboard():
    """Create security monitoring dashboard"""
    
    st.subheader("🔒 Security Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # API Key Status
        openai_key = os.getenv("OPENAI_API_KEY")
        murf_key = os.getenv("MURF_API_KEY")
        
        key_status = "✅ Configured" if (openai_key and murf_key) else "⚠️ Partial" if (openai_key or murf_key) else "❌ Missing"
        st.metric("API Keys", key_status)
    
    with col2:
        # Rate Limiting Status
        rate_limit_status = "✅ Active" if SecurityConfig().enable_rate_limiting else "❌ Disabled"
        st.metric("Rate Limiting", rate_limit_status)
    
    with col3:
        # Input Validation Status
        validation_status = "✅ Active" if SecurityConfig().enable_input_validation else "❌ Disabled"
        st.metric("Input Validation", validation_status)
    
    # Security Configuration
    with st.expander("🔧 Security Configuration"):
        config = SecurityConfig()
        
        st.write("**Current Settings:**")
        st.code(f"""
Max Text Length: {config.max_text_length} characters
Max Requests/Minute: {config.max_requests_per_minute}
API Key Min Length: {config.api_key_min_length}
Rate Limiting: {'Enabled' if config.enable_rate_limiting else 'Disabled'}
Input Validation: {'Enabled' if config.enable_input_validation else 'Disabled'}
        """)
    
    # Security Tips
    with st.expander("💡 Security Best Practices"):
        st.markdown("""
        **For Production Deployment:**
        
        1. **Environment Variables**: Always use environment variables for API keys
        2. **HTTPS**: Deploy with SSL/TLS encryption
        3. **Rate Limiting**: Monitor and adjust rate limits based on usage
        4. **Input Validation**: Never trust user input - validate everything
        5. **Logging**: Monitor security events and API usage
        6. **Access Control**: Implement authentication for sensitive features
        7. **Regular Updates**: Keep dependencies updated for security patches
        
        **API Key Security:**
        - Never commit API keys to version control
        - Use separate keys for development and production
        - Rotate keys regularly
        - Monitor API usage for anomalies
        """)

# Initialize global session manager
session_manager = SessionManager()
