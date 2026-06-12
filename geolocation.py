"""
Geolocation utilities for tracking test locations
"""
import requests
from typing import Dict, Optional
import json

class GeolocationService:
    """Service to get geolocation information"""
    
    def __init__(self):
        self.cache = {}
    
    def get_location(self, force_refresh: bool = False) -> Dict[str, str]:
        """
        Get current geolocation based on IP address.
        """
        
        if not force_refresh and 'location' in self.cache:
            return self.cache['location']
        
        location = None
        
        try:
            response = requests.get('https://ipapi.co/json/', timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('country_name') and data.get('country_name') != 'Unknown':
                    location = {
                        'country': data.get('country_name', 'Unknown'),
                        'country_code': data.get('country_code', 'XX'),
                        'region': data.get('region', 'Unknown'),
                        'city': data.get('city', 'Unknown'),
                        'latitude': str(data.get('latitude', 0)),
                        'longitude': str(data.get('longitude', 0)),
                        'timezone': data.get('timezone', 'UTC'),
                        'ip': data.get('ip', 'Unknown')
                    }
                    self.cache['location'] = location
                    return location
        except Exception as e:
            print(f"ipapi.co failed: {e}")
        
        try:
            response = requests.get('http://ip-api.com/json/', timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success' and data.get('country'):
                    location = {
                        'country': data.get('country', 'Unknown'),
                        'country_code': data.get('countryCode', 'XX'),
                        'region': data.get('regionName', 'Unknown'),
                        'city': data.get('city', 'Unknown'),
                        'latitude': str(data.get('lat', 0)),
                        'longitude': str(data.get('lon', 0)),
                        'timezone': data.get('timezone', 'UTC'),
                        'ip': data.get('query', 'Unknown')
                    }
                    self.cache['location'] = location
                    return location
        except Exception as e:
            print(f"ip-api.com failed: {e}")
        
        try:
            response = requests.get('https://ipinfo.io/json', timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('country'):
                    # Parse location string like "37.7749,-122.4194"
                    loc_parts = data.get('loc', '0,0').split(',')
                    location = {
                        'country': data.get('country', 'Unknown'),
                        'country_code': data.get('country', 'XX'),
                        'region': data.get('region', 'Unknown'),
                        'city': data.get('city', 'Unknown'),
                        'latitude': loc_parts[0] if len(loc_parts) > 0 else '0',
                        'longitude': loc_parts[1] if len(loc_parts) > 1 else '0',
                        'timezone': data.get('timezone', 'UTC'),
                        'ip': data.get('ip', 'Unknown')
                    }
                    self.cache['location'] = location
                    return location
        except Exception as e:
            print(f"ipinfo.io failed: {e}")
        
        return {
            'country': 'Unknown',
            'country_code': 'XX',
            'region': 'Unknown',
            'city': 'Unknown',
            'latitude': '0',
            'longitude': '0',
            'timezone': 'UTC',
            'ip': 'Unknown'
        }
    
    def get_location_string(self) -> str:
        """Get location as a formatted string"""
        try:
            location = self.get_location()
            
            if not location or not isinstance(location, dict):
                return 'Unknown'
            
            parts = []
            if location.get('city') and location.get('city') != 'Unknown':
                parts.append(location['city'])
            if location.get('region') and location.get('region') != 'Unknown' and location.get('region') != location.get('city'):
                parts.append(location['region'])
            if location.get('country') and location.get('country') != 'Unknown':
                parts.append(location['country'])
            
            if parts:
                return ', '.join(parts)
            return 'Unknown'
        except Exception as e:
            print(f"Error getting location string: {e}")
            return 'Unknown'
    
    def get_country_flag(self, country_code: str = None) -> str:
        """Get country flag emoji from country code"""
        try:
            if country_code is None:
                location = self.get_location()
                if not location or not isinstance(location, dict):
                    return '🌍'
                country_code = location.get('country_code', 'XX')
            
            if not country_code or country_code == 'XX' or country_code == 'Unknown':
                return '🌍'
            
            flag = ''.join(chr(ord(c) + 127397) for c in country_code.upper())
            return flag
        except Exception as e:
            print(f"Error getting country flag: {e}")
            return '🌍'

geo_service = GeolocationService()

