#!/usr/bin/env python3
"""
Quick test script to check Google Gemini API status and quota.
Run this to verify if your API key is working and if you've hit quota limits.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

try:
    import google.generativeai as genai
    from google.api_core import exceptions as google_exceptions
except ImportError:
    print("ERROR: google-generativeai not installed. Install with: pip install google-generativeai")
    sys.exit(1)

def test_api():
    """Test the Gemini API with a simple call."""
    api_key = os.getenv('GOOGLE_API_KEY')
    
    if not api_key:
        print("❌ ERROR: GOOGLE_API_KEY not found in environment variables")
        print("   Make sure you have a .env file with GOOGLE_API_KEY=your_key")
        return False
    
    print(f"✓ API Key found: {api_key[:10]}...{api_key[-5:]} (length: {len(api_key)})")
    
    try:
        # Configure API
        genai.configure(api_key=api_key)
        
        # Try a simple API call
        print("\n🔄 Testing API call...")
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content("Say 'Hello, API is working!'")
        
        if response and response.text:
            print(f"✅ SUCCESS: API is working!")
            print(f"   Response: {response.text[:100]}")
            return True
        else:
            print("⚠️  WARNING: API call succeeded but got empty response")
            return False
            
    except google_exceptions.ResourceExhausted as e:
        print("\n❌ QUOTA/RATE LIMIT ERROR (429)")
        print(f"   Error: {str(e)}")
        print("\n   This means:")
        print("   - Your API key has hit its quota/rate limit")
        print("   - You may need to wait for the quota to reset (usually daily)")
        print("   - Or check your quota limits at: https://ai.dev/usage?tab=rate-limit")
        print("\n   Common causes:")
        print("   - Free tier has low limits (e.g., 15 requests/minute)")
        print("   - Daily quota exceeded")
        print("   - Too many requests in a short time")
        return False
        
    except google_exceptions.PermissionDenied as e:
        print("\n❌ PERMISSION ERROR")
        print(f"   Error: {str(e)}")
        print("   Your API key may be invalid or not have the right permissions")
        return False
        
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}")
        print(f"   {str(e)}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("Google Gemini API Status Check")
    print("="*60)
    
    success = test_api()
    
    print("\n" + "="*60)
    if success:
        print("✅ API is working correctly!")
        print("\nIf your scripts are still failing, check:")
        print("  1. Are you making too many requests too quickly?")
        print("  2. Have you exceeded your daily quota?")
        print("  3. Check usage at: https://ai.dev/usage?tab=rate-limit")
    else:
        print("❌ API test failed. See errors above.")
        print("\nNext steps:")
        print("  1. Check your API key at: https://aistudio.google.com/app/apikey")
        print("  2. Check quota/usage at: https://ai.dev/usage?tab=rate-limit")
        print("  3. Wait for quota to reset (usually resets daily)")
        print("  4. Consider upgrading your API plan if needed")
    print("="*60)

