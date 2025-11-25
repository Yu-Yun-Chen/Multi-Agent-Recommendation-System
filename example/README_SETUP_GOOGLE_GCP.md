# Using Google GCP Credits for AgentSociety Challenge

This guide shows you how to use your $50 Google GCP credits with the enhanced agents. **You only need a Google API key - no OpenAI key required!**

## 🔑 Getting Your Google API Key

### Step 1: Access Google AI Studio
1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account (the one with GCP credits)

### Step 2: Create API Key
1. Click **"Create API Key"**
2. Select your **GCP project** (where you have the $50 credits)
3. Copy the generated API key
4. Save it securely!

### Step 3: Enable Billing
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to **Billing** → **Link a billing account**
3. Make sure your project with the $50 credits is linked
4. The Gemini API will use these credits automatically

## 📦 Installation

### Install Required Packages

```bash
# Navigate to project root
cd /path/to/AgentSocietyChallenge

# Install dependencies using Poetry (recommended)
poetry install

# OR using pip
pip install langchain-google-genai google-generativeai python-dotenv

# OR update using Poetry
poetry add langchain-google-genai google-generativeai python-dotenv
```

## 🚀 Quick Start

### Option A: Using .env File (Recommended - Most Secure)

```bash
# 1. Navigate to example directory
cd example

# 2. Create .env file from template
cp env_template.txt .env

# 3. Edit .env file with your Google API key
# Open .env and add:
GOOGLE_API_KEY=your-api-key-here

# Note: You do NOT need OPENAI_API_KEY anymore!
```

### Option B: Set Environment Variable

```bash
# For current session
export GOOGLE_API_KEY="your-api-key-here"

# Or add to ~/.bashrc or ~/.zshrc for permanent
echo 'export GOOGLE_API_KEY="your-api-key-here"' >> ~/.bashrc
```

## 🎯 Running Tests

### Test Basic Functionality

```bash
cd example

# Test the Google Gemini LLM
python -c "
from GoogleGeminiLLM import GoogleGeminiLLM
import os

llm = GoogleGeminiLLM(api_key=os.getenv('GOOGLE_API_KEY'))
print(llm(messages=[{'role': 'user', 'content': 'Say hello!'}]))
print('✅ Google Gemini LLM working!')
print('✅ Embeddings:', llm.get_embedding_model() is not None)
"
```

### Run Recommendation Accuracy Tests

```bash
cd example

# Test on 5 tasks (quick test)
python test_recommendation_accuracy.py --num-tasks 5 --workflows default openagi

# Test on more tasks (full evaluation)
python test_recommendation_accuracy.py --num-tasks 10 --workflows default self_refine openagi

# Test all workflows (expensive!)
python test_recommendation_accuracy.py --num-tasks 10 --workflows all
```

## 💰 Cost Estimates with Google Credits

### Gemini API Pricing (uses your $50 GCP credits)
- **gemini-2.0-flash** (default): ~$0.10 per 1M input tokens (latest, fastest)
- **gemini-1.5-flash**: ~$0.075 per 1M input tokens (cheaper alternative)
- **Embeddings (embedding-001)**: $0.025 per 1M tokens

### Estimated Costs Per Run
- 10 tasks, default workflow: ~$0.20-0.40
- 100 tasks, default workflow: ~$2-4
- 400 tasks (full evaluation): ~$8-16

**Your $50 credits should be enough for extensive testing!**

## 🔧 What's Using Google APIs

The `GoogleGeminiLLM` now uses:
1. **Google Gemini** for LLM calls (main model)
2. **Google Embeddings** (`embedding-001`) for memory/vector search

**No OpenAI API key needed at all!**

## 🐛 Troubleshooting

### Error: "No module named 'langchain_google_genai'"
```bash
pip install langchain-google-genai google-generativeai
```

### Error: "Google API key is required"
Make sure you've set the environment variable:
```bash
export GOOGLE_API_KEY="your-key-here"
```

Or create a `.env` file with:
```
GOOGLE_API_KEY=your-key-here
```

### Error: "Embedding model not available"
This means `langchain-google-genai` isn't installed. Run:
```bash
pip install langchain-google-genai
```

## 📚 Additional Resources

- [Google AI Studio](https://aistudio.google.com/app/apikey) - Get your API key
- [Gemini API Documentation](https://ai.google.dev/docs)
- [Google Cloud Console](https://console.cloud.google.com/) - Manage billing