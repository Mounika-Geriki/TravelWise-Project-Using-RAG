# TravelWiseNYC-Chatbot
👋 Welcome to TravelWise NYC!

## Introduction:
Are you planning a trip to the Big Apple, or just looking to discover something new in New York City? TravelWise NYC is your smart, always-on travel companion!
Whether you're searching for the best pizza in Brooklyn, hidden art galleries, or the fastest subway route, our intelligent chatbot delivers up-to-date, reliable answers-right when you need them.

🚀 What Can TravelWise NYC Do for You?

- Ask Anything NYC: Museums, parks, food, transit, events-just type your question!
- Real-Time Answers: Combines trusted local knowledge with live web search for the freshest info.
- Easy to Use: Clean, intuitive web interface. No technical skills needed.
- Always Improving: Learns from your questions to get smarter every day.

In this project we'll develop a Retreival Augmented Generation workflow using Agents from LangGraph.
This project will help you to understand the basic of Adaptive RAG and Agents in LangGraph and also provide a real world example of Agents.

## Goal of The Project:
The aim of this project is to improve the traveling experience of tourists visiting New York City by providing general and up-to-date information about the city.
With this application, tourists will be able to get answers to general questions such as 'Where is the Empire State Building?' or 'What should I eat in Chinatown?' as well as up-to-date questions such as 'What are the subway ticket fares in New York?' or 'What is the weather like in New York City?'

## Key Features:

- 🧠 Dynamic routing between vector database information and live web search
- ✅ Self-verification mechanism ensuring high-quality responses
- 🎨 Clean, user-friendly web interface with markdown support
- 🔍 Smart context-aware recommendations
- ⚡ Real-time updates for NYC information
- 📊 Performance tracking and analytics
- 📱 Responsive design

## 📁 Project Structure

```
TravelWise-Project-Using-RAG/
│
├── data/                        # PDF knowledge base files
├── mgllm/                       # (custom LLM utilities, if any)
├── nyc_faiss_google_index/      # FAISS vector index files
│   ├── index.faiss
│   └── index.pkl
├── static/
│   ├── images/
│   └── styles.css               # Custom CSS for frontend
├── templates/
│   ├── error.html
│   ├── index.html
│   └── result.html
├── tllm/                        # (custom LLM code, if any)
├── .env                         # API keys and configuration
├── app.py                       # Main Flask application
├── requirements.txt             # Python dependencies

```

## 🔧 Installation

1. **Prerequisites**
- Python 3.8+
- pip (Python package installer)
- Git

2. **Clone Repository**
   ```bash
   git clone https://github.com/rishiguptha/nutismart_rag_project.git
   cd nutismart_rag_project
   ```

3. **Set Up Virtual Environment**
   ```bash
   # Create virtual environment
    python -m venv venv
    source venv/bin/activate
   ```
4. **Install Dependencies**
   ```bash
    pip install -r requirements.txt
   ```
📦 Requirements
 The main dependencies include:
   
  ``` bash
  flask==2.0.1
  langchain==0.1.5
  langchain_community==0.0.12
  faiss-cpu==1.7.4
  pymupdf==1.23.8
  requests==2.31.0
  python-dotenv==1.0.0
  tavily-python==0.2.1
  ```
## ⚙️**Configuration**

Create a .env file in the root directory with the following variables:
```
GEMINI_API_KEY=your_google_gemini_api_key
TAVILY_API_KEY=your_tavily_api_key
```

 **API Keys**
 
You'll need to obtain the following API keys:

Google AI Studio (Gemini): Get your API key from [Google AI Studio](https://aistudio.google.com/prompts/new_chat)
Tavily API: Register for a key at [Tavily](https://app.tavily.com/home)

**Knowledge Base**
- Place your NYC-related PDF documents in the data/ folder.
- On first run, the FAISS vector index will be built from these files and saved in nyc_faiss_google_index/.

## **Running the Application**
1. Start the Flask Server

``` bash
python app.py

```
- The web app will be available at http:.
- The main interface is served from templates/index.html.


## Example Queries

```
"What are the top museums in NYC?"
"Best pizza spots in NYC"
"How to travel to Brooklyn from Manhattan?"
"What parks can I visit in NYC?"
"3-day NYC itinerary"
```

## 📊 Performance Tracking
The application tracks performance metrics:

- Total queries processed
- Successfully answered queries
- Query rejection rate
- Processing errors
- Answer generation rate

## Future Enhancements
- Predictive analytics for user intent.
- Real-time push notifications (weather, transit).
- Multimodal (image/video) support.
- Expansion to other cities and document types.






## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

