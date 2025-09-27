# The Webcast - Swiss {ai} Weeks Lausanne

# APERTUS Web Search Integration With Audio Prompting & Playback

This project adds a **web search feature** and **audio playback for answers** to the existing APERTUS model. It enables the model to fetch live search results and play back responses in audio format. The application is fully containerized with Docker for easy deployment.

## Features

- Integrates web search capabilities into APERTUS.
- Audio input prompt capabilities
- Plays model answers as audio output.
- Fully containerized for consistent environment across machines.  
- Configurable via `.env` file for API keys.

## Prerequisites

- Docker installed on your machine ([Docker installation guide](https://docs.docker.com/get-docker/)).  
- API keys : Swiss AI Platform / HuggingFace / ElevenLabs / OpenAI.  


## Setup

1. **Clone the repository**

```bash
git clone <repository_url>
cd <repository_folder>
```

2. **Configure environment variables**

* Copy the example environment file:

Linux / macOS

```bash
cp .env.example .env
```

Windows

```cmd
copy .env.example .env
```


* Open `.env` and fill in your API keys and other required values:

```env
SWISS_AI_PLATFORM_API_KEY=""
HG_API_KEY=""
OPENAI_API_KEY=""
ELEVENLABS_API_KEY=""
```

3. **Run the application**

Linux / macOS

```bash
./run.sh
```

Windows
```cmd
./run.bat
```
This script will build and run the Docker container, starting the APERTUS model with web search integration.
Once the container is running, open your browser and go to http://localhost:8501


## License

Apache 2.0

