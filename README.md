# The Webcast

# APERTUS Web Search Integration With Audio Prompting & Playback

This project adds a **web search feature** and **audio playback for answers** to the existing APERTUS model. It enables the model to fetch live search results and play back responses in audio format. The application is fully containerized with Docker for easy deployment.

## Features

- Integrates web search capabilities into APERTUS.  
- Plays model answers as audio output.  
- Fully containerized for consistent environment across machines.  
- Configurable via `.env` file for API keys and settings.  
- Simple one-command run using `./run.sh`.  

## Prerequisites

- Docker installed on your machine ([Docker installation guide](https://docs.docker.com/get-docker/)).  
- API keys for any services used (e.g., search APIs, text-to-speech API if applicable).  


## Setup

1. **Clone the repository**

```bash
git clone <repository_url>
cd <repository_folder>
````

2. **Configure environment variables**

* Copy the example environment file:

```bash
cp .env.example .env
```

* Open `.env` and fill in your API keys and other required values:

```env
SEARCH_API_KEY=your_search_api_key_here
OTHER_API_KEY=your_other_api_key_here
```

> ⚠️ **Important:** Keep your `.env` file secret. Do not commit it to version control.

3. **Run the application**

```bash
./run.sh
```

This script will build and run the Docker container, starting the APERTUS model with web search integration.

## License

Apache 2.0

