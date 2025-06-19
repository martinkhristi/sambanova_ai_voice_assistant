# 🎧 Real-Time Audio Analysis with AI

This project lets you upload or stream live audio and get real-time **transcription** and **AI analysis** using:

- 🧠 [SambaNova's Whisper-Large-v3](https://sambanova.ai/) for speech-to-text
- 🤖 DeepSeek-R1-0528 for smart reasoning and analysis
- 🎙️ [AssemblyAI](https://www.assemblyai.com/) for live streaming transcription

Built with ❤️ using **Gradio** and **Python**.

---

## 🔥 Features

- Upload audio files and get instant transcriptions
- Start live audio streaming from your microphone
- Perform real-time AI analysis on transcripts
- Analysis types include:
  - Summary
  - Key Points
  - Action Items
  - Sentiment Analysis
  - Questions & Answers

---

## 🚀 Demo

Launch the app locally:

```bash
python app.py
Or use share=True for public access via Gradio.

📦 Installation
pip install -r requirements.txt
Ensure you have the following APIs:

🔑 SambaNova API Key (https://sambanova.ai/)

🔑 AssemblyAI API Key (https://www.assemblyai.com/)
