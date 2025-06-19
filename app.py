import os
import requests
import gradio as gr
from openai import OpenAI
import json
import threading
import time
import logging
from typing import Type
import assemblyai as aai
from assemblyai.streaming.v3 import (
    BeginEvent,
    StreamingClient,
    StreamingClientOptions,
    StreamingError,
    StreamingEvents,
    StreamingParameters,
    StreamingSessionParameters,
    TerminationEvent,
    TurnEvent,
)

# API Keys - Replace with your actual keys
SAMBANOVA_API_KEY = "sambanova_api_key"
ASSEMBLYAI_API_KEY = "assembly_api_key"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for live streaming
live_transcript = ""
streaming_client = None
is_streaming = False

def analyze_reasoning(audio_file_path):
    """Transcribe uploaded audio file using Whisper-Large-v3"""
    headers = {
        "Authorization": f"Bearer {SAMBANOVA_API_KEY}"
    }

    files = {
        "file": open(audio_file_path, "rb")
    }

    data = {
        "model": "Whisper-Large-v3",
        "language": "english",
        "response_format": "json",
    }

    try:
        response = requests.post(
            "https://api.sambanova.ai/v1/audio/transcriptions",
            headers=headers,
            files=files,
            data=data
        )
        return response.json().get("text", "No transcription found.")
    except Exception as e:
        return f"Error in audio transcription: {str(e)}"

# DeepSeek reasoning model setup
client = OpenAI(
    api_key=SAMBANOVA_API_KEY,
    base_url="https://api.sambanova.ai/v1",
)

def get_reasoning_response(text_input):
    """Get AI analysis from DeepSeek"""
    try:
        response = client.chat.completions.create(
            model="DeepSeek-R1-0528",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes text and provides insights."},
                {"role": "user", "content": text_input}
            ],
            temperature=0.1,
            top_p=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error in reasoning: {str(e)}"

# AssemblyAI Live Streaming Functions
def on_begin(self: Type[StreamingClient], event: BeginEvent):
    print(f"Live session started: {event.id}")

def on_turn(self: Type[StreamingClient], event: TurnEvent):
    global live_transcript
    print(f"{event.transcript} ({event.end_of_turn})")
    live_transcript += f"{event.transcript} "
    
    if event.end_of_turn and not event.turn_is_formatted:
        params = StreamingSessionParameters(format_turns=True)
        self.set_params(params)

def on_terminated(self: Type[StreamingClient], event: TerminationEvent):
    print(f"Live session terminated: {event.audio_duration_seconds} seconds processed")

def on_error(self: Type[StreamingClient], error: StreamingError):
    print(f"Streaming error: {error}")

def start_live_streaming():
    global streaming_client, is_streaming, live_transcript

    if is_streaming:
        return "Already streaming!"

    live_transcript = ""
    is_streaming = True

    try:
        streaming_client = StreamingClient(
            StreamingClientOptions(
                api_key=ASSEMBLYAI_API_KEY,
                api_host="streaming.assemblyai.com",
            )
        )

        streaming_client.on(StreamingEvents.Begin, on_begin)
        streaming_client.on(StreamingEvents.Turn, on_turn)
        streaming_client.on(StreamingEvents.Termination, on_terminated)
        streaming_client.on(StreamingEvents.Error, on_error)

        streaming_client.connect(
            StreamingParameters(
                sample_rate=16000,
                format_turns=True,
            )
        )

        def stream_audio():
            try:
                streaming_client.stream(
                    aai.extras.MicrophoneStream(sample_rate=16000)
                )
            except Exception as e:
                print(f"Streaming error: {e}")

        streaming_thread = threading.Thread(target=stream_audio)
        streaming_thread.daemon = True
        streaming_thread.start()

        return "🎤 Live streaming started! Speak into your microphone..."

    except Exception as e:
        is_streaming = False
        return f"Error starting stream: {str(e)}"

def stop_live_streaming():
    global streaming_client, is_streaming

    if not is_streaming:
        return "Not currently streaming"

    try:
        if streaming_client:
            streaming_client.disconnect(terminate=True)
        is_streaming = False
        return "🛑 Live streaming stopped"
    except Exception as e:
        return f"Error stopping stream: {str(e)}"

def get_live_transcript():
    global live_transcript
    return live_transcript if live_transcript else "No transcript yet..."

def analyze_live_transcript(analysis_type):
    global live_transcript

    if not live_transcript:
        return "No transcript to analyze yet"

    prompts = {
        "Summary": "Please provide a concise summary of the following live transcript:",
        "Key Points": "Extract the main key points from this live conversation:",
        "Action Items": "Identify any action items or tasks mentioned:",
        "Sentiment Analysis": "Analyze the sentiment and tone:",
        "Questions & Answers": "Extract questions and answers from the conversation:"
    }

    reasoning_prompt = f"{prompts.get(analysis_type, prompts['Summary'])}\n\nLive Transcript: {live_transcript}"
    return get_reasoning_response(reasoning_prompt)

def process_audio_file(audio_file, analysis_type):
    if audio_file is None:
        return "Please upload an audio file", ""

    transcription = analyze_reasoning(audio_file)

    prompts = {
        "Summary": "Please provide a concise summary of the following transcribed audio:",
        "Key Points": "Extract the main key points and important information from this transcription:",
        "Action Items": "Identify any action items, tasks, or to-dos mentioned in this transcription:",
        "Sentiment Analysis": "Analyze the sentiment and emotional tone of this transcription:",
        "Questions & Answers": "Extract any questions asked and answers provided in this transcription:"
    }

    reasoning_prompt = f"{prompts.get(analysis_type, prompts['Summary'])}\n\nTranscription: {transcription}"
    analysis = get_reasoning_response(reasoning_prompt)

    return transcription, analysis

# Create Gradio interface
with gr.Blocks(title="Live Audio Analysis with AI", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🎵 Live Audio Analysis with AI")
    gr.Markdown("**Upload audio files or stream live audio for real-time transcription and AI analysis — powered by Whisper-Large-v3 & DeepSeek-R1-0528 via SambaNova")

    with gr.Tabs():
        with gr.TabItem("📁 Upload Audio File"):
            with gr.Row():
                with gr.Column(scale=1):
                    audio_input = gr.Audio(label="Upload Audio File", type="filepath")
                    file_analysis_type = gr.Dropdown(
                        choices=["Summary", "Key Points", "Action Items", "Sentiment Analysis", "Questions & Answers"],
                        value="Summary",
                        label="Analysis Type"
                    )
                    analyze_file_btn = gr.Button("Analyze Audio File", variant="primary", size="lg")

                with gr.Column(scale=2):
                    file_transcription = gr.Textbox(label="Transcription (Whisper-Large-v3)", lines=6)
                    file_analysis = gr.Textbox(label="AI Analysis (DeepSeek-R1)", lines=6)

        with gr.TabItem("🎤 Live Audio Stream"):
            with gr.Row():
                with gr.Column(scale=1):
                    start_btn = gr.Button("🎤 Start Live Stream", variant="primary", size="lg")
                    stop_btn = gr.Button("🛑 Stop Stream", variant="secondary", size="lg")
                    live_analysis_type = gr.Dropdown(
                        choices=["Summary", "Key Points", "Action Items", "Sentiment Analysis", "Questions & Answers"],
                        value="Summary",
                        label="Analysis Type"
                    )
                    analyze_live_btn = gr.Button("Analyze Live Transcript", variant="primary")
                    refresh_btn = gr.Button("Refresh Transcript", variant="secondary")
                    stream_status = gr.Textbox(label="Stream Status", value="Ready to start streaming", interactive=False)

                with gr.Column(scale=2):
                    live_transcript_display = gr.Textbox(label="Live Transcript (AssemblyAI)", lines=8, interactive=False)
                    live_analysis_display = gr.Textbox(label="Live Analysis (DeepSeek-R1)", lines=8)

    analyze_file_btn.click(fn=process_audio_file, inputs=[audio_input, file_analysis_type], outputs=[file_transcription, file_analysis])
    start_btn.click(fn=start_live_streaming, outputs=stream_status)
    stop_btn.click(fn=stop_live_streaming, outputs=stream_status)
    refresh_btn.click(fn=get_live_transcript, outputs=live_transcript_display)
    analyze_live_btn.click(fn=analyze_live_transcript, inputs=[live_analysis_type], outputs=live_analysis_display)

if __name__ == "__main__":
    app.launch(share=True, server_name="0.0.0.0", server_port=7860)
