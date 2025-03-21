# Vox Locus
This project allows you to locally run an application that will transcribe your conversational data with privacy and security. Transcripts are generated and stay on your device. Depending on your machine specification, you can use the base, medium, or large whisper models.

This does not currently add diarization, but that can be backfilled on the full transcript using tools like [whisper-diarization](https://github.com/MahmoudAshraf97/whisper-diarization).

Outputs will be stored to a generated folder with a name like `transcriptions_yyyymmdd_timestamp`.

## Setup

The script depends on `pyaudio`, so you may also need to install [portaudio](https://people.csail.mit.edu/hubert/pyaudio/) on your machine. 

```bash
brew install portaudio # MacOS
python -m pip install pyaudio # Windows
sudo apt-get install python3-pyaudio  # Linux
```

Once you have that, run the `live_transcribe` script to get going.

```bash
git clone git@github.com:sabaimran/vox-locus.git && cd vox-locus
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r pyproject.toml
python3 live_transcribe.py
```

Run `Ctrl+C` to kill the recording and get your final transcripts.

There is also a `record.py` script that just takes in your audio and dumps it into a `.wav` file for any post-processing you might want to do.