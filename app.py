from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import edge_tts
import os
import re
import random
import asyncio
import logging

# --------------------------------------------------
# Logging
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# App
# --------------------------------------------------

app = FastAPI()

# --------------------------------------------------
# Config
# --------------------------------------------------

AUDIO_DIR = "audio"
MAX_TEXT_LENGTH = 14000
MAX_TITLE_LENGTH = 170
TTS_CONCURRENCY_LIMIT = 3
TTS_TIMEOUT = 30

os.makedirs(AUDIO_DIR, exist_ok=True)

app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

# concurrency control
tts_semaphore = asyncio.Semaphore(TTS_CONCURRENCY_LIMIT)

# --------------------------------------------------
# Request Model
# --------------------------------------------------

class TTSRequest(BaseModel):
    title: str = Field(..., max_length=MAX_TITLE_LENGTH)
    text: str = Field(..., max_length=MAX_TEXT_LENGTH)

# --------------------------------------------------
# Safe filename
# --------------------------------------------------

def safe_filename(name: str):
    name = re.sub(r'[^a-zA-Z0-9_\- ]', '', name)
    name = name.replace(" ", "_")
    return name[:80]

# --------------------------------------------------
# Humanize list text
# --------------------------------------------------

def humanize_lists(sentences):

    LIST_PATTERN = re.compile(
        r'^\s*(\d+|[a-zA-Z]|i{1,3}|iv|v|vi{0,3}|ix|x)[\.\)\-]\s*',
        re.IGNORECASE
    )

    ORDINAL_STYLES = [
        ["Firstly", "Secondly", "Thirdly", "Fourthly", "Fifthly", "Next", "Then", "Finally"],
        ["To begin with", "Next", "Then", "After that", "Another point", "Moving on", "Lastly"],
        ["The first point is", "The second point is", "The third point is", "Another point is", "One more point is", "Finally"],
    ]

    current_style = None
    ordinal_index = 0
    humanized = []

    for s in sentences:

        match = LIST_PATTERN.match(s)

        if match:

            s = LIST_PATTERN.sub('', s).strip()

            if current_style is None:
                current_style = random.choice(ORDINAL_STYLES)
                ordinal_index = 0

            prefix = current_style[min(ordinal_index, len(current_style)-1)]

            ordinal_index += 1

            humanized.append(f"{prefix}, {s}")

        else:
            humanized.append(s)

    return humanized

# --------------------------------------------------
# Norah AI explanation layer
# --------------------------------------------------

def norah_explain(text):

    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

    sentences = humanize_lists(sentences)

    OPENERS = [
        "Let us take this step by step.",
        "Here is a simpler way to see this.",
        "Let me explain this briefly.",
        "Think of it this way."
    ]

    enriched = []

    for i, s in enumerate(sentences):

        enriched.append(s)

        if i == 0:
            enriched.append(random.choice(OPENERS))

    return " ".join(enriched)

# --------------------------------------------------
# Health check endpoint for UptimeRobot
# --------------------------------------------------
@app.get("/ping")
async def ping():
    return {"status": "alive"}
# --------------------------------------------------
# Generate endpoint
# --------------------------------------------------

@app.post("/generate")
async def generate_audio(req: TTSRequest):

    TITLE = req.title
    TEXT = req.text

    logger.info(f"Audio request received: {TITLE}")

    # Process text
    TEXT = norah_explain(TEXT)

    sentences = re.split(r'(?<=[.!?])\s+', TEXT)

    TEXT_WITH_PAUSES = " ".join(
        [s.strip() + ". " for s in sentences if s.strip()]
    )

    voice = "en-US-AvaNeural"
    rate = "-15%"
    pitch = "-5Hz"

    filename = safe_filename(TITLE) + ".mp3"
    filepath = os.path.join(AUDIO_DIR, filename)

    # caching
    if os.path.exists(filepath):

        logger.info("Serving cached audio")

        return {
            "audio_url": f"/audio/{filename}"
        }

    try:

        async with tts_semaphore:

            logger.info("Generating new audio")

            communicate = edge_tts.Communicate(
                TEXT_WITH_PAUSES,
                voice,
                rate=rate,
                pitch=pitch
            )

            await asyncio.wait_for(
                communicate.save(filepath),
                timeout=TTS_TIMEOUT
            )

    except asyncio.TimeoutError:
        raise HTTPException(status_code=500, detail="TTS generation timeout")

    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=500, detail="TTS generation failed")

    return {
        "audio_url": f"/audio/{filename}"
    }
