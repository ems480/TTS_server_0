from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles

import edge_tts
import os
import re
import random

app = FastAPI()

# folder for generated audio
AUDIO_DIR = "audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")


class TTSRequest(BaseModel):
    title: str
    text: str


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
# API endpoint
# --------------------------------------------------

@app.post("/generate")
async def generate_audio(req: TTSRequest):

    TEXT = req.text
    TITLE = req.title

    TEXT = norah_explain(TEXT)

    sentences = re.split(r'(?<=[.!?])\s+', TEXT)

    TEXT_WITH_PAUSES = " ".join(
        [s.strip() + ". " for s in sentences if s.strip()]
    )

    voice = "en-US-AvaNeural"
    rate = "-15%"
    pitch = "-5Hz"

    filename = TITLE.replace(":", "").strip() + ".mp3"

    filepath = os.path.join(AUDIO_DIR, filename)

    # audio caching
    if os.path.exists(filepath):

        return {
            "audio_url": f"/audio/{filename}"
        }

    communicate = edge_tts.Communicate(
        TEXT_WITH_PAUSES,
        voice,
        rate=rate,
        pitch=pitch
    )

    await communicate.save(filepath)

    return {
        "audio_url": f"/audio/{filename}"
    }
