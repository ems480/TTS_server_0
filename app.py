from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

import edge_tts
import os
import re
import random
import asyncio
import logging
import hashlib
from typing import List, Optional

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# App
# --------------------------------------------------
app = FastAPI(
    title="EduVoice TTS",
    description="Professional teacher-style text-to-speech for educational content",
    version="2.1"
)

# --------------------------------------------------
# Config
# --------------------------------------------------
AUDIO_DIR = "audio"
MAX_TEXT_LENGTH = 14000
MAX_TITLE_LENGTH = 170
TTS_CONCURRENCY_LIMIT = 3
TTS_TIMEOUT = 180

os.makedirs(AUDIO_DIR, exist_ok=True)
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")
tts_semaphore = asyncio.Semaphore(TTS_CONCURRENCY_LIMIT)

# --------------------------------------------------
# Teacher Personas (Text-Based Styling Only)
# --------------------------------------------------
TEACHER_PERSONAS = {
    "warm_mentor": {
        "voice": "en-US-JennyNeural",
        "rate": "-10%",
        "volume": "+0%",
        "pitch": "+2Hz",
        "openings": [
            "Hello everyone, and welcome to today's lesson.",
            "Good day, learners! Let's explore this topic together.",
            "Hi there! I'm excited to walk through this with you."
        ],
        "transitions": [
            "Now, let's move on to the next important point.",
            "That brings us to something really interesting...",
            "Here's where it gets even more helpful..."
        ],
        "encouragements": [
            "You're doing great... let's keep going.",
            "This is a key concept... so take a moment to absorb it.",
            "Excellent progress. Now, let's build on that."
        ],
        "recaps": [
            "So, to quickly recap what we've covered...",
            "Let's pause and summarize the main ideas.",
            "Before we continue, let's reinforce what we just learned."
        ]
    },
    "clear_instructor": {
        "voice": "en-US-GuyNeural", 
        "rate": "-5%",
        "volume": "+0%",
        "pitch": "0Hz",
        "openings": [
            "Welcome. Today we will examine",
            "Let us begin our study of",
            "In this lesson, we focus on"
        ],
        "transitions": [
            "Next, we consider",
            "Moving to the following point",
            "This leads us to"
        ],
        "encouragements": [
            "Note this carefully",
            "This is an essential distinction",
            "Pay particular attention to"
        ],
        "recaps": [
            "To summarize the key points",
            "The essential takeaways are",
            "In review"
        ]
    },
    "engaging_educator": {
        "voice": "en-US-AriaNeural",
        "rate": "-8%",
        "volume": "+0%",
        "pitch": "+5Hz", 
        "openings": [
            "Hey learners! Ready to dive into",
            "Welcome back! Today's topic is fascinating because",
            "Hello! Let's unlock the secrets of"
        ],
        "transitions": [
            "Now here's the exciting part",
            "Wait until you hear this next piece",
            "This connects beautifully to what comes next"
        ],
        "encouragements": [
            "You've got this!",
            "Think about how this applies to your world",
            "Great thinking... let's expand on that"
        ],
        "recaps": [
            "Let's lock this in with a quick review",
            "Here's your mental checkpoint",
            "Before we level up, let's consolidate"
        ]
    }
}

# --------------------------------------------------
# Request Model
# --------------------------------------------------
class TTSRequest(BaseModel):
    title: str = Field(..., max_length=MAX_TITLE_LENGTH)
    text: str = Field(..., max_length=MAX_TEXT_LENGTH)
    persona: Optional[str] = Field(default="warm_mentor")
    include_recap: Optional[bool] = Field(default=True)
    
    @validator('persona')
    def validate_persona(cls, v):
        if v not in TEACHER_PERSONAS:
            raise ValueError(f"Persona must be one of: {list(TEACHER_PERSONAS.keys())}")
        return v

# --------------------------------------------------
# Content Parser for Your <>* Format
# --------------------------------------------------
def parse_educational_content(raw_text: str) -> dict:
    """Parse content in your format: <>metadata* content"""
    sections = []
    major_sections = raw_text.split('*')
    
    for section in major_sections:
        section = section.strip()
        if not section:
            continue
        if '<>' in section:
            parts = section.split('<>', 1)
            if len(parts) >= 2:
                metadata = parts[0].strip()
                content = parts[1].strip()
                if metadata and not content:
                    continue
                content_parts = content.split('<>', 1)
                teachable_text = content_parts[0].strip() if content_parts else content
                if teachable_text:
                    sections.append({
                        'metadata': metadata,
                        'content': teachable_text,
                        'is_image_ref': bool(metadata and not content_parts[0].strip())
                    })
        elif section.strip():
            sections.append({
                'metadata': '',
                'content': section.strip(),
                'is_image_ref': False
            })
    
    return {'sections': sections, 'total_sections': len(sections)}

# --------------------------------------------------
# Clean Formatting Tags
# --------------------------------------------------
def clean_educational_tags(text: str) -> str:
    """Remove educational formatting tags for natural speech"""
    text = re.sub(r'\[color=[^\]]*\](.*?)\[/color\]', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'\[i\](.*?)\[/i\]', r' \1 ', text)
    text = re.sub(r'\[[^\]]*\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --------------------------------------------------
# Text-Based Pacing & Emphasis (SSML Alternative)
# --------------------------------------------------
def apply_text_pacing(text: str, persona: dict) -> str:
    """
    Add natural pacing using TEXT tricks since edge-tts doesn't support <break> tags.
    Uses punctuation, ellipses, and spacing to create pauses.
    """
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if not sentences:
        return text
    
    paced = []
    
    # Opening with natural pause
    paced.append(random.choice(persona['openings']))
    paced.append("...")  # Creates ~300-500ms pause naturally
    
    for i, sentence in enumerate(sentences):
        # Add emphasis via word choice and punctuation
        key_terms = ['important', 'remember', 'note', 'key', 'essential', 'critical']
        for term in key_terms:
            if term.lower() in sentence.lower():
                # Use CAPS + spacing for vocal emphasis (works with neural voices)
                sentence = re.sub(
                    rf'(\b{re.escape(term)}\b)',
                    r' \1 ',
                    sentence,
                    flags=re.IGNORECASE
                )
                break
        
        paced.append(sentence)
        
        # Strategic pauses after complex ideas (using ellipses)
        if len(sentence) > 150 or any(w in sentence.lower() for w in ['therefore', 'consequently', 'in conclusion']):
            paced.append("...")  # Natural pause
        
        # Rhetorical questions every 4 sentences for engagement
        if i > 0 and i % 4 == 0 and len(sentence) < 120:
            rhetorical = random.choice([
                "Does that make sense?",
                "Can you see how this connects?",
                "Think about that for a moment."
            ])
            paced.append(rhetorical)
            paced.append("...")
    
    # Recap section with pause
    if len(sentences) >= 3:
        paced.append("...")
        paced.append(random.choice(persona['recaps']))
        if len(sentences) >= 2:
            first = sentences[0][:80]
            last = sentences[-1][:80]
            paced.append(f"We started with {first}... and concluded with {last}.")
        paced.append("...")
        paced.append(random.choice(persona['encouragements']))
    
    # Join with natural spacing (single space = natural pause)
    return " ".join(paced)

# --------------------------------------------------
# Humanize Lists for Natural Speech
# --------------------------------------------------
def humanize_educational_lists(text: str) -> str:
    """Convert numbered lists to natural spoken enumeration"""
    list_patterns = [
        (r'^\s*(\d+)[\.\)]\s+', 'numbered'),
        (r'^\s*([a-zA-Z])[\.\\)]\s+', 'lettered'),
        (r'^\s*[-*•]\s+', 'bulleted'),
    ]
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    humanized = []
    list_context = None
    list_counter = 0
    
    ordinal_openers = {
        'numbered': ["First", "Second", "Third", "Fourth", "Fifth", "Next", "Then", "Finally"],
        'lettered': ["The first point", "The second point", "The third point", "Next", "Additionally"],
        'bulleted': ["One important aspect", "Another key point", "Also worth noting", "Moreover", "Finally"],
    }
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        matched = False
        for pattern, list_type in list_patterns:
            match = re.match(pattern, sentence)
            if match:
                matched = True
                clean_sentence = re.sub(pattern, '', sentence).strip()
                
                if list_context != list_type:
                    list_context = list_type
                    list_counter = 0
                    humanized.append("Let me break this down for you...")
                
                openers = ordinal_openers.get(list_type, ordinal_openers['numbered'])
                ordinal = openers[min(list_counter, len(openers)-1)]
                
                # Use ellipses for natural pause before list items
                humanized.append(f"... {ordinal}: {clean_sentence}")
                list_counter += 1
                break
        
        if not matched:
            if list_context:
                list_context = None
            humanized.append(sentence)
    
    return ' '.join(humanized)

# --------------------------------------------------
# Cache Key Generation
# --------------------------------------------------
def generate_cache_key(title: str, text: str, persona: str, settings: dict) -> str:
    content_hash = hashlib.md5(
        f"{title}|{text}|{persona}|{settings}".encode()
    ).hexdigest()[:12]
    safe_title = re.sub(r'[^a-zA-Z0-9_\-]', '', title.lower())[:50]
    return f"{safe_title}_{content_hash}.mp3"

# --------------------------------------------------
# Health Check
# --------------------------------------------------
@app.get("/ping")
async def ping():
    return {
        "status": "alive", 
        "service": "EduVoice TTS",
        "version": "2.1",
        "personas": list(TEACHER_PERSONAS.keys()),
        "note": "Uses text-based pacing (edge-tts does not support custom SSML)"
    }

# --------------------------------------------------
# Generate Endpoint - WORKING VERSION
# --------------------------------------------------
@app.post("/generate")
async def generate_audio(req: TTSRequest):
    TITLE = req.title
    RAW_TEXT = req.text
    PERSONA = TEACHER_PERSONAS[req.persona]
    
    logger.info(f"🎓 Audio request: '{TITLE}' | Persona: {req.persona}")
    
    try:
        # STEP 1: Parse educational content format
        parsed = parse_educational_content(RAW_TEXT)
        teachable_sections = [
            sec['content'] for sec in parsed['sections'] 
            if not sec['is_image_ref'] and sec['content'].strip()
        ]
        
        if not teachable_sections:
            teachable_content = RAW_TEXT
        else:
            # Join sections with natural pause (using ellipses)
            teachable_content = ' ... '.join(teachable_sections)
        
        # STEP 2: Clean formatting tags
        clean_text = clean_educational_tags(teachable_content)
        
        # STEP 3: Humanize lists for natural speech
        list_enhanced = humanize_educational_lists(clean_text)
        
        # STEP 4: Apply teacher speech patterns with TEXT-BASED pacing
        teacher_enhanced = apply_text_pacing(list_enhanced, PERSONA)
        
        # STEP 5: Generate cache key
        cache_settings = f"{req.include_recap}"
        filename = generate_cache_key(TITLE, RAW_TEXT, req.persona, cache_settings)
        filepath = os.path.join(AUDIO_DIR, filename)
        
        # STEP 6: Check cache
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            logger.info(f"✅ Serving cached audio: {filename}")
            return JSONResponse(content={
                "audio_url": f"/audio/{filename}",
                "cached": True
            })
        
        # STEP 7: Generate with edge-tts (NO SSML - use parameters only)
        async with tts_semaphore:
            logger.info(f"🔊 Generating with {PERSONA['voice']} | rate={PERSONA['rate']} pitch={PERSONA['pitch']}")
            
            communicate = edge_tts.Communicate(
                text=teacher_enhanced,  # Plain text with pacing via punctuation
                voice=PERSONA['voice'],
                rate=PERSONA['rate'],    # Library parameter (works!)
                volume=PERSONA['volume'],
                pitch=PERSONA['pitch']   # Library parameter (works!)
            )
            
            await asyncio.wait_for(
                communicate.save(filepath),
                timeout=TTS_TIMEOUT
            )
            
            if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
                raise RuntimeError("Audio file generation failed")
        
        logger.info(f"✅ Audio generated: {filename}")
        
        return JSONResponse(content={
            "audio_url": f"/audio/{filename}",
            "cached": False,
            "persona_used": req.persona,
            "sections_processed": len(teachable_sections)
        })
        
    except asyncio.TimeoutError:
        logger.error(f"⏱️ Timeout: {TITLE}")
        raise HTTPException(status_code=504, detail="Generation timed out")
    except edge_tts.exceptions.NoAudioReceived:
        logger.error(f"🔇 No audio from TTS: {TITLE}")
        raise HTTPException(status_code=502, detail="TTS service returned no audio")
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

# --------------------------------------------------
# Startup
# --------------------------------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("🎓 EduVoice TTS Server v2.1 started (text-based pacing)")
    logger.info(f"👥 Personas: {list(TEACHER_PERSONAS.keys())}")
    logger.info("⚠️  Note: edge-tts does not support custom SSML; using text pacing instead")


# from fastapi import FastAPI, HTTPException
# from fastapi.staticfiles import StaticFiles
# from pydantic import BaseModel, Field

# import edge_tts
# import os
# import re
# import random
# import asyncio
# import logging

# # --------------------------------------------------
# # Logging
# # --------------------------------------------------

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # --------------------------------------------------
# # App
# # --------------------------------------------------

# app = FastAPI()

# # --------------------------------------------------
# # Config
# # --------------------------------------------------

# AUDIO_DIR = "audio"
# MAX_TEXT_LENGTH = 14000
# MAX_TITLE_LENGTH = 170
# TTS_CONCURRENCY_LIMIT = 3
# TTS_TIMEOUT = 120

# os.makedirs(AUDIO_DIR, exist_ok=True)

# app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

# # concurrency control
# tts_semaphore = asyncio.Semaphore(TTS_CONCURRENCY_LIMIT)

# # --------------------------------------------------
# # Request Model
# # --------------------------------------------------

# class TTSRequest(BaseModel):
#     title: str = Field(..., max_length=MAX_TITLE_LENGTH)
#     text: str = Field(..., max_length=MAX_TEXT_LENGTH)

# # --------------------------------------------------
# # Safe filename
# # --------------------------------------------------

# def safe_filename(name: str):
#     name = re.sub(r'[^a-zA-Z0-9_\- ]', '', name)
#     name = name.replace(" ", "_")
#     return name[:80]

# # --------------------------------------------------
# # Humanize list text
# # --------------------------------------------------

# def humanize_lists(sentences):

#     LIST_PATTERN = re.compile(
#         r'^\s*(\d+|[a-zA-Z]|i{1,3}|iv|v|vi{0,3}|ix|x)[\.\)\-]\s*',
#         re.IGNORECASE
#     )

#     ORDINAL_STYLES = [
#         ["Firstly", "Secondly", "Thirdly", "Fourthly", "Fifthly", "Next", "Then", "Finally"],
#         ["To begin with", "Next", "Then", "After that", "Another point", "Moving on", "Lastly"],
#         ["The first point is", "The second point is", "The third point is", "Another point is", "One more point is", "Finally"],
#     ]

#     current_style = None
#     ordinal_index = 0
#     humanized = []

#     for s in sentences:

#         match = LIST_PATTERN.match(s)

#         if match:

#             s = LIST_PATTERN.sub('', s).strip()

#             if current_style is None:
#                 current_style = random.choice(ORDINAL_STYLES)
#                 ordinal_index = 0

#             prefix = current_style[min(ordinal_index, len(current_style)-1)]

#             ordinal_index += 1

#             humanized.append(f"{prefix}, {s}")

#         else:
#             humanized.append(s)

#     return humanized

# # --------------------------------------------------
# # Norah AI explanation layer
# # --------------------------------------------------

# def norah_explain(text):

#     sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

#     sentences = humanize_lists(sentences)

#     OPENERS = [
#         "Let us take this step by step.",
#         "Here is a simpler way to see this.",
#         "Let me explain this briefly.",
#         "Think of it this way."
#     ]

#     enriched = []

#     for i, s in enumerate(sentences):

#         enriched.append(s)

#         if i == 0:
#             enriched.append(random.choice(OPENERS))

#     return " ".join(enriched)

# # --------------------------------------------------
# # Health check endpoint for UptimeRobot
# # --------------------------------------------------
# @app.get("/ping")
# async def ping():
#     return {"status": "alive"}
# # --------------------------------------------------
# # Generate endpoint
# # --------------------------------------------------

# @app.post("/generate")
# async def generate_audio(req: TTSRequest):

#     TITLE = req.title
#     TEXT = req.text

#     logger.info(f"Audio request received: {TITLE}")

#     # Process text
#     TEXT = norah_explain(TEXT)

#     sentences = re.split(r'(?<=[.!?])\s+', TEXT)

#     TEXT_WITH_PAUSES = " ".join(
#         [s.strip() + ". " for s in sentences if s.strip()]
#     )

#     voice = "en-US-AvaNeural"
#     rate = "-15%"
#     pitch = "-5Hz"

#     filename = safe_filename(TITLE) + ".mp3"
#     filepath = os.path.join(AUDIO_DIR, filename)

#     # caching
#     if os.path.exists(filepath):

#         logger.info("Serving cached audio")

#         return {
#             "audio_url": f"/audio/{filename}"
#         }

#     try:

#         async with tts_semaphore:

#             logger.info("Generating new audio")

#             communicate = edge_tts.Communicate(
#                 TEXT_WITH_PAUSES,
#                 voice,
#                 rate=rate,
#                 pitch=pitch
#             )

#             await asyncio.wait_for(
#                 communicate.save(filepath),
#                 timeout=TTS_TIMEOUT
#             )

#     except asyncio.TimeoutError:
#         raise HTTPException(status_code=500, detail="TTS generation timeout")

#     except Exception as e:
#         logger.error(str(e))
#         raise HTTPException(status_code=500, detail="TTS generation failed")

#     return {
#         "audio_url": f"/audio/{filename}"
#     }
