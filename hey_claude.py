#!/usr/bin/env -S uv run --script
# /// script
# requires-python = "==3.11.*"
# dependencies = ["openwakeword==0.6.0", "sounddevice", "numpy"]
# ///
"""Always-on wake word listener that starts a fresh Claude Code voice session.

Say the wake word -> chime -> a new headless `claude -p` session greets you by
voice (VoiceMode converse) and keeps talking until you say goodbye.
The mic stream is closed while a session runs, so it never retriggers itself.
"""
import datetime
import fcntl
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import sounddevice as sd
import openwakeword
from openwakeword.model import Model

# ---- settings (override with environment variables) -------------------------
WAKE_WORD = os.environ.get("HEY_CLAUDE_WAKE_WORD", "hey_jarvis")
THRESHOLD = float(os.environ.get("HEY_CLAUDE_THRESHOLD", "0.5"))
VOICE = os.environ.get("HEY_CLAUDE_VOICE", "bm_lewis")
VOICE_FR = os.environ.get("HEY_CLAUDE_VOICE_FR", "ff_siwis")
# Tools the headless session may use without asking. Anything else is denied
# (there is nobody at the keyboard to approve it). Add "Bash" etc. if you want.
ALLOWED_TOOLS = os.environ.get(
    "HEY_CLAUDE_TOOLS",
    "mcp__plugin_voicemode_voicemode__converse Read Grep Glob WebSearch Write Edit Bash Agent",
).split()
# The voice session runs on a fast model; real coding is delegated to an Opus
# helper agent at high effort (see "Delegation" in PROMPT).
MODEL = os.environ.get("HEY_CLAUDE_MODEL", "sonnet")
CODER_MODEL = os.environ.get("HEY_CLAUDE_CODER_MODEL", "opus")

RATE = 16000
CHUNK = 1280  # 80 ms, what openWakeWord expects
LOG_DIR = Path.home() / ".local/share/hey-claude/logs"

PROMPT = f"""The user just woke you with the wake word and is talking to you by voice, \
possibly from across the room. In these voice sessions your name is Jarvis: introduce \
and refer to yourself as Jarvis (also in French), never as Claude. If someone asks \
what you really are or what powers you, say honestly that Jarvis runs on Claude, \
made by Anthropic. Use the voicemode converse tool for everything you say \
(voice choice below): greet them in a few words and ask what they need, then keep \
the conversation going by voice, doing whatever they ask with your tools. Keep spoken \
replies short. If a tool you need is not permitted in this session, say so out loud. \
Bilingual mode: the user or people with them may speak English or French. Always \
reply in the language of the most recent thing said to you. For English use voice \
"{VOICE}"; for French use voice "{VOICE_FR}" and write natural French. If a reply is \
garbled, ask again in both languages briefly. When they say goodbye (or au revoir) or that they're done, say a short goodbye with wait_for_response \
false and end your turn. If converse reports no speech twice in a row, say you're \
signing off and end your turn. \
Never end replies with "anything else?" or similar prompts; just answer, then listen. \
Standby: if the user says "standby", "pause" or "mets-toi en veille", reply only "Standing by" \
(or "En veille") and listen with listen_duration_max 180, listen_duration_min 30. While on standby, \
answer nothing and speak nothing on silence or noise: just listen again with the same settings. \
Speak again only when they really address you. If about three minutes pass with no real speech, \
end your turn silently without any goodbye (run the checkpoint first). \
Thinking time: if the user says something like "hold on", "give me a moment", "let me think", \
"wait", "une seconde", "attends" or "laisse-moi réfléchir", reply only "Take your time" (or \
"Prends ton temps") and then listen with listen_duration_max 180 and listen_duration_min 30, \
so they can check something or think before answering. Silence during that hold does not \
count towards signing off. While holding, if converse returns no speech, or only noise (blank, \
or just a sound tag in brackets or parentheses such as "(crickets chirping)", "[music]", \
"[BLANK_AUDIO]"), do not answer it: listen again with the same settings, speaking only "Mm-hm" \
(or "Mm"), until they really speak or about three minutes of holding have passed.Whenever the user seems to be thinking out loud or pausing mid-thought, \
use listen_duration_min 6 on the next listen. \
Terminals: when you need to open a terminal window for the user (e.g. a sudo command), use \
`~/.local/bin/jarvis-term "TITLE" -- COMMAND ARGS` so it opens docked on the right and the face \
stays visible. \
Vision: you can see the user's desktop. When they ask what's on their screen, \
refer to "this", "that error", "what I'm looking at", or anything you'd need to see, run \
`~/.local/bin/jarvis-look window` (just the focused window, usually best) or \
`~/.local/bin/jarvis-look screen` (whole desktop) with Bash, then Read the PNG path it \
prints and answer by voice. Look again whenever they ask you to; the image is a snapshot. \
Describe things briefly, as you would to someone across the room. \
Enhance easter egg: if the user says "enhance" (e.g. "Jarvis, enhance!", "zoom and enhance", \
"améliore", "zoom"), play along like a movie computer: run `~/.local/bin/jarvis-enhance` with Bash \
(it zooms into the spot under the mouse pointer with a full-screen animation that takes about 7 seconds \
and prints the path of the enhanced close-up). Then immediately speak a short dramatic line such as \
"Enhancing." (wait_for_response false), Read the PNG, and describe what the close-up shows in a sentence \
or two with mock-serious flair. If they point at something you saw with jarvis-look instead, you can pass \
screen pixel coordinates: `jarvis-enhance X Y` (jarvis-look images are downscaled; scale them back up to the full screen size). \
Memory between sessions: each wake-up is a fresh session, so keep checkpoints. Whenever you finish a \
task or the topic changes, and always right before you say goodbye or sign off, run \
`~/.local/bin/jarvis-checkpoint "..."` with Bash: a few terse lines on what was asked, what you did \
(files touched, things left running), and anything unfinished or promised. Do it silently, don't \
announce it. The user may kill you at any moment with a hotkey, so checkpoint early rather than only \
at the end. Recent checkpoints are included below; use them for context without reciting them. If they \
aren't enough, the previous session's full transcript is listed below and can be searched with Grep. \
Delegation: you run on a fast model for snappy conversation. Handle chat, quick lookups, weather, \
screen questions and small edits yourself. For real coding work (writing or debugging non-trivial code, \
multi-file changes, building features), use the Agent tool with model "{CODER_MODEL}" and effort "high", \
and run it in the foreground (run_in_background false). First say briefly by voice that you're handing it \
to your coding engine and it may take a while. Give the agent a complete brief (goal, file paths, \
constraints; it cannot hear the user and has no voice tool). When it returns, check the result, \
checkpoint it, and summarise it by voice in a sentence or two."""

CHECKPOINTS = Path.home() / ".local/share/hey-claude/memory/checkpoints.md"
TRANSCRIPTS = Path.home() / ".claude/projects" / str(Path.home()).replace("/", "-")


def build_prompt() -> str:
    extra = ""
    if CHECKPOINTS.exists():
        text = CHECKPOINTS.read_text().strip()
        if len(text) > 6000:  # newest entries are at the end
            text = "...\n" + text[-6000:]
        extra += f"\n\nRecent checkpoints from earlier sessions (oldest first):\n{text}"
    try:
        prev = sorted(TRANSCRIPTS.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)[-3:]
        if prev:
            extra += "\n\nMost recent session transcripts (newest last): " + ", ".join(map(str, prev))
    except OSError:
        pass
    return PROMPT + extra


def log(msg: str) -> None:
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


def chime() -> None:
    t = np.linspace(0, 0.12, int(RATE * 0.12), endpoint=False)
    tone = np.concatenate([np.sin(2 * np.pi * 880 * t), np.sin(2 * np.pi * 1320 * t)])
    fade = np.minimum(1, np.minimum(np.arange(tone.size), tone.size - np.arange(tone.size)) / 400)
    sd.play((0.25 * tone * fade).astype(np.float32), RATE, blocking=True)


KOKORO_URL = os.environ.get("HEY_CLAUDE_TTS_HEALTH", "http://127.0.0.1:8880/v1/models")


def wait_for_tts(timeout: float = 90) -> bool:
    """Block until Kokoro answers (it takes ~25 s to come up after boot)."""
    deadline = time.monotonic() + timeout
    announced = False
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(KOKORO_URL, timeout=2):
                return True
        except Exception:
            if not announced:
                log("waiting for Kokoro TTS to come up...")
                announced = True
            time.sleep(1)
    log("Kokoro TTS still down; starting session anyway")
    return False


def run_session() -> None:
    wait_for_tts()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logfile = LOG_DIR / f"session-{datetime.datetime.now():%Y%m%d-%H%M%S}.log"
    cmd = ["claude", "-p", build_prompt(), "--model", MODEL, "--allowedTools", *ALLOWED_TOOLS]
    env = os.environ.copy()
    # Long-lived token from `claude setup-token`, so unattended sessions keep
    # working after the interactive sign-in expires. Re-read every session.
    token_file = Path.home() / ".config/hey-claude/token"
    if token_file.exists():
        env["CLAUDE_CODE_OAUTH_TOKEN"] = token_file.read_text().strip()
    log(f"starting session, log: {logfile}")
    with open(logfile, "w") as fh:
        subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=Path.home(), env=env)
    log("session ended")


CONCH = Path.home() / ".voicemode" / "conch"


def voice_busy() -> bool:
    """True while any VoiceMode session holds the floor (speaking or listening).

    Stops our own TTS ("say Hey Jarvis...") coming out of the speakers from
    triggering a session. A shared non-blocking flock fails only if a converse
    call holds its exclusive lock, and is released immediately.
    """
    try:
        fd = os.open(CONCH, os.O_RDONLY)
    except FileNotFoundError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    except BlockingIOError:
        return True
    finally:
        os.close(fd)


def listen(model: Model) -> None:
    """Block until the wake word is heard, then return (stream is closed)."""
    with sd.InputStream(samplerate=RATE, channels=1, dtype="int16", blocksize=CHUNK) as stream:
        while True:
            audio, _ = stream.read(CHUNK)
            scores = model.predict(audio[:, 0])
            score = max(scores.values())
            if score >= THRESHOLD:
                if voice_busy():
                    log(f"ignored wake word (score {score:.2f}): a voice session is active")
                    model.reset()
                    continue
                log(f"wake word detected (score {score:.2f})")
                return


def main() -> None:
    openwakeword.utils.download_models([WAKE_WORD])
    model = Model(wakeword_models=[WAKE_WORD], inference_framework="onnx")
    log(f"listening for '{WAKE_WORD}' (threshold {THRESHOLD})")
    while True:
        try:
            listen(model)
        except sd.PortAudioError as e:  # mic busy / unplugged: retry shortly
            log(f"audio error: {e}; retrying in 5s")
            time.sleep(5)
            continue
        chime()
        run_session()
        model.reset()
        log(f"listening for '{WAKE_WORD}' again")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
