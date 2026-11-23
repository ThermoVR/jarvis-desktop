#!/usr/bin/env bash
# F5 (XF86AudioPlay) handler.
# - Voice session speaking/listening (VoiceMode conch lock held):
#     `voicemode control skip-forward` -> ends the listen turn now (transcribes
#     what it has), or cuts the current spoken reply and starts listening.
# - Otherwise: normal Play/Pause for any MPRIS media player (FreeTube etc.).
# Bound via Cinnamon custom keybinding "hey-claude-f5"; the stock media-keys
# 'play' binding was cleared so this one gets the key.

CONCH="$HOME/.voicemode/conch"
LOG="$HOME/.local/share/hey-claude/logs/f5.log"

if [ -e "$CONCH" ] && ! flock -n -s "$CONCH" true 2>/dev/null; then
    echo "$(date +%T) voice active -> skip-forward" >>"$LOG"
    exec "$HOME/.local/bin/voicemode" control skip-forward >>"$LOG" 2>&1
fi

# No voice turn in progress: Play/Pause every MPRIS player on the session bus.
players=$(dbus-send --session --print-reply --dest=org.freedesktop.DBus \
    /org/freedesktop/DBus org.freedesktop.DBus.ListNames 2>/dev/null |
    grep -o 'org\.mpris\.MediaPlayer2\.[^"]*')
echo "$(date +%T) no voice -> play/pause: ${players:-none}" >>"$LOG"
for p in $players; do
    dbus-send --session --type=method_call --dest="$p" /org/mpris/MediaPlayer2 \
        org.mpris.MediaPlayer2.Player.PlayPause
done
