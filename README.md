# Telegram Translation Checker

This prototype monitors a Telegram group for messages that contain a source text
on the first line and a proposed French translation on the second line. Each
message is evaluated by an LLM; good translations receive a heart reaction,
while improved translations are sent as replies. All events are logged in a
simple Tkinter GUI that also provides optional manual override buttons.

## Requirements

- Python 3.10+
- [Telethon](https://docs.telethon.dev)
- `openai` or `google-generativeai` depending on the chosen LLM provider
- Tkinter (bundled with Python on most platforms)
- SQLite (built into Python)

## Configuration

Set the following environment variables before running the application:

```bash
export TELEGRAM_API_ID="your_api_id"
export TELEGRAM_API_HASH="your_api_hash"
export TELEGRAM_GROUP="target_group_id_or_username"
export OPENAI_API_KEY="your_openai_key"    # required for OpenAI mode
# export GEMINI_API_KEY="your_gemini_key"  # required for Gemini mode
# export LLM_PROVIDER=gemini               # defaults to openai
```

## Running

```bash
python telegram_translation_bot.py
```

The script opens a window displaying real-time logs. When the LLM marks a
translation as correct the bot reacts with a heart. Otherwise it replies with an
improved French version. All processed messages are stored in `translations.db`.

## Packaging

To create a standalone executable (e.g., for Windows) use
[PyInstaller](https://pyinstaller.org/):

```bash
pyinstaller --onefile telegram_translation_bot.py
```

## Notes

- The code assumes you are running it with a user session. Telethon will prompt
  for your phone number and login code on first launch.
- Buttons in the GUI allow you to force accept, force revise, or ignore the
  current message.
