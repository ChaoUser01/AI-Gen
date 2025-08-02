import asyncio
import os
import logging
import sqlite3
from datetime import datetime
from queue import Queue
from threading import Thread
from typing import Optional

from telethon import TelegramClient, events
from telethon.tl.functions.messages import SendReactionRequest
from tkinter import Tk, Button
from tkinter.scrolledtext import ScrolledText

# Optional imports are placed inside functions to avoid ImportError at compile time

LOG_DB = "translations.db"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # 'openai' or 'gemini'
GROUP_ID = int(os.getenv("TELEGRAM_GROUP", "0"))  # Target group id or username

logging.basicConfig(level=logging.INFO)

PROMPT_TEMPLATE = (
    "You are a translation checker. "
    "Given a source sentence and a proposed French translation, "
    "reply with '✅ Good' if the translation is correct and natural. "
    "Otherwise reply only with an improved French translation."
)


async def call_openai(source: str, french: str) -> str:
    try:
        from openai import AsyncOpenAI
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("openai package not installed") from exc

    client = AsyncOpenAI()
    messages = [
        {"role": "system", "content": PROMPT_TEMPLATE},
        {"role": "user", "content": f"Source: {source}\nFrench: {french}"},
    ]
    resp = await client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return resp.choices[0].message.content.strip()


async def call_gemini(source: str, french: str) -> str:
    try:
        import google.generativeai as genai
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("google-generativeai package not installed") from exc

    genai.configure()
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"{PROMPT_TEMPLATE}\nSource: {source}\nFrench: {french}"
    response = await asyncio.to_thread(model.generate_content, prompt)
    return response.text.strip()


async def evaluate_translation(source: str, french: str) -> str:
    if LLM_PROVIDER.lower() == "gemini":
        return await call_gemini(source, french)
    return await call_openai(source, french)


class TranslationApp:
    def __init__(self, queue: Queue):
        self.queue = queue
        self.root = Tk()
        self.root.title("Telegram Translation Checker")
        self.text = ScrolledText(self.root, width=80, height=30)
        self.text.pack(fill="both", expand=True)
        btn_frame = Button(self.root, text="Force Accept", command=self.force_accept)
        btn_frame.pack(side="left")
        Button(self.root, text="Force Revise", command=self.force_revise).pack(side="left")
        Button(self.root, text="Ignore", command=self.ignore).pack(side="left")
        self.current_event: Optional[events.NewMessage.Event] = None
        self.verdict: Optional[str] = None
        self.root.after(200, self.poll_queue)

    def log(self, msg: str) -> None:
        self.text.insert("end", f"{datetime.now():%H:%M:%S} - {msg}\n")
        self.text.see("end")

    def poll_queue(self) -> None:
        while not self.queue.empty():
            self.log(self.queue.get())
        self.root.after(200, self.poll_queue)

    def force_accept(self):
        if self.current_event:
            asyncio.run_coroutine_threadsafe(
                send_heart(self.current_event.client, self.current_event),
                self.current_event.client.loop,
            )
            self.log("Force accepted by user")
        self.current_event = None

    def force_revise(self):
        if self.current_event and self.verdict:
            asyncio.run_coroutine_threadsafe(
                self.current_event.reply(self.verdict),
                self.current_event.client.loop,
            )
            self.log("Force revised by user")
        self.current_event = None

    def ignore(self):
        self.log("Ignored by user")
        self.current_event = None

    def run(self):
        self.root.mainloop()


def init_db():
    conn = sqlite3.connect(LOG_DB)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            msg_id INTEGER PRIMARY KEY,
            chat_id INTEGER,
            source TEXT,
            proposal TEXT,
            verdict TEXT,
            corrected TEXT,
            ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def save_verdict(event: events.NewMessage.Event, source: str, proposal: str, verdict: str, corrected: Optional[str]) -> None:
    conn = sqlite3.connect(LOG_DB)
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO messages (msg_id, chat_id, source, proposal, verdict, corrected) VALUES (?, ?, ?, ?, ?, ?)",
        (event.id, event.chat_id, source, proposal, verdict, corrected),
    )
    conn.commit()
    conn.close()


async def send_heart(client: TelegramClient, event: events.NewMessage.Event) -> None:
    await client(SendReactionRequest(peer=event.chat_id, msg_id=event.id, reaction="❤️"))


async def process_message(event: events.NewMessage.Event, log_queue: Queue):
    text = event.raw_text
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return
    source, proposal = lines[0], lines[1]
    log_queue.put(f"Processing message {event.id}")
    try:
        result = await evaluate_translation(source, proposal)
    except Exception as exc:  # pragma: no cover
        log_queue.put(f"Error from LLM: {exc}")
        return

    verdict = "good" if result.startswith("✅") else "revise"
    corrected = None if verdict == "good" else result
    if verdict == "good":
        await send_heart(event.client, event)
        log_queue.put("Translation accepted")
    else:
        await event.reply(result)
        log_queue.put("Sent correction")
    save_verdict(event, source, proposal, verdict, corrected)


async def run_telegram(log_queue: Queue, app: TranslationApp):
    api_id = int(os.environ["TELEGRAM_API_ID"])
    api_hash = os.environ["TELEGRAM_API_HASH"]
    client = TelegramClient("user", api_id, api_hash)

    @client.on(events.NewMessage(chats=GROUP_ID))
    async def handler(event: events.NewMessage.Event):
        app.current_event = event
        app.verdict = None
        task = asyncio.create_task(process_message(event, log_queue))
        task.add_done_callback(lambda t: None)

    async with client:
        await client.run_until_disconnected()


def start_asyncio(loop, func, *args):
    asyncio.set_event_loop(loop)
    loop.run_until_complete(func(*args))


def main():
    init_db()
    log_queue: Queue = Queue()
    app = TranslationApp(log_queue)

    loop = asyncio.new_event_loop()
    thread = Thread(target=start_asyncio, args=(loop, run_telegram, log_queue, app), daemon=True)
    thread.start()

    app.run()


if __name__ == "__main__":
    main()
