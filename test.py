"""Smoke test for the Anthropic connection. Run this before anything else.

The key used to be hardcoded on line 4 of this file. It now lives in `.env`
(gitignored) - see `.env.example`. Rotate the old one at
https://console.anthropic.com/settings/keys, it has been sitting in plaintext.

    python test.py
"""

import anthropic
from dotenv import load_dotenv

from data.raw.config import LLM_MODEL, LLM_MAX_TOKENS
from generate import sampling_args

load_dotenv()

# No api_key= argument: the SDK reads ANTHROPIC_API_KEY from the environment,
# which .env just populated. Nothing secret ever appears in source.
client = anthropic.Anthropic()

message = client.messages.create(
    model=LLM_MODEL,
    max_tokens=LLM_MAX_TOKENS,
    **sampling_args(LLM_MODEL),
    messages=[{"role": "user", "content": "Hello, Claude"}],
)
for block in message.content:
    if block.type == "text":
        print(block.text)

print(f"\n[ok] {LLM_MODEL} | in {message.usage.input_tokens} tok, "
      f"out {message.usage.output_tokens} tok")
