

import math
import os

# ═════════════════════════════════════════════════════════════════════════════
# PART A — Build a realistic prompt string (no API calls)
# ═════════════════════════════════════════════════════════════════════════════

# 40–70 words: factual guardrails for the model
safety_block = (
    "You are a helpful writing assistant. Always stay factual and grounded. "
    "Do not invent, fabricate, or imply the existence of sponsors, partners, "
    "donors, or organisations that have not been explicitly mentioned in the "
    "text provided to you. If any information required to complete the task is "
    "missing, state clearly that you cannot proceed rather than guessing or "
    "filling in details on your own."
)

# 100–140 words: thank-you email to fictional sponsor GreenLeaf Motors
email_body = (
    "Subject: Thank You for Supporting Our Tree-Planting Drive\n\n"
    "Dear GreenLeaf Motors Team,\n\n"
    "On behalf of the Eco-Council at Greenfield College, we would like to "
    "express our heartfelt gratitude for your generous sponsorship of our "
    "Annual Tree-Planting Drive, held on 15 March 2025 in Riverside Park, "
    "Austin. Your contribution made it possible for over sixty student "
    "volunteers to plant more than two hundred saplings across the park's "
    "eastern trails. The enthusiasm our volunteers showed throughout the day "
    "was truly inspiring, and your support was central to making the event a "
    "success. We hope this marks the beginning of a long and meaningful "
    "partnership between GreenLeaf Motors and our council as we continue "
    "working toward a greener campus and community.\n\n"
    "Warm regards,\n"
    "The Eco-Council, Greenfield College"
)

# Exactly one sentence asking for a subject line under 60 characters
follow_up_instruction = (
    "Please suggest a compelling email subject line for the message above "
    "that is strictly under 60 characters."
)

# Concatenate: safety → email → follow-up
full_prompt = (
    safety_block
    + "\n\n"
    + email_body
    + "\n\n"
    + follow_up_instruction
)

print("=" * 65)
print("PART A — Prompt assembled")
print("=" * 65)
print(f"  safety_block word count       : {len(safety_block.split())}")
print(f"  email_body word count         : {len(email_body.split())}")
print(f"  follow_up_instruction sentence: 1")

# ═════════════════════════════════════════════════════════════════════════════
# PART B — Exact prompt token count with tiktoken
# ═════════════════════════════════════════════════════════════════════════════
import tiktoken

try:
    try:
        encoder = tiktoken.encoding_for_model("gpt-4")
    except KeyError:
        encoder = tiktoken.get_encoding("cl100k_base")
        print("Note: gpt-4 encoding not found — using cl100k_base fallback.")
    prompt_tokens = len(encoder.encode(full_prompt))
except Exception:
    # Network-restricted environment: BPE file cannot be downloaded.
    # GPT-4 / cl100k_base averages ~0.75 words per token for English.
    # This approximation matches the same heuristic used in Part C.
    encoder = None
    prompt_tokens = round(len(full_prompt.split()) / 0.75)
    print("Note: tiktoken BPE download unavailable — using word÷0.75 "
          "approximation for prompt token count.")
print("\n" + "=" * 65)
print("PART B — Token count")
print("=" * 65)
print(f"Prompt tokens (tiktoken): {prompt_tokens}")

# ═════════════════════════════════════════════════════════════════════════════
# PART C — Completion budget estimate (rule of thumb)
# ═════════════════════════════════════════════════════════════════════════════
COMPLETION_WORDS = 220
# heuristic: tokens ≈ words ÷ 0.75  (i.e. ~1.33 tokens per word in English)
completion_tokens_est = COMPLETION_WORDS / 0.75

print("\n" + "=" * 65)
print("PART C — Completion estimate")
print("=" * 65)
print(f"Estimated completion tokens (rule of thumb): {completion_tokens_est:.2f}")

# ═════════════════════════════════════════════════════════════════════════════
# PART D — Context window check
# ═════════════════════════════════════════════════════════════════════════════
CONTEXT_LIMIT = 4096
completion_tokens_rounded = math.ceil(completion_tokens_est)   # round up for fit test
total_tokens = prompt_tokens + completion_tokens_rounded
fits = total_tokens <= CONTEXT_LIMIT

print("\n" + "=" * 65)
print("PART D — Context window check  (limit = 4096 tokens)")
print("=" * 65)
print(f"Total tokens needed (rounded-up completion): {total_tokens}")
print(f"Fits in 4096 window? {fits}")

# ═════════════════════════════════════════════════════════════════════════════
# PART E — Concept recall: probabilistic generation
# ═════════════════════════════════════════════════════════════════════════════
concept_paragraph = (
    "Large language models generate text one token at a time. At each step "
    "the model computes a probability distribution over its entire vocabulary "
    "and then samples from that distribution to pick the next token. Because "
    "sampling is a random process, even an identical prompt can lead to a "
    "different token being chosen at the very first step, and those small "
    "differences cascade: each chosen token shifts the probabilities for the "
    "next one, sending the output down a distinct path. A parameter called "
    "temperature controls how spread out the distribution is — higher values "
    "make low-probability tokens more likely, producing more varied and "
    "sometimes surprising replies, while lower values concentrate probability "
    "on the top candidates, giving more predictable output."
)

print("\n" + "=" * 65)
print("PART E — Why the same prompt gives different answers")
print("=" * 65)
print(concept_paragraph)
print(f"\n  (Word count: {len(concept_paragraph.split())})")

# ═════════════════════════════════════════════════════════════════════════════
# PART F — Live temperature demo (optional, requires OPENAI_API_KEY)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PART F — Temperature demo")
print("=" * 65)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("Skipping live temperature demo — set OPENAI_API_KEY to enable.")
else:
    from openai import OpenAI

    # SDK method: client.chat.completions.create
    # API key env var: OPENAI_API_KEY
    client = OpenAI(api_key=OPENAI_API_KEY)

    USER_MSG = "Describe how is the cloud today??"
    MODEL_ID  = "gpt-4o-mini"

    for temp in [0.2, 0.9]:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": USER_MSG}],
            temperature=temp,
        )
        reply = response.choices[0].message.content
        print(f"\n--- temperature={temp} ---")
        print(reply)

print("\n" + "=" * 65)
print("Done.")
print("=" * 65)