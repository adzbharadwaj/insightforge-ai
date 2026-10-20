"""Token counting for budgets.

Uses tiktoken's cl100k_base as a provider-neutral estimate. Claude's tokenizer
differs slightly, so budgets keep a safety margin (see SAFETY_MARGIN) rather
than relying on exact counts.
"""

from functools import lru_cache

import tiktoken

SAFETY_MARGIN = 1.10  # count 10% high so an estimate never pushes a prompt over budget


@lru_cache
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return int(len(_encoding().encode(text, disallowed_special=())) * SAFETY_MARGIN) + 1
