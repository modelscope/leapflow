"""Terminal-based IOProvider for interactive CLI confirmation."""

from __future__ import annotations

import asyncio


class TerminalIOProvider:
    """IOProvider implementation using stdin/stdout for CLI interaction.

    Implements the IOProvider protocol defined in confirmation.py,
    enabling CONFIRM and STEP level confirmations in terminal sessions.
    """

    async def prompt(self, message: str) -> str:
        """Display message and read user input asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: input(message).strip()
        )

    async def display(self, message: str) -> None:
        """Display a message to the user."""
        print(message, flush=True)
