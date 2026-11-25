"""Conversation state: the cached intent, the entity cache and the pending clarification.

Pipeline role: `pipeline.handle_turn` reads the session before answering a message and
writes the merged result back afterwards, so follow-ups ("And in 2020?", "What about
Norris?") can reuse what was said earlier. The UI shows the cache so every reuse is traceable.
"""
from dataclasses import dataclass, field

# The season slots form one group: a single year replaces a cached range and vice versa.
SEASON_SLOTS = ("year", "year_from", "year_to")


@dataclass
class Session:
    intent: str | None = None                   # intent of the last in-domain question
    entities: dict = field(default_factory=dict)   # slot -> value, kept across turns
    pending: dict | None = None                 # open clarification: intent, found entities, ambiguous candidates
    history: list = field(default_factory=list)    # every Turn so far, rendered as the chat thread

    def merge(self, found: dict):
        """Replace only the slots the new message names; every other cached slot is kept."""
        if any(slot in found for slot in SEASON_SLOTS):
            for slot in SEASON_SLOTS:
                self.entities.pop(slot, None)
        self.entities.update(found)

    def clear(self):
        """Forget everything (the sidebar's reset button)."""
        self.intent, self.entities, self.pending, self.history = None, {}, None, []
