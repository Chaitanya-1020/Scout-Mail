"""
Email pattern generator.

Implements: PRD §6.3 (Contact Finder Agent — generates candidate email
addresses from common organizational naming patterns), §6a.1 (Layered
Confidence Pipeline — pattern generation is one evidence layer; its output is
a set of unverified candidates, existence is checked by a later, separate
layer).
Roadmap: Epic 5 - Contact Finder Agent, Story 4 - Email Pattern Inference,
Task 1.

Generates ranked candidate email addresses for a person at a company domain
using common organizational email naming conventions. This module performs
no verification of any kind (no DNS, no SMTP, no external API) — it is pure,
deterministic string generation (Single Responsibility, per
docs/coding_guidelines.md). Existence verification is a separate, later
pipeline stage (app/agents/contact_finder/smtp_validator.py,
app/agents/contact_finder/hunter_client.py).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


class PatternGenerationError(Exception):
    """Raised when candidate emails cannot be generated for a person."""


@dataclass(frozen=True)
class EmailCandidate:
    """A single generated candidate email address.

    `rank` is the candidate's position in generation order (0 = most
    common/likely pattern), used by downstream layers to prioritize which
    candidates to verify first (PRD §6a.1 — cheap/free checks before paid
    ones).
    """

    email: str
    pattern_name: str
    rank: int


def _slugify_name_part(name_part: str) -> str:
    """Normalize a single name component to lowercase ASCII letters only,
    suitable for use in an email local-part (e.g. "José" -> "jose",
    "O'Brien" -> "obrien").
    """
    normalized = unicodedata.normalize("NFKD", name_part)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z]", "", ascii_only.lower())


class EmailPatternGenerator:
    """Generates ranked candidate email addresses for a person at a domain.

    Splits a full name into first and last name components, then applies a
    fixed, ordered set of common corporate email naming patterns. Patterns
    are ordered from most to least commonly observed in practice, so callers
    that need to limit verification cost (PRD §6a.1) can consume the
    highest-ranked candidates first.
    """

    def generate(self, full_name: str, domain: str) -> list[EmailCandidate]:
        """Generate ranked candidate emails for `full_name` at `domain`.

        Args:
            full_name: the person's full name (e.g. "Jane Q. Doe"). Only the
                first and last tokens are used; middle names/initials are
                ignored.
            domain: bare company domain (e.g. "acme.com").

        Returns:
            Candidates ordered by ascending `rank` (most likely pattern
            first). Middle-initial and single-letter patterns that
            coincide with already-generated candidates are de-duplicated.

        Raises:
            PatternGenerationError: if a first and last name cannot both be
                derived from `full_name`, or `domain` is empty.
        """
        if not domain or not domain.strip():
            raise PatternGenerationError("Cannot generate candidate emails for an empty domain.")

        domain = domain.strip().lower().removeprefix("www.")

        first_raw, last_raw = self._split_name(full_name)
        first = _slugify_name_part(first_raw)
        last = _slugify_name_part(last_raw)

        if not first or not last:
            raise PatternGenerationError(
                f"Could not derive both a first and last name from '{full_name}'."
            )

        first_initial = first[0]
        last_initial = last[0]

        pattern_specs: list[tuple[str, str]] = [
            ("first.last", f"{first}.{last}"),
            ("firstlast", f"{first}{last}"),
            ("flast", f"{first_initial}{last}"),
            ("firstl", f"{first}{last_initial}"),
            ("first_last", f"{first}_{last}"),
            ("last.first", f"{last}.{first}"),
            ("lastfirst", f"{last}{first}"),
            ("first", first),
            ("last", last),
            ("last_first", f"{last}_{first}"),
            ("f.last", f"{first_initial}.{last}"),
            ("first.l", f"{first}.{last_initial}"),
        ]

        candidates: list[EmailCandidate] = []
        seen_local_parts: set[str] = set()

        for rank, (pattern_name, local_part) in enumerate(pattern_specs):
            if local_part in seen_local_parts:
                continue
            seen_local_parts.add(local_part)

            candidates.append(
                EmailCandidate(
                    email=f"{local_part}@{domain}",
                    pattern_name=pattern_name,
                    rank=rank,
                )
            )

        return candidates

    def _split_name(self, full_name: str) -> tuple[str, str]:
        tokens = [token for token in full_name.strip().split() if token]

        if len(tokens) < 2:
            raise PatternGenerationError(
                f"Full name '{full_name}' does not contain both a first and last name."
            )

        return tokens[0], tokens[-1]