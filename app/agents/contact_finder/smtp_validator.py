"""
SMTP mailbox existence validator.

Implements: PRD §6.3 (Contact Finder Agent — verifies whether a candidate
email address's mailbox exists via SMTP handshake), §6a.1 (Layered
Confidence Pipeline — SMTP validation is one evidence layer contributing to a
contact's confidence score; it never sends an email, it only probes mailbox
existence), §13.2 (Non-Goals Are Enforced Constraints — no email is ever
transmitted by this or any other layer of the Contact Finder pipeline).
Roadmap: Epic 5 - Contact Finder Agent, Story 5 - SMTP Validation, Task 1.

Verifies a candidate email's mailbox existence via an MX lookup followed by
an SMTP handshake up to (but never including) the DATA command: MAIL FROM,
RCPT TO, then RSET/QUIT. No message body is ever composed or sent by this
module. Depends only on `dnspython` (MX lookup) and the standard library
`smtplib` — no dependency on `app/services/email_sender.py` or any send-path
module, and no shared code path with them (Single Responsibility, per
docs/coding_guidelines.md §5 — send-path isolation).
"""

from __future__ import annotations

import smtplib
import socket
from dataclasses import dataclass
from enum import Enum

import dns.exception
import dns.resolver
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import get_settings

_DNS_TIMEOUT_SECONDS = 5
_SMTP_PORT = 25

# A plausible-but-unrelated sender address used only for the SMTP handshake's
# MAIL FROM step, required by the SMTP protocol to probe RCPT TO. This
# address is never used to compose or send an actual message.
_PROBE_SENDER_ADDRESS = "verify-probe@example.com"


class SmtpValidationOutcome(str, Enum):
    """Result of attempting to validate a candidate mailbox via SMTP."""

    MAILBOX_EXISTS = "mailbox_exists"
    MAILBOX_NOT_FOUND = "mailbox_not_found"
    UNKNOWN = "unknown"
    """The mailbox server accepted RCPT TO for any address (catch-all) or
    the check was otherwise inconclusive — treated as non-evidence, not as
    confirmation."""
    NO_MX_RECORD = "no_mx_record"
    CONNECTION_FAILED = "connection_failed"


class SmtpValidationError(Exception):
    """Raised when SMTP validation cannot be attempted at all (e.g. invalid input)."""


@dataclass(frozen=True)
class SmtpValidationResult:
    """Structured outcome of an SMTP mailbox-existence check for one candidate email."""

    email: str
    outcome: SmtpValidationOutcome
    mx_host: str | None
    smtp_response_code: int | None
    detail: str


class _TransientSmtpError(Exception):
    """Internal marker for a connection-level failure eligible for retry."""


class SmtpMailboxValidator:
    """Checks whether a candidate email's mailbox exists via an SMTP
    handshake, without ever sending a message.

    The handshake sequence is: resolve MX records for the domain, connect to
    the highest-priority mail server, issue HELO, MAIL FROM, and RCPT TO,
    then inspect the RCPT TO response code, and finally RSET + QUIT. The
    DATA command is never issued, so no message content is ever transmitted
    or accepted by the remote server (PRD §13.2).
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._smtp_timeout_seconds = settings.smtp_handshake_timeout_seconds

    def validate(self, email: str) -> SmtpValidationResult:
        """Validate a single candidate email's mailbox existence.

        Raises:
            SmtpValidationError: if `email` is not a syntactically plausible
                address (missing '@' or domain part). Network/protocol
                failures are captured in the returned result's `outcome`
                rather than raised, so a batch of candidates can be
                processed without one failure aborting the rest.
        """
        if not email or "@" not in email:
            raise SmtpValidationError(f"'{email}' is not a valid email address to validate.")

        domain = email.rsplit("@", 1)[-1].strip()
        if not domain:
            raise SmtpValidationError(f"'{email}' has no domain part to validate against.")

        mx_host = self._resolve_mx_host(domain)
        if mx_host is None:
            return SmtpValidationResult(
                email=email,
                outcome=SmtpValidationOutcome.NO_MX_RECORD,
                mx_host=None,
                smtp_response_code=None,
                detail=f"No MX record found for domain '{domain}'.",
            )

        try:
            return self._attempt_handshake(email=email, mx_host=mx_host)
        except _TransientSmtpError as exc:
            return SmtpValidationResult(
                email=email,
                outcome=SmtpValidationOutcome.CONNECTION_FAILED,
                mx_host=mx_host,
                smtp_response_code=None,
                detail=f"SMTP connection to '{mx_host}' failed after retries: {exc}",
            )

    def _resolve_mx_host(self, domain: str) -> str | None:
        resolver = dns.resolver.Resolver()
        resolver.timeout = _DNS_TIMEOUT_SECONDS
        resolver.lifetime = _DNS_TIMEOUT_SECONDS

        try:
            answers = resolver.resolve(domain, "MX")
        except dns.exception.DNSException:
            return None

        mx_records = sorted(answers, key=lambda r: r.preference)
        if not mx_records:
            return None

        return str(mx_records[0].exchange).rstrip(".")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(_TransientSmtpError),
        reraise=True,
    )
    def _attempt_handshake(self, email: str, mx_host: str) -> SmtpValidationResult:
        try:
            with smtplib.SMTP(timeout=self._smtp_timeout_seconds) as smtp:
                smtp.connect(mx_host, _SMTP_PORT)
                smtp.helo(socket.getfqdn())

                smtp.mail(_PROBE_SENDER_ADDRESS)
                response_code, response_message = smtp.rcpt(email)

                # Never issue DATA. RSET clears the transaction state before
                # QUIT, leaving no pending message on the server side.
                smtp.rset()

            return self._to_result(email, mx_host, response_code, response_message)

        except (TimeoutError, socket.timeout, ConnectionError, smtplib.SMTPConnectError) as exc:
            raise _TransientSmtpError(str(exc)) from exc
        except smtplib.SMTPServerDisconnected as exc:
            raise _TransientSmtpError(str(exc)) from exc
        except smtplib.SMTPException as exc:
            # Non-transient protocol-level rejection (e.g. HELO refused) is
            # treated as inconclusive rather than retried indefinitely.
            return SmtpValidationResult(
                email=email,
                outcome=SmtpValidationOutcome.UNKNOWN,
                mx_host=mx_host,
                smtp_response_code=None,
                detail=f"SMTP protocol error during handshake: {exc}",
            )

    def _to_result(
        self,
        email: str,
        mx_host: str,
        response_code: int,
        response_message: bytes,
    ) -> SmtpValidationResult:
        detail = response_message.decode("utf-8", errors="replace")

        if response_code in (250, 251):
            outcome = SmtpValidationOutcome.MAILBOX_EXISTS
        elif response_code in (550, 551, 553):
            outcome = SmtpValidationOutcome.MAILBOX_NOT_FOUND
        else:
            # Greylisting (4xx) and other ambiguous codes are not treated as
            # evidence either way (PRD §6a.1 — do not overstate confidence).
            outcome = SmtpValidationOutcome.UNKNOWN

        return SmtpValidationResult(
            email=email,
            outcome=outcome,
            mx_host=mx_host,
            smtp_response_code=response_code,
            detail=detail,
        )