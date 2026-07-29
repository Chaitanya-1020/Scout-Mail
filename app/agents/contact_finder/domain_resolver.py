"""
Company domain resolver.

Implements: PRD §6.3 (Contact Finder Agent — resolves a company's primary
domain as the foundation for email pattern inference and public-source
scraping), §6a.1 (Layered Confidence Pipeline — domain resolution is the
first, foundational layer other layers depend on), §6a.3 (Caching — resolved
company domains cached, keyed by company name, reused across runs).
Roadmap: Epic 5 - Contact Finder Agent, Story 1 - Company Domain Resolution,
Task 1.

Resolves a company's primary domain using public, non-login-walled methods
(DNS resolution against name-derived candidate domains, with HTTP redirect
following to confirm reachability). Defines the `DomainResolver` interface
and a concrete implementation. Per docs/architecture.md, downstream Contact
Finder collaborators (public-source scraper, pattern generator) depend on
this interface's result type, never on DNS/HTTP libraries directly.

Note: caching resolved domains by company name (PRD §6a.3) is handled by the
caller via the existing `CacheProvider` abstraction
(app/services/cache.py) — this module is pure resolution logic and does not
depend on the cache itself (Single Responsibility, per
docs/coding_guidelines.md).
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from urllib.parse import urlparse

import dns.exception
import dns.resolver
import requests

_REQUEST_TIMEOUT_SECONDS = 10
_DNS_TIMEOUT_SECONDS = 5

# Common corporate suffixes stripped before deriving a domain candidate from
# a company name (e.g. "Acme Inc." -> "acme").
_COMPANY_SUFFIXES = (
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "company",
    "llc",
    "ltd",
    "limited",
    "plc",
    "gmbh",
    "group",
    "holdings",
)

_TLD_CANDIDATES = (".com", ".io", ".ai", ".co", ".net")


class DomainResolutionError(Exception):
    """Raised when a company's domain cannot be resolved by any method."""


@dataclass(frozen=True)
class DomainResolutionResult:
    """Structured outcome of resolving a company's primary domain.

    Every layer downstream in the Contact Finder pipeline (PRD §6a.1) needs
    to know not just the domain but how confidently it was resolved and by
    what method, so this is a first-class result type rather than a bare
    string.
    """

    company_name: str
    domain: str
    source: str
    """How the domain was resolved, e.g. 'known_source_url', 'name_derived_dns',
    'name_derived_http_redirect'. Used as evidence in confidence scoring
    (PRD §6a.1)."""
    confirmed_reachable: bool
    """True if an HTTP request to the domain succeeded (stronger evidence
    than DNS resolution alone)."""
    candidate_domains_tried: list[str]
    """All candidate domains attempted, for auditability (PRD §9)."""


class DomainResolver(ABC):
    """Abstraction over resolving a company's primary domain."""

    @abstractmethod
    def resolve(self, company_name: str, known_source_url: str | None = None) -> DomainResolutionResult:
        """Resolve the primary domain for a company.

        Args:
            company_name: the company's display name (e.g. from a job
                posting's `company_name` field).
            known_source_url: an optional URL already known to belong to the
                company (e.g. a job posting's `jd_url`), used as the
                strongest-confidence resolution path when present.

        Raises:
            DomainResolutionError: if no domain could be resolved by any
                available method.
        """
        raise NotImplementedError


class PublicDomainResolver(DomainResolver):
    """Concrete `DomainResolver` using only public, non-login-walled methods:
    extracting a domain from a known source URL, then falling back to
    DNS/HTTP probing of name-derived candidate domains.
    """

    def resolve(
        self, company_name: str, known_source_url: str | None = None
    ) -> DomainResolutionResult:
        if not company_name or not company_name.strip():
            raise DomainResolutionError("Cannot resolve a domain for an empty company name.")

        candidates_tried: list[str] = []

        if known_source_url:
            domain_from_url = self._extract_domain(known_source_url)
            if domain_from_url:
                candidates_tried.append(domain_from_url)
                return DomainResolutionResult(
                    company_name=company_name,
                    domain=domain_from_url,
                    source="known_source_url",
                    confirmed_reachable=True,
                    candidate_domains_tried=candidates_tried,
                )

        name_slug = self._slugify_company_name(company_name)
        if not name_slug:
            raise DomainResolutionError(
                f"Company name '{company_name}' could not be reduced to a usable slug "
                "for domain candidate generation."
            )

        for tld in _TLD_CANDIDATES:
            candidate_domain = f"{name_slug}{tld}"
            candidates_tried.append(candidate_domain)

            if self._dns_resolves(candidate_domain):
                reachable = self._http_reachable(candidate_domain)
                return DomainResolutionResult(
                    company_name=company_name,
                    domain=candidate_domain,
                    source="name_derived_http_redirect" if reachable else "name_derived_dns",
                    confirmed_reachable=reachable,
                    candidate_domains_tried=candidates_tried,
                )

        raise DomainResolutionError(
            f"Could not resolve a domain for company '{company_name}'. "
            f"Tried: {', '.join(candidates_tried)}."
        )

    def _extract_domain(self, url: str) -> str | None:
        try:
            parsed = urlparse(url if "://" in url else f"https://{url}")
        except ValueError:
            return None

        hostname = parsed.hostname
        if not hostname:
            return None

        return hostname.lower().removeprefix("www.")

    def _slugify_company_name(self, company_name: str) -> str:
        name = company_name.lower().strip()
        name = re.sub(r"[^a-z0-9\s]", "", name)

        tokens = [
            token
            for token in name.split()
            if token not in _COMPANY_SUFFIXES
        ]
        slug = "".join(tokens)
        return slug

    def _dns_resolves(self, domain: str) -> bool:
        resolver = dns.resolver.Resolver()
        resolver.timeout = _DNS_TIMEOUT_SECONDS
        resolver.lifetime = _DNS_TIMEOUT_SECONDS

        try:
            resolver.resolve(domain, "A")
            return True
        except dns.exception.DNSException:
            try:
                resolver.resolve(domain, "AAAA")
                return True
            except dns.exception.DNSException:
                return False

    def _http_reachable(self, domain: str) -> bool:
        for scheme in ("https", "http"):
            try:
                response = requests.head(
                    f"{scheme}://{domain}",
                    timeout=_REQUEST_TIMEOUT_SECONDS,
                    allow_redirects=True,
                )
                if response.status_code < 500:
                    return True
            except requests.RequestException:
                continue
        return False