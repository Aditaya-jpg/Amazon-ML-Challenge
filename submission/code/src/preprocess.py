"""
Text normalization, domain extraction, and address canonicalization.
"""

import re
import unicodedata
from typing import Optional, Tuple, Set

# Compiled regular expressions for fast execution
RE_WHITESPACE = re.compile(r"\s+")
RE_PUNCTUATION = re.compile(r"[^\w\s]")
RE_DIGITS = re.compile(r"\b\d+\b")
RE_URL_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.)?([a-zA-Z0-9-]+)\.(?:com|in|org|net|co|fr|io|biz|info|gov|edu)(?:/[^\s]*)?",
    re.IGNORECASE,
)

# Common legal suffixes across US, India, and France
LEGAL_SUFFIX_PATTERNS = [
    # Multi-word forms first
    r"\bprivate limited\b",
    r"\bpvt ltd\b",
    r"\bpvt\.?\s*ltd\.?\b",
    r"\bllp\b",
    r"\bllc\b",
    r"\binc\b",
    r"\bincorporated\b",
    r"\bcorp\b",
    r"\bcorporation\b",
    r"\bltd\b",
    r"\blimited\b",
    r"\bcompany\b",
    r"\bco\b",
    # France legal entities
    r"\bsarl\b",
    r"\bsas\b",
    r"\bsa\b",
    r"\beurl\b",
    r"\bsnc\b",
    r"\bsci\b",
    r"\bste\b",
    r"\bsociete\b",
]

RE_LEGAL_SUFFIXES = re.compile(
    r"(?:" + "|".join(LEGAL_SUFFIX_PATTERNS) + r")(?:\s*[\.,]?)*$",
    re.IGNORECASE,
)

# Address abbreviations mapping (bidirectional unification to standard expanded term)
ADDR_ABBREVIATIONS = {
    "rd": "road",
    "st": "street",
    "str": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "bd": "boulevard",
    "bld": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "pl": "place",
    "pkwy": "parkway",
    "hwy": "highway",
    "cir": "circle",
    "ste": "suite",
    "apt": "apartment",
    "bldg": "building",
    "fl": "floor",
    "opp": "opposite",
    "nr": "near",
    # Common US state abbreviations
    "ny": "new york",
    "ca": "california",
    "tx": "texas",
    "fl": "florida",
    "il": "illinois",
    "pa": "pennsylvania",
    "oh": "ohio",
    "ga": "georgia",
    "nc": "north carolina",
    "mi": "michigan",
    "nj": "new jersey",
    "va": "virginia",
    "wa": "washington",
    "az": "arizona",
    "ma": "massachusetts",
}

RE_ADDR_TOKENS = re.compile(r"\b([a-zA-Z]+)\b")


def strip_accents(text: str) -> str:
    """Normalize Unicode characters using NFKD decomposition and strip accents."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def clean_url_or_brand(name: str) -> Tuple[str, bool]:
    """
    Detect if business_name is a URL or domain alias.
    Returns (cleaned_name, is_url_flag).
    """
    if not name:
        return "", False
    match = RE_URL_PATTERN.search(name)
    if match:
        domain_root = match.group(1).replace("-", " ")
        return domain_root, True
    return name, False


def normalize_business_name(name: Optional[str]) -> str:
    """
    Normalize business name:
    1. Strip accents & Unicode diacritics
    2. Lowercase & extract root from domain if URL
    3. Remove punctuation
    4. Strip trailing legal forms (Inc, Ltd, Pvt Ltd, SARL, etc.)
    5. Clean extra whitespace
    """
    if not name or str(name).strip() == "" or str(name).lower() == "nan":
        return ""
    
    text = strip_accents(str(name).lower().strip())
    text, _ = clean_url_or_brand(text)
    
    # Replace & with and
    text = text.replace("&", " and ")
    
    # Remove punctuation
    text = RE_PUNCTUATION.sub(" ", text)
    text = RE_WHITESPACE.sub(" ", text).strip()
    
    # Strip legal suffixes from end of business name
    text = RE_LEGAL_SUFFIXES.sub("", text).strip()
    return text


def normalize_address(address: Optional[str]) -> str:
    """
    Normalize address:
    1. Strip accents & lowercase
    2. Expand common road/street/city abbreviations
    3. Remove punctuation except essential token separators
    4. Normalize whitespace
    """
    if not address or str(address).strip() == "" or str(address).lower() == "nan":
        return ""
    
    text = strip_accents(str(address).lower().strip())
    text = text.replace("&", " and ")
    text = RE_PUNCTUATION.sub(" ", text)
    
    tokens = text.split()
    expanded_tokens = [ADDR_ABBREVIATIONS.get(t, t) for t in tokens]
    return " ".join(expanded_tokens).strip()


def extract_address_numbers(address: Optional[str]) -> Set[str]:
    """
    Extract all numeric sequences (house numbers, postal PIN/zip codes) from address.
    """
    if not address or str(address).strip() == "" or str(address).lower() == "nan":
        return set()
    return set(RE_DIGITS.findall(str(address)))
