"""Parse raw eligibility text into inclusion and exclusion criteria lists."""

import re


def _clean_lines(lines: list[str]) -> list[str]:
    """Strip bullet prefixes and blank lines."""
    cleaned = []
    for line in lines:
        line = line.strip()
        line = re.sub(r"^[-*•]\s*", "", line)
        if line:
            cleaned.append(line)
    return cleaned


def parse_numeric_range(text: str) -> dict[str, float | None]:
    """Extract lower/upper numeric bounds from a criterion string.

    Supports formats:
      between N and M, N to M, N-M, N – M, from N to M,
      >= N, > N, at least N, N or older,
      <= M, < M, at most M, M or younger.

    Returns:
        Dict with keys 'lower' and 'upper' (float or None).
    """
    t = text.lower()
    lower: float | None = None
    upper: float | None = None

    # between N and M  /  from N to M  /  N to M  /  N - M  /  N – M
    m = re.search(
        r"(?:between\s+|from\s+)?(\d+(?:\.\d+)?)\s*(?:to|-|–|and)\s*(\d+(?:\.\d+)?)",
        t,
    )
    if m:
        lower = float(m.group(1))
        upper = float(m.group(2))
        return {"lower": lower, "upper": upper}

    # minimum-only: >= / > / at least / N or older
    m = re.search(r"(?:>=|≥)\s*(\d+(?:\.\d+)?)", t)
    if m:
        lower = float(m.group(1))
    elif re.search(r">\s*(\d+(?:\.\d+)?)", t):
        m2 = re.search(r">\s*(\d+(?:\.\d+)?)", t)
        lower = float(m2.group(1))  # type: ignore[union-attr]
    m2 = re.search(r"at\s+least\s+(\d+(?:\.\d+)?)", t)
    if m2:
        lower = float(m2.group(1))
    m2 = re.search(r"(\d+(?:\.\d+)?)\s+(?:years?\s+)?or\s+older", t)
    if m2:
        lower = float(m2.group(1))

    # maximum-only: <= / < / at most / N or younger
    m3 = re.search(r"(?:<=|≤)\s*(\d+(?:\.\d+)?)", t)
    if m3:
        upper = float(m3.group(1))
    elif re.search(r"<\s*(\d+(?:\.\d+)?)", t):
        m4 = re.search(r"<\s*(\d+(?:\.\d+)?)", t)
        upper = float(m4.group(1))  # type: ignore[union-attr]
    m4 = re.search(r"at\s+most\s+(\d+(?:\.\d+)?)", t)
    if m4:
        upper = float(m4.group(1))
    m4 = re.search(r"(\d+(?:\.\d+)?)\s+(?:years?\s+)?or\s+younger", t)
    if m4:
        upper = float(m4.group(1))

    return {"lower": lower, "upper": upper}


def parse_eligibility_criteria(text: str) -> dict[str, list[str] | str]:
    """Parse raw eligibility text into inclusion and exclusion criteria.

    Args:
        text: Raw eligibility text from a clinical trial.

    Returns:
        Dictionary with keys:
            - inclusion_criteria: list of inclusion criterion strings
            - exclusion_criteria: list of exclusion criterion strings
            - raw_eligibility: original unmodified text
    """
    inclusion: list[str] = []
    exclusion: list[str] = []

    heading_pattern = re.compile(
        r"^(inclusion criteria|exclusion criteria|inclusion|exclusion)\s*[:\-]?\s*$",
        re.IGNORECASE,
    )

    lines = text.splitlines()
    current_section: str | None = None
    found_heading = False
    no_heading_lines: list[str] = []

    for line in lines:
        if heading_pattern.match(line.strip()):
            found_heading = True
            label = line.strip().lower()
            if "exclusion" in label:
                current_section = "exclusion"
            else:
                current_section = "inclusion"
        else:
            if not found_heading:
                no_heading_lines.append(line)
            elif current_section == "inclusion":
                inclusion.append(line)
            elif current_section == "exclusion":
                exclusion.append(line)

    if not found_heading:
        inclusion = _clean_lines(no_heading_lines)
    else:
        inclusion = _clean_lines(inclusion)
        exclusion = _clean_lines(exclusion)

    return {
        "inclusion_criteria": inclusion,
        "exclusion_criteria": exclusion,
        "raw_eligibility": text,
    }
