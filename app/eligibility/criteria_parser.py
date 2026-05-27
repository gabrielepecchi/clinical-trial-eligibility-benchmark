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
