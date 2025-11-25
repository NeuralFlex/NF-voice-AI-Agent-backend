import textwrap


def safe_prompt(fstring: str) -> str:
    return textwrap.dedent(fstring).strip()
