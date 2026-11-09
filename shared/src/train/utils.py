"""
Utils for training

"""

from typing import List


DEFAULT_QUERY_INSTRUCTION = (
    "Conduct counterfactual reasoning on the query, then retrieve the target "
    "knowledge or reasoning that is useful for answering it."
)


def add_instruction(query: str, instruction: str | None = None) -> str:
    instruction = instruction or DEFAULT_QUERY_INSTRUCTION
    return f"Instruct: {instruction}\nQuery: {query}"


def add_instructions(queries: List[str], instruction: str | None = None) -> list[str]:
    return [add_instruction(query, instruction) for query in queries]
