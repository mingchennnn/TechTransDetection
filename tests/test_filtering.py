from __future__ import annotations

from techtransdetection.filtering import _binary_answer, document_prompt, keyword_prompt


def test_prompts_insert_sector_and_evidence() -> None:
    assert "camera" in keyword_prompt(["lens", "sensor"], "camera")
    assert "lens" in keyword_prompt(["lens", "sensor"], "camera")
    assert "camera" in document_prompt("A sensor arrangement", "camera")


def test_binary_response_parser() -> None:
    assert _binary_answer("yes") == "yes"
    assert _binary_answer("No.") == "no"
