"""Voice design relevance rule.

Voice design is creating a voice that belongs to no real speaker: sampling
novel speakers (TacoSpawn), generating a voice from a text description, or
designing one from attribute controls. A paper is accepted when its title
and abstract name a voice design task, mention speech synthesis, and are not
about voice user interfaces or loudspeakers. Cloning an existing speaker and
plain style prompting are out of scope.
"""

import re

from papers_pipeline.models import Paper
from papers_pipeline.topics import TopicDecision

EXCLUDED_PHRASES = (
    "voice user interface",
    "voice interface",
    "loudspeaker",
)

VOICE_DESIGN_PHRASES = (
    "voice design",
    "speaker generation",
    "text-to-voice",
    "text to voice",
    "voice creation",
    "speaker creation",
    "timbre generation",
    "voice description",
    "speaker description",
    "text-described",
    "description-based voice",
    "description-based speaker",
    "natural language description",
    "non-existent speaker",
    "nonexistent speaker",
)

SYNTHESIS_PHRASES = (
    "text-to-speech",
    "text to speech",
    "speech synthesis",
    "speech synthesizer",
    "synthesize speech",
    "synthesizing speech",
    "synthesized speech",
    "synthetic speech",
    "speech generation",
    "voice generation",
    "vocoder",
)

# Bare "TTS" needs word boundaries: a substring match would hit every "https://".
TTS_ACRONYM = re.compile(r"\btts\b")


def accept_topic(paper: Paper) -> TopicDecision:
    """Apply the voice design relevance rule to a paper's title and abstract.

    Args:
        paper: Normalized paper to classify.

    Returns:
        Acceptance decision with the rule that decided it.
    """
    text = f"{paper.title} {paper.abstract}".lower()
    excluded = next((phrase for phrase in EXCLUDED_PHRASES if phrase in text), None)
    if excluded is not None:
        return TopicDecision(False, f"matched excluded term: {excluded}")
    if not any(phrase in text for phrase in VOICE_DESIGN_PHRASES):
        return TopicDecision(False, "missing voice design signal")
    if not (
        any(phrase in text for phrase in SYNTHESIS_PHRASES) or TTS_ACRONYM.search(text)
    ):
        return TopicDecision(False, "missing speech synthesis signal")
    return TopicDecision(True, "accepted")
