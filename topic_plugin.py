"""Voice design relevance rule.

Voice design is creating a voice that belongs to no real speaker: sampling
novel speakers (TacoSpawn), generating a voice from a text description, or
designing one from attribute controls. Cloning an existing speaker and plain
style prompting are out of scope.

A paper is accepted when its title and abstract avoid the excluded topics
and either say "speaker generation" (TacoSpawn's name for the task), or
mention speech synthesis and also name a voice design task or describe a
voice in text (a description term plus a voice identity term).
"""

import re

from papers_pipeline.models import Paper
from papers_pipeline.topics import TopicDecision

# "Voice design" also names interface, robot, and assistant voice studies.
EXCLUDED_PHRASES = (
    "voice user interface",
    "voice interface",
    "voice assistant",
    "human-robot",
    "robot voice",
    "loudspeaker",
    "deepfake",
    "spoof",
    "voice detection",
)

# Not "multi-speaker generation", which is multi-speaker synthesis.
SPEAKER_GENERATION = re.compile(r"(?<![\w-])speaker generation")

# Phrases that name the task when speech synthesis is also mentioned.
VOICE_DESIGN_PHRASES = (
    "voice design",
    "voice creation",
    "speaker creation",
    "non-existent speaker",
    "nonexistent speaker",
    "pseudo-speaker generat",
)

# A voice described in text: one of these...
DESCRIPTION_PHRASES = (
    "text-to-voice",
    "text description",
    "textual description",
    "natural language description",
    "natural language prompt",
    "text prompt",
    "description prompt",
    "descriptive prompt",
    "voice description",
    "speaker description",
    "text-described",
    "description-based",
)

# ...together with one of these.
VOICE_IDENTITY_PHRASES = (
    "speaker identity",
    "speaker identities",
    "voice identity",
    "new voices",
    "a new voice",
    "novel voices",
    "a novel voice",
    "new speaker",
    "novel speaker",
    "generating voices",
    "generate voices",
    "voice variability",
    "timbre",
    "voice characteristics",
    "speaker characteristics",
    "speaker attributes",
    "voice attributes",
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
    if SPEAKER_GENERATION.search(text):
        return TopicDecision(True, "accepted")
    if not (
        any(phrase in text for phrase in VOICE_DESIGN_PHRASES)
        or (
            any(phrase in text for phrase in DESCRIPTION_PHRASES)
            and any(phrase in text for phrase in VOICE_IDENTITY_PHRASES)
        )
    ):
        return TopicDecision(False, "missing voice design signal")
    if not (
        any(phrase in text for phrase in SYNTHESIS_PHRASES) or TTS_ACRONYM.search(text)
    ):
        return TopicDecision(False, "missing speech synthesis signal")
    return TopicDecision(True, "accepted")
