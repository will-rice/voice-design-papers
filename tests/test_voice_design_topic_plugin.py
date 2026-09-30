import pytest

from papers_pipeline.config import TopicConfig
from papers_pipeline.models import Paper
from papers_pipeline.topics import TopicDecision, build_topic_gate
from topic_plugin import accept_topic


@pytest.mark.parametrize(
    ("title", "abstract", "decision"),
    [
        (
            "Speaker Generation",
            "This work explores the task of synthesizing speech in nonexistent"
            ' human-sounding voices. We call this task "speaker generation", and'
            " present TacoSpawn, a recurrent attention-based text-to-speech model.",
            TopicDecision(True, "accepted"),
        ),
        (
            "Text-to-Voice from Captions",
            "We generate a new voice from a natural language description and"
            " synthesize speech with it.",
            TopicDecision(True, "accepted"),
        ),
        (
            "Designing a Voice User Interface",
            "Guidelines for voice design in assistants that use text-to-speech.",
            TopicDecision(False, "matched excluded term: voice user interface"),
        ),
        (
            "Zero-Shot Voice Cloning",
            "A TTS model that clones a target speaker from a 3-second prompt.",
            TopicDecision(False, "missing voice design signal"),
        ),
        (
            "Image Retrieval",
            "Matching images to a natural language description; see https://x.test.",
            TopicDecision(False, "missing speech synthesis signal"),
        ),
    ],
)
def test_accept_topic_applies_voice_design_rule(
    paper: Paper, title: str, abstract: str, decision: TopicDecision
) -> None:
    candidate = paper.model_copy(update={"title": title, "abstract": abstract})

    assert accept_topic(candidate) == decision


def test_papers_yml_plugin_reference_builds_repository_gate(paper: Paper) -> None:
    gate = build_topic_gate(TopicConfig(plugin="topic_plugin:accept_topic"))
    candidate = paper.model_copy(
        update={"title": "Speaker Generation", "abstract": "Novel voices for TTS."}
    )

    assert gate(candidate) == TopicDecision(True, "accepted")
