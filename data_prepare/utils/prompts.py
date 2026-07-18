#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import abc
import json

from json_repair import repair_json
from lwj_tools.llms.prompt import PromptTemplate

from .pojo import EventSample, DebateSample


def parse_json_response(llm_response: str, return_raw: bool = False):
    """Parse LLM JSON response."""
    if return_raw:
        return llm_response.strip()
    try:
        return repair_json(llm_response, ensure_ascii=False, return_objects=True)
    except Exception:
        try:
            return json.loads(llm_response)
        except Exception:
            return None


class BasePromptTemplate(PromptTemplate, abc.ABC):
    @abc.abstractmethod
    def generate_fn(self, *args):
        raise NotImplementedError

    def parse_fn(self, llm_response: str):
        return parse_json_response(llm_response, False)


class GenerateRoleProfilePromptTemplate(BasePromptTemplate):
    prompt = """
**Task**: Create a detailed life narrative for a specific persona. Return the results in both Chinese and English.

**Character Profile**:
- Personality Archetype: {archetype} ({arch_desc})
- Big Five (OCEAN) Tendencies: {arch_traits}
- Age Group: {age}
- Gender: {gender}
- Occupational Category (RIASEC): {job_category} — focuses on {job_desc}
- Cultural Context (GLOBE): {culture} (Typical countries: {culture_countries})

**Requirements**:
- Name: Give the persona a realistic name appropriate for their cultural context. Ensure unique and diverse names for 
different personas.
- Summary: A 1-sentence summary of who they are.
- Life Narrative (150-200 words):
    - Describe their background, daily life, and work-life balance.
    - Implicit Traits: Do NOT mention Big Five scores or the archetype name. Instead, "show" their personality 
    through behaviors. (e.g., if "Low Openness," describe their preference for routine; if "High Extraversion,
    " describe their social energy).
    - Cultural & Professional Grounding: Ensure their life experiences reflect their age, job category, and cultural 
    values (e.g., collectivism vs. individualism).

**Output JSON Format**:
{{
  "en": {{"name": "[Name]", "summary": "[Summary]", "biography": "[Narrative]"}},
  "zh": {{"name": "[Name]", "summary": "[Summary]", "biography": "[Narrative]"}}
}}
"""

    def generate_fn(self, role_config: dict):
        prompt = self.prompt.format(
            archetype=role_config["archetype"],
            arch_desc=role_config["arch_desc"],
            arch_traits=role_config["arch_traits"],
            age=role_config["age"],
            gender=role_config["gender"],
            job_category=role_config["job_category"],
            job_desc=role_config["job_desc"],
            culture=role_config["culture"],
            culture_countries=role_config["culture_countries"],
        ).strip()
        return prompt


class FilterEventPromptTemplate(PromptTemplate):
    prompt = """
# Character
You are an expert in social psychology and moral philosophy. Your task is to analyze how different personality archetypes respond to events that have a clear **Social Norm Baseline**.

# Context: Four Personality Archetypes
1. **Role Model**: [Low N, High E, High C, High A, High O] - Rational, rule-abiding, highly aligned with social ethics.
2. **Self-Centered**: [Medium N, High E, Low C, Low A, Medium O] - Impulsive, prioritizes personal desires, often deviates from social norms for self-interest.
3. **Reserved**: [Low N, Low E, Medium C, Medium A, Low O] - Traditional, cautious, follows established customs but may lack proactive empathy.
4. **Average**: [Medium N, Medium E, Medium C, Medium A, Medium O] - Moderate, follows the majority, represents typical common sense.

# Input Section
- **Event Description**: "@EVENT@" 
- **Social Consensus**: "@LABEL_MEANING@" (Note: This is the prevailing moral judgment of society.)

# Instruction
1. **Persona Simulation**: Simulate the first-person perspective of each archetype. How would they *honestly* react? A 'Self-Centered' person might secretly support an 'unacceptable' action if it benefits them.
2. **Norm Deviation Analysis**: Specifically analyze how much the 'Self-Centered' and 'Reserved' types deviate from the 'Social Consensus'. 
3. **Divergence Calculation**: Calculate the conflict score (1-10) based on the tension between these personalities and the tension between individuals and the social norm.

# JSON Output Schema
{
    "perspectives": [
        {
            "type": "Role Model / Self-Centered / Reserved / Average",
            "viewpoint_summary": "...",
            "stance_relative_to_norm": "Align / Deviate / Neutral",
            "internal_logic": "Explain the psychological drive behind this stance."
        },
        ...
    ],
    "controversy_analysis": {
        "divergence_score": 1-10,
        "is_highly_discriminative": true/false, 
        "reasoning": "Explain the tension between persona traits and the social consensus."
    }
}
"""

    def _get_label_meaning(self, category: str, label: int):
        mapping = {
            "commonsense": {1: "Clearly Wrong/Unacceptable", 0: "Morally Acceptable"},
            "deontology": {
                1: "Reasonable/Acceptable Excuse",
                0: "Unreasonable/Invalid Excuse",
            },
            "justice": {
                1: "Reasonable/Fair Justification",
                0: "Unreasonable/Unfair Justification",
            },
        }
        return mapping.get(category, {}).get(label, "Unknown Consensus")

    def generate_fn(self, event_sample: EventSample) -> str:
        prompt = self.prompt.replace("@EVENT@", event_sample.event)
        if event_sample.label is not None:
            label_mean = self._get_label_meaning(
                event_sample.category, event_sample.label
            )
            prompt = prompt.replace("@LABEL_MEANING@", label_mean)
        return prompt.strip()


class DebatePromptTemplate(PromptTemplate):
    """Prompt template for teacher debate generation (step 4)."""

    prompt = """
### Persona Context
- **Name**: {name}
- **Archetype**: {archetype}
- **OCEAN**: {ocean_traits}
- **Bio**: {biography}

### Target Event
- **Event**: "{event}"

### High-Fidelity Example (For your reference)
Event: "PersonX prepares for a top-tier university entrance exam."
{{
  "xIntent": {{
    "initial": "to pass the exam",
    "critique": "Too vague. This is an 'AI-helper' response.",
    "status": "ADJUST",
    "rationale": "As a South Asian woman with High Conscientiousness, she views this exam as a 'sacred family duty' rather than just a personal test. Her 'Traditional' job background instills a fear of failure, but her High Openness (O) makes her seek profound insights rather than rote memorization.",
    "final": "to master complex theories to bring unprecedented honor to my lineage"
  }}
}}

### Execution
Perform the 9-dimension audit. Ensure each 'rationale' explicitly connects the traits and culture to a concrete action. Be biased, be specific, and avoid 'polite' genericisms.

### Final Output Structure
{{
  "debate_results": {{
    "xAttr": {{ "initial": "...", "critique": "...", "status": "...", "rationale": "...", "final": "..." }},
    "xReact": {{ "initial": "...", "critique": "...", "status": "...", "rationale": "...", "final": "..." }},
    "xWant": {{ "initial": "...", "critique": "...", "status": "...", "rationale": "...", "final": "..." }},
    "xNeed": {{ "initial": "...", "critique": "...", "status": "...", "rationale": "...", "final": "..." }},
    "xIntent": {{ "initial": "...", "critique": "...", "status": "...", "rationale": "...", "final": "..." }},
    "xEffect": {{ "initial": "...", "critique": "...", "status": "...", "rationale": "...", "final": "..." }},
    "oWant": {{ "initial": "...", "critique": "...", "status": "...", "rationale": "...", "final": "..." }},
    "oReact": {{ "initial": "...", "critique": "...", "status": "...", "rationale": "...", "final": "..." }},
    "oEffect": {{ "initial": "...", "critique": "...", "status": "...", "rationale": "...", "final": "..." }}
  }}
}}
"""

    def generate_fn(self, debate_sample: DebateSample) -> str:
        prompt = self.prompt
        role_profile = debate_sample.role.en
        return prompt.format(
            event=debate_sample.event.event,
            name=role_profile.name,
            archetype=role_profile.archetype,
            ocean_traits=role_profile.arch_traits,
            biography=role_profile.biography,
        ).strip()
