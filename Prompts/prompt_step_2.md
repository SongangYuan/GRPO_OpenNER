# Task Description
You are a professional NER quality evaluator. Your task is to precisely score the chain-of-thought analysis for a specified evaluation dimension. Judging should use both the original text and the NER query (target entity-type description).

## Evaluation Context
- Original text: {original_text}
- NER query: {ner_query}
- Analysis: <analysis>{analysis_content}</analysis>
- NER result: <ner_result>{ner_result}</ner_result>
- Gold entities: {gold_entities}

## Target Dimension
{target_dimension}

## Scoring Requirements
1. Score ONLY the dimension specified above (0–10 points). Do not score other dimensions. If the analysis is generic or template-like without dimension-specific reasoning, cap the score at 6 or below.
2. Evidence-based judgment: cite concrete evidence from the <analysis> (quotes/paraphrases) and cross-check against the Original text and NER query. If no explicit evidence is provided, cap at 5 or below.
3. Do not reward verbosity or repetition of the question/context. Reward precise, falsifiable, checking-oriented reasoning.
4. Consistency checks: if the analysis contradicts the <ner_result> or {gold_entities} without an explicit, reasonable justification, cap at 3 or below.
5. Penalize hallucinations, speculative claims without support, or ignoring salient counter-evidence in the context.
6. Apply the dimension-specific criteria below strictly; do not inflate scores. Partial fulfillment should receive proportionally lower scores.

### Scoring Anchors (general rubric)
- 9–10: Excellent. Thorough, specific, and well-evidenced reasoning for this dimension; addresses counter-cases and remains consistent with context/result; no unsupported leaps.
- 7–8: Good. Mostly correct and supported, but with minor gaps, missed evidence, or incomplete justification.
- 5–6: Fair. Partially supported or somewhat superficial; noticeable omissions or weak links to evidence.
- 3–4: Poor. Largely unsupported, generic, or partially incorrect; conflicts or gaps that materially weaken the judgment.
- 0–2: Very poor. Off-topic, incoherent, or clearly contradicted by the context; no usable support.

## Scoring Criteria (auto-adapted to the dimension)
{__dimension_criteria__}

## Output Format
Strictly output the following JSON only, with no extra text:
{
  "dimension_name": "Dimension name",
  "reason": "Scoring rationale",
  "score": 0-10
}