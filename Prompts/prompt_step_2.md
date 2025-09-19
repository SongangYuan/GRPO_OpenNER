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
1. Score ONLY the dimension specified above (0–10 points). Do not score other dimensions.
2. Provide a concrete reason citing evidence from the <analysis> content.
3. Be objective and fair. The score should reflect the analysis quality, not the final NER accuracy.
4. Strictly follow the provided scoring criteria.

## Scoring Criteria (auto-adapted to the dimension)
{__dimension_criteria__}

## Output Format
Strictly output the following JSON only, with no extra text:
{
  "dimension_name": "Dimension name",
  "reason": "Scoring rationale",
  "score": 0-10
}