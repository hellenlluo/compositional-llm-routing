# Prompts

## Full-Question Prompts

### MuSiQue

```
You are given a set of context paragraphs and a question that requires reasoning over multiple paragraphs to answer. Read the paragraphs carefully and answer the question.

Context:
{for each paragraph: "[title]: [paragraph_text]", separated by blank lines}

Question: {question}

Answer the question concisely based only on the information provided in the context paragraphs.
```

### MoReHopQA

```
You are given a set of context passages and a multi-hop reasoning question. The question may require combining information from more than one passage to arrive at the final answer.

Context:
{for each context entry: "[title]: [sentences joined together]", separated by blank lines}

Question: {question}

Think step by step, then provide a short final answer based only on the context above.
```

### StepCoT

```
You are a radiologist assistant. You are given a radiology report from the {origin} dataset. Read the report carefully and answer the diagnostic question by selecting one of the provided options.

Report:
{report}

Question: {step 7 question}

Options:
{step 7 options, one per line}

Select the single best option letter, and state why.
```

---

## Subtask Prompts

### MuSiQue — Subtask

For subtask i (1-indexed) out of N sub-questions in `question_decomposition`:

```
You are given a set of context paragraphs and a series of sub-questions that build on each other. Answer the current sub-question using the context and any previously answered sub-questions.

Context:
{for each paragraph: "[title]: [paragraph_text]", separated by blank lines}

{if i > 1:}
Previously answered sub-questions:
Q1: {sub-question 1 text}
A1: {sub-question 1 gold answer}
...
Q{i-1}: {sub-question i-1 text}
A{i-1}: {sub-question i-1 gold answer}
{end if}

Current sub-question: {sub-question i text}

Answer based on the context and any previous answers above.
```

### MoReHopQA — Subtask

For subtask i (1-indexed) out of N sub-questions in `question_decomposition`:

```
You are given context passages and a series of sub-questions that together solve a multi-hop reasoning problem. Answer the current sub-question.

Context:
{for each context entry: "[title]: [sentences joined together]", separated by blank lines}

{if i > 1:}
Previously answered sub-questions:
Q1: {sub-question 1 text}
A1: {sub-question 1 gold answer}
...
Q{i-1}: {sub-question i-1 text}
A{i-1}: {sub-question i-1 gold answer}
{end if}

Current sub-question: {sub-question i text}

Answer concisely based on the context and any previous answers above.
```

### StepCoT — Subtask

For step i (1-indexed) out of the 7 steps in `vqa_chain`:

```
You are a radiologist assistant analyzing a report from the {origin} dataset. Answer the current step of a structured diagnostic reasoning chain by selecting one of the provided options. If no finding is relevant, answer "N/A".

Report:
{report}

{if i > 1:}
Previous diagnostic steps:
Step 1: {step 1 question}
Answer: {step 1 gold answer}
...
Step {i-1}: {step i-1 question}
Answer: {step i-1 gold answer}
{end if}

Step {i}: {step i question}

Options:
{step i options, one per line}

Select the single best option letter, or "N/A" if not applicable, and state why.
```

## Evaluator LLM Prompts

### MuSiQue — Evaluator

```
You are a strict evaluator for a multi-hop question answering task.

Task:
Determine whether the model answer is correct given the question and ground truth answer.

Rules:
- The model answer must match the ground truth semantically. Exact wording is not required.
- Any listed alias is equally acceptable as the ground truth answer.
- Numeric answers are correct if the values are equal, regardless of whether written as digits or spelled out (e.g. "4" and "four" are both correct; "5" is not).
- Do NOT accept partially correct answers.
- Do NOT infer missing information.
- If there is any contradiction or incorrect detail, mark it as incorrect.
- If the model expresses uncertainty without committing to an answer, mark it as incorrect.

Question: {question}
Ground Truth Answer: {ground_truth}
Ground Truth Aliases (also acceptable): {answer_aliases}
Model Answer: {prediction}

Respond ONLY in JSON:
{
  "correct": true or false,
  "reasoning": "one or two sentences explaining your decision"
}
```

### MoReHopQA — Evaluator

```
You are a strict evaluator for a multi-hop reasoning task. Many answers are the results of calculations such as letter counts, ASCII values, or date arithmetic.

Task:
Determine whether the model answer is correct given the question and ground truth answer.

Rules:
- The model answer must match the ground truth semantically. Exact wording is not required.
- For answers that are results of calculations (counts, ASCII values, dates, letter manipulations), the value must be exactly correct. Do not accept approximate or close answers.
- Numeric answers are correct if the values are equal, regardless of whether written as digits or spelled out (e.g. "4" and "four" are both correct; "5" is not).
- Do NOT accept partially correct answers.
- Do NOT infer missing information.
- If there is any contradiction or incorrect detail, mark it as incorrect.
- If the model expresses uncertainty without committing to an answer, mark it as incorrect.

Question: {question}
Ground Truth Answer: {ground_truth}
Model Answer: {prediction}

Respond ONLY in JSON:
{
  "correct": true or false,
  "reasoning": "one or two sentences explaining your decision"
}
```

### StepCoT — Evaluator

```
You are a strict evaluator for a medical imaging multiple-choice task.

Task:
Determine whether the model answer matches the ground truth option.

Rules:
- The model must select exactly the same option letter as the ground truth.
- If the ground truth is "N/A", the model answer must contain exactly the string "N/A". Do not accept paraphrases such as "not applicable" or "does not apply".
- Ignore any extra explanation the model provides; evaluate only the selected option letter or "N/A".
- If the model selects a different letter or fails to commit to a single option, mark it as incorrect.

Question: {question}
Options: {options}
Ground Truth Answer: {ground_truth}
Model Answer: {prediction}

Respond ONLY in JSON:
{
  "correct": true or false,
  "reasoning": "one or two sentences explaining your decision"
}
```
