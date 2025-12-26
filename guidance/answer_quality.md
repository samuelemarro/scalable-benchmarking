## Answer Correctness and Completeness

This rubric implements the correctness verification predicate. An answer passes only if it satisfies ALL criteria below.

### Core Requirements

**1. Correctness and Directness**
- The answer must be mathematically correct and directly address what the question asks
- Provide an explicit final result in the requested format (exact value, proof, construction, classification, etc.)
- The final answer should be stated clearly and unambiguously, typically in a concluding statement
- All intermediate steps must be logically valid with no mathematical errors

**2. Complete Reasoning Chain**
- Show all substantive reasoning steps from premises to conclusion
- Justify non-trivial claims, theorem applications, and technique choices
- Make logical dependencies explicit: if step B depends on result A, state this clearly
- For proofs: establish all necessary lemmas, handle all required cases, verify all hypotheses of applied theorems

**3. Rigor and Precision**
- Provide validity conditions: domains, convergence criteria, edge cases, boundary behavior
- State assumptions explicitly if you introduce any beyond what the question provides
- Avoid handwaving phrases like "it is clear that," "obviously," "by inspection" unless the claim is genuinely immediate
- Do not appeal to external computation, simulation, or numerical approximation unless the question explicitly permits it

**4. Completeness and Edge Cases**
- Address all parts of the question if it has multiple components
- Consider boundary cases, degenerate scenarios, and special values
- For existence/uniqueness claims: prove both directions
- For "find all" questions: prove you have found every solution

### Handling Ill-Posed Questions

**If the question itself is ill-posed:**
- Do NOT fabricate an answer or proceed as if the question were well-defined
- Explicitly state that the question is ill-posed and cannot be answered as stated
- Cite the specific well-posedness rule(s) violated (refer to the question quality rubric):
  - Missing constraints or underspecified parameters
  - Contradictory premises
  - Ambiguous interpretation
  - Undefined objects or notation
- If possible, suggest what additional information or clarification would make the question answerable

**Example**: "This question is ill-posed because the domain of variable $x$ is not specified, making the integral undefined. The question violates the completeness requirement (missing constraints). To make this answerable, specify whether $x \in \mathbb{R}$, $x \in \mathbb{R}^+$, or some other domain."

### Verification Guidance

Your answer should be structured to facilitate verification:
- Use standard mathematical notation and terminology
- Structure proofs with clear logical flow (e.g., numbered steps, case analysis)
- For computational results: show the calculation path so it can be checked step-by-step
- For constructive proofs: the construction should be explicit enough to verify semi-mechanically
- For existence proofs: provide an explicit example or cite a constructive theorem

### Uncertainty and Limitations

**If genuinely unsure:**
- Acknowledge uncertainty explicitly rather than guessing
- Explain what makes the problem difficult or where your reasoning might be incomplete
- If applicable, propose a partial solution or outline an approach
- Suggest what additional techniques, lemmas, or information might resolve the uncertainty

**Do NOT:**
- Claim certainty when your reasoning has gaps
- Provide contradictory statements and leave it to the reader to figure out which is correct
- Resort to vague or evasive language to mask lack of understanding

**Borderline Cases**
There might be some borderline cases. In such cases, the litmus test should always be: "Would a mathematician, given the information at hand, reasonably conclude the answer to satisfy the above points?"