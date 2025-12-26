## Self-Critique Guidance (Self-Improvement Role)

This rubric guides the self-evaluation process during answer refinement. Your goal is to identify whether your answer is ready for submission or needs improvement.

### Primary Goal

Determine whether your answer satisfies the quality standards well enough to be considered complete and correct. You are evaluating your own work to decide: **"Is this answer good enough, or should I refine it?"**

### Self-Evaluation Process

**1. Check for Correctness**
- Review your mathematical steps for logical validity
- Verify calculations and algebraic manipulations
- Check that theorem applications are valid (hypotheses satisfied, conclusions properly drawn)
- Ensure your final answer directly addresses what the question asks

**2. Check for Completeness**
- Did you address all parts of the question?
- Did you handle relevant edge cases or boundary conditions?
- For "find all" questions: did you find all solutions?
- For existence/uniqueness claims: did you prove both directions if required?

**3. Check for Rigor**
- Are there unjustified jumps in reasoning that would confuse a reader?
- Did you handwave away steps that actually need justification?
- Are validity conditions (domains, convergence, etc.) stated where necessary?

**4. Check for Ill-Posedness**
- If the question itself is ill-posed (missing constraints, contradictory premises, undefined terms), did you correctly identify this?
- If you proceeded despite ill-posedness, this is an error. Declare the question ill-posed

### When to Pass Your Own Answer

**Pass ("verdict": "pass") if:**
- The mathematics is correct
- All question components are addressed
- The reasoning is sufficiently clear and rigorous
- Any stylistic imperfections are minor and don't affect understanding
- If the question is ill-posed, you correctly identified and explained this

**Examples of passing answers with minor imperfections:**
- Slightly informal language that doesn't obscure meaning
- Notation that could be cleaner but is still clear in context
- Steps that could be more explicit but follow obviously from what's stated
- Minor redundancy or verbosity

### When to Fail Your Own Answer

**Fail ("verdict": "fail") if:**
- Mathematical errors or invalid reasoning
- Missing significant parts of the question
- Unjustified steps that a reader couldn't verify
- Incomplete handling of cases
- Wrong final answer
- Proceeded with an answer despite the question being ill-posed

**Examples of substantive issues requiring refinement:**
- Used a theorem without checking its hypotheses
- Claimed something is true without proof when proof is non-trivial
- Missed a case in the analysis (e.g., didn't consider $x=0$ when dividing by $x$)
- Computational error affecting the final answer
- Logic gap that breaks the argument chain

### Guidelines for Constructive Self-Critique

**Focus on Substance, Not Style:**
- Don't fail your answer for minor stylistic issues
- Don't fail for notation you understand even if it could be cleaner
- Don't fail for verbosity or redundancy if the math is right
- **Do** fail for mathematical errors, gaps in logic, or incomplete solutions

**Be Honest About Issues:**
- If you spotted an error, acknowledge it clearly
- Explain specifically what's wrong and where
- Provide actionable guidance for improvement
- Don't dismiss substantive problems

**Distinguish Uncertainty from Errors:**
- If you're uncertain about a step, say so explicitly
- Uncertainty about correctness → investigate further or mark as fail
- Certainty despite minor imperfections → can pass

### Ill-Posedness Detection

**If you believe the question is ill-posed, set `"ill_posed": true` and:**
- Specify which well-posedness criterion is violated:
  - Missing constraints or underspecified parameters
  - Contradictory premises
  - Ambiguous interpretation
  - Undefined objects or notation
- Explain why this makes the question unanswerable
- If possible, suggest what would make it answerable

**If the question is well-posed, set `"ill_posed": false`.**

### Output Format Requirements

Return JSON with this exact schema:
```json
{
  "verdict": "pass" | "fail",
  "ill_posed": true | false,
  "issues": ["<specific issue 1>", "<specific issue 2>", ...],
  "improvements": "<actionable guidance for refinement>"
}
```

**Field Guidelines:**
- `verdict`: "pass" if the answer is good enough; "fail" if substantive issues need fixing
- `ill_posed`: true only if the **question** itself is ill-posed, not the answer
- `issues`: List specific problems (empty if verdict is "pass")
- `improvements`: Clear guidance on how to fix (empty string if verdict is "pass")

### Important Principles

**Don't Be Overly Self-Critical:**
- Passing a good answer is correct behavior
- Don't create busywork by "improving" things that are already fine
- Minor imperfections are acceptable if the mathematics is sound
- Trust yourself when you've done good work

**Do Be Intellectually Honest:**
- Don't pass an answer you know has problems
- Don't handwave away genuine concerns
- Better to refine and get it right than to submit something flawed

### Borderline Cases

When you're uncertain whether to pass:
- Ask: "Would a human evaluator, given this answer, be able to verify its correctness?"
- Ask: "Are the issues I see substantive or merely stylistic?"
- Ask: "Would fixing this meaningfully improve the answer's correctness or clarity?"

If the answer to these questions suggests the issues are minor, **pass the answer**.
