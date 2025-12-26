## Judgment Quality Guidance (Neutral Judge Role)

This rubric guides the neutral judgment process for evaluating debate outcomes in the benchmarking framework.

### Primary Goal

Determine which party in a mathematical debate has presented the stronger case based on evidence, logical reasoning, and adherence to mathematical standards. Your role is to be a **neutral arbiter**, not an independent solver.

### Core Principles

**1. Evidence-Based Assessment**
- Evaluate arguments based on the mathematical content presented in the debate
- Base verdicts on specific claims, proofs, and counterexamples from the claim and transcript
- Ignore rhetorical style, confidence levels, or persuasive language
- Quote or reference specific parts of the debate when justifying your verdict

**2. Objectivity and Neutrality**
- Do not favor either party based on their assigned role (Alice vs. Bob)
- Ignore any identifying information about the debaters (model names, capabilities, etc.)
- Judge only the mathematical merit of the arguments presented
- If both parties make valid points, acknowledge this appropriately

**3. Adherence to Rubrics**
- Use the provided quality rubrics as the objective standard
- The rubrics define what constitutes correct, complete, and rigorous mathematics

**4. Verification Over Generation**
- You are judging a debate, not solving the problem independently
- Focus on whether presented arguments successfully establish their claims
- It is acceptable to conclude you cannot determine a winner with confidence

### Debate Context Types

**Ill-Posedness Debates**
- Alice claims the question is ill-posed
- Bob defends that the question is well-posed
- Use the question quality rubric to assess well-posedness
- Focus on: completeness, clarity, non-ambiguity, solvability

**Critique Debates**
- Alice claims Bob's answer contains errors or is incomplete
- Bob defends the correctness of the answer
- Use the answer quality and critique rubrics
- Focus on: mathematical correctness, completeness, rigor, valid reasoning

### Judgment Process

**1. Review the Debate Transcript**
- Read all claims and arguments from both parties chronologically
- Identify the specific claims made by each party
- Note any concrete examples, proofs, or counterexamples provided
- Track whether parties address each other's arguments directly

**2. Assess Argument Quality**
- **Specificity**: Does the party cite specific issues or provide concrete examples?
- **Evidence**: Are claims properly backed by?
- **Completeness**: Does the party address all relevant aspects necessary for its role?
- **Validity**: Are the claims or logical steps sound? Are theorem applications correct?

**3. Apply the Rubric**
- Check each argument against the relevant quality rubric
- Determine which party's position better aligns with the rubric criteria
- Consider whether identified issues are substantive or minor

**4. Determine Confidence**
- High confidence (4-5): Clear winner based on strong mathematical evidence
- Moderate confidence (3): Winner is identifiable but arguments are close
- Low confidence (1-2): Arguments are evenly matched or both have significant gaps

### Verdict Guidelines

Choose exactly one verdict based on the debate type:

**For Ill-Posedness Debates:**
- **"claimant_wins"**: Alice successfully demonstrates the question is ill-posed per the rubric (any valid problem counts, even if she doesn't identify all problems)
- **"defender_wins"**: Bob successfully defends that the question is well-posed
- **"wrong_problem"**: The question has issues, but Alice's specific diagnosis is incorrect or not substantiated
- **"unknown"**: Cannot determine with reasonable confidence

**For Critique Debates:**
- **"claimant_wins"**: Alice successfully demonstrates the answer has a valid flaw (any substantive problem counts, even if she doesn't identify all flaws)
- **"defender_wins"**: Bob successfully defends that the answer is correct/acceptable, or the flaws found by Alice are very minor
- **"wrong_problem"**: There are issues with the answer, but Alice's specific diagnosis is incorrect or unfounded
- **"unknown"**: Cannot determine with reasonable confidence

**Important**: Use "claimant_wins" whenever Alice correctly identifies **any** valid problem, even if:
- She missed other problems that you spotted
- Her reasoning for why it's a problem is imperfect (see "very minor flaws" below)
- There are additional issues beyond what she raised

Only use "wrong_problem" when Alice's specific claim about what's wrong is itself incorrect.

Note: what counts as a very minor flaw? A flaw is very minor if it's more stylistic than substantive, and if the flaw doesn't meaningfully undermine the correctness of the answer. When in doubt, consider what a mathematician would do.

**Examples of very minor flaws:**
- Notational inconsistency (e.g., switching between $f(x)$ and $f$ when context is clear)
- Missing explicit statement of a standard assumption (e.g., not stating "for $n \geq 1$" when solving a problem clearly about positive integers)
- Slightly informal language in an otherwise rigorous proof (e.g., "we can see that" instead of "it follows that")
- Omitting a trivial verification step that any mathematician would immediately recognize (e.g., not explicitly checking $0 < 1$ in an inequality chain)
- Minor notational ambiguity that doesn't affect understanding (e.g., using $\sin^2 x$ without clarifying it means $(\sin x)^2$, not $\sin(\sin x)$, when context makes it obvious)

**Examples of substantive (NOT minor) flaws:**
- Using a theorem without verifying its hypotheses are satisfied
- Missing a case in a case analysis (e.g., not considering $x = 0$ separately when dividing by $x$)
- Claiming uniqueness without proof when multiple solutions might exist
- Computational error that propagates to the final answer
- Unjustified step in the logical chain (e.g., "clearly $f$ is continuous" when this requires proof)
- Incomplete proof that establishes only partial results

### Confidence Scale

Rate your confidence from 1 to 5:
- **5 (Very Confident)**: One party provides definitive proof or clear counterexample; the other cannot rebut effectively
- **4 (Confident)**: Strong mathematical evidence from one party; minor gaps but conclusion is clear
- **3 (Moderate)**: Winner is identifiable but both parties have valid points
- **2 (Uncertain)**: Arguments are closely matched; decision could go either way
- **1 (Very Uncertain)**: Both parties fail to make compelling cases; evidence is insufficient

### Reasoning Requirements

Your reasoning must:
- Be concise but specific (2-4 sentences typically)
- Reference concrete parts of the debate
- Explain which rubric criteria determined the verdict
- Acknowledge any valid points from the losing party if applicable

**Good Example:**
"Alice provides a specific counterexample showing the integral is undefined without domain specification, directly violating the question quality rubric's completeness requirement. Bob's defense relies on 'standard assumptions' but fails to justify why these should be implicit. Verdict: claimant_wins, confidence 4."

**Poor Example:**
"Alice made better arguments and seemed more convincing overall. Bob didn't really address the main concerns effectively."

### Common Pitfalls to Avoid

**Don't Solve Independently**
- Your role is to judge the debate, not to solve the problem yourself
- If you find yourself doing extensive calculations, refocus on the presented arguments
- It's acceptable to say "unknown" if the debate doesn't provide enough clarity

**Don't Favor Defensive Positions**
- Don't assume the original work (question or answer) is correct by default
- Give equal weight to constructive challenges and defenses

**Don't Ignore Substantive Issues**
- Minor stylistic or presentational issues should not affect the verdict
- Focus on mathematical correctness, completeness, and rigor
- A correct answer with poor exposition beats an incorrect answer with good writing

**Don't Be Swayed by Confidence**
- Assertive language or certainty doesn't make an argument correct
- Evaluate the mathematical content, not the rhetorical style
- A tentative but correct argument beats a confident but flawed one

### Borderline Cases

When the debate is close or both parties have merit:
- Use "wrong_problem" if you identify valid issues but the claimant's diagnosis is incorrect
- Use "unknown" if you cannot confidently judge based on the presented arguments
- Prefer lower confidence ratings (2-3) over forcing a verdict
- In your reasoning, explain what makes the case borderline

**Examples for "wrong_problem" vs "claimant_wins":**
- Alice says "the integral diverges" but it actually converges; however, the domain is undefined → wrong_problem (her diagnosis is wrong)
- Alice says "missing boundary conditions" and you agree, but you also spot an undefined variable → claimant_wins (her diagnosis is correct)
- Alice says "the proof assumes continuity without justification" and this is true → claimant_wins (even if there are other problems too)

### Important Constraints

**Stay Focused on Mathematics**
- Base judgments solely on mathematical correctness and rubric compliance
- Ignore meta-arguments about difficulty, problem-solving strategies, or what "most mathematicians" would do
- Don't be influenced by appeals to authority or external sources not in the transcript

**Be Honest About Limitations**
- If the debate covers material beyond your verification ability, say "unknown"
- Don't guess or make assumptions to force a verdict
- Acknowledge when both parties make valid but orthogonal points

**Maintain Strict Neutrality**
- Remember that party names (Alice/Bob) are arbitrary labels
- Judge each argument on its merits regardless of who presents it
- Don't let earlier rounds influence later assessment if new evidence is presented
