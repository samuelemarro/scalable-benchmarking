## Judgment Quality Guidance (Neutral Judge Role)

This rubric guides the neutral judgment process for evaluating claims in the benchmarking framework.

### Primary Goal

Determine whether **Alice's original claim is valid** based on the mathematical evidence presented in the debate. Your role is to judge the **claim itself**, not who "won" the debate rhetorically. The debate transcript serves as evidence to help assess whether Alice's claim is substantiated.

For example, if Alice makes a valid claim and Bob acknowledges the issue or provides a fix during the debate, *and you agree with the concession*, this confirms Alice's claim was correct. The verdict should be **claimant_wins** because Alice was right about the issue, even though Bob conceded.

### Core Principles

**1. Judge the Claim, Not the Debate**
- Your task is to evaluate whether **Alice's original claim is valid**
- The debate is evidence that helps you assess the claim, not a competition to "win"
- Bob's acknowledgment of an issue, concession, or proposed fix **confirms** Alice's claim was correct
- Focus on: Was Alice right about the specific problem she identified?

**2. Evidence-Based Assessment**
- Evaluate the mathematical validity of Alice's claim using the debate as evidence
- Base verdicts on specific claims, proofs, and counterexamples from the claim and transcript
- Ignore rhetorical style, confidence levels, or persuasive language
- Quote or reference specific parts of the debate when justifying your verdict

**3. Objectivity and Neutrality**
- Do not favor either party based on their assigned role (Alice vs. Bob)
- Ignore any identifying information about the debaters (model names, capabilities, etc.)
- Judge only the mathematical merit of the claim and supporting evidence
- Bob conceding or fixing an issue is evidence, but not a definitive proof: a party might be incorrectly conceding or addressing pointless issues

**4. Adherence to Rubrics**
- Use the provided quality rubrics as the objective standard
- The rubrics define what constitutes correct, complete, and rigorous mathematics

**5. Verification Over Generation**
- You are judging whether a claim is valid, not solving the problem independently
- Focus on whether Alice's claim is substantiated by the evidence
- It is acceptable to conclude you cannot determine validity with confidence

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

**1. Identify Alice's Original Claim**
- What specific issue did Alice identify in her initial claim?
- Is she claiming the question is ill-posed, or the answer has errors?
- What evidence did she provide for this claim?

**2. Review the Debate Transcript as Evidence**
- Read all arguments from both parties chronologically
- Note any concrete examples, proofs, or counterexamples provided
- Pay special attention to:
  - Does Bob acknowledge the issue Alice raised?
  - Does Bob provide a fix or correction?
  - Does Bob demonstrate the issue doesn't exist?
  - Does Bob show the issue is very minor (stylistic only)?

**3. Evaluate the Claim's Validity**
- Apply the relevant quality rubric to assess whether Alice's claim is substantiated
- Consider whether the issue Alice identified is:
  - **Substantive**: Mathematical error, missing case, unjustified step, ill-posedness
  - **Very minor**: Stylistic, notational, trivial omission (per the rubrics)
- If Bob acknowledged the issue or provided a fix, this **confirms** Alice was right
- If Bob successfully demonstrates the issue doesn't exist or is very minor, this challenges Alice's claim

**4. Determine Confidence**
- High confidence (4-5): Alice's claim is clearly valid or clearly invalid based on strong evidence
- Moderate confidence (3): The claim's validity is identifiable but evidence is not definitive
- Low confidence (1-2): Cannot determine claim validity; evidence is insufficient or contradictory

### Verdict Guidelines

Choose exactly one verdict based on whether **Alice's claim is valid**:

**For Ill-Posedness Debates:**
- **"claimant_wins"**: Alice's claim is correct: the question is ill-posed per the rubric (any valid problem counts, even if she doesn't identify all problems)
  - Use this even if Bob concedes or acknowledges the issue during the debate
- **"defender_wins_incorrect"**: Alice's claim is incorrect: Bob successfully demonstrates the question is well-posed and Alice misidentified the issue
- **"wrong_problem"**: The question has issues, but Alice's specific diagnosis is incorrect or not substantiated
- **"mixed"**: Alice makes multiple claims, some correct and some incorrect (not all claims are valid). Only use for factually incorrect claims, not "all are correct, but some are nitpickings"
- **"unknown"**: Cannot determine whether Alice's claim is valid with reasonable confidence

**For Critique Debates:**
- **"claimant_wins"**: Alice's claim is correct: the answer has a substantive flaw that she correctly identified (any valid problem counts, even if she doesn't identify all flaws)
  - Use this even if Bob concedes, fixes the issue, or provides missing justification during the debate
  - Bob's concession or fix **confirms** Alice was right
- **"defender_wins_incorrect"**: Alice's claim is incorrect: Bob successfully demonstrates the answer is correct and Alice misidentified a problem that doesn't exist
- **"defender_wins_minor"**: Alice's claim is technically correct but about very minor issues only: Bob successfully shows the flaws Alice identified are purely stylistic and don't affect mathematical correctness
- **"wrong_problem"**: There are issues with the answer, but Alice's specific diagnosis is incorrect or unfounded
- **"mixed"**: Alice makes multiple claims, some correct and some incorrect (not all claims are valid). Only use for factually incorrect claims, not "all are correct, but some are nitpickings"
- **"unknown"**: Cannot determine whether Alice's claim is valid with reasonable confidence

**Critical Principle**: Use "claimant_wins" whenever Alice correctly identifies **any** valid problem, even if:
- She missed other problems that you spotted
- Her reasoning for why it's a problem is imperfect (see "very minor flaws" below)
- There are additional issues beyond what she raised
- Bob acknowledges the problem, concedes, or provides a fix during the debate

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

Rate your confidence from 1 to 5 based on how certain you are about **the validity of Alice's claim**:
- **5 (Very Confident)**: Alice's claim is definitively valid or invalid based on clear evidence
- **4 (Confident)**: Strong evidence for or against Alice's claim; minor gaps but conclusion is clear
- **3 (Moderate)**: Claim validity is identifiable but evidence is not definitive
- **2 (Uncertain)**: Evidence is closely balanced; claim validity could go either way
- **1 (Very Uncertain)**: Insufficient evidence to determine claim validity

### Reasoning Requirements

Your reasoning must:
- Be concise but specific (2-4 sentences typically)
- Reference concrete parts of the debate that support your verdict
- Explain which rubric criteria determined whether Alice's claim is valid
- Make clear whether you're judging the claim itself (not who argued better)

**Good Examples:**

*Example 1 (Bob concedes):*
"Alice claimed the answer applied the Dominated Convergence Theorem without verifying the integrable dominating function condition. In round 1, Bob acknowledged this was missing and provided the justification. Bob's acknowledgment confirms Alice's claim was correct: the original answer lacked necessary justification. Verdict: claimant_wins, confidence 5."

*Example 2 (Alice identifies ill-posedness):*
"Alice claimed the domain of variable $x$ is unspecified. Bob argued it's 'clearly the reals,' but this assumption appears nowhere in the problem statement. Per the question quality rubric, all constraints must be explicit. Alice's claim is substantiated. Verdict: claimant_wins, confidence 4."

*Example 3 (Very minor issue):*
"Alice claimed the answer uses inconsistent notation ($f(x)$ vs $f$). Bob correctly notes that context makes the meaning clear and this is a stylistic preference, not a mathematical error. Per the answer quality rubric, minor notational inconsistencies don't invalidate otherwise sound work. Alice's claim identifies a very minor issue only. Verdict: defender_wins_minor, confidence 4."

*Example 4 (Incorrect claim):*
"Alice claimed the proof assumes continuity without justification. Bob demonstrated that continuity was explicitly proven in line 3 using the sequential criterion. Alice misidentified the issue—the justification exists. Verdict: defender_wins_incorrect, confidence 5."

**Poor Example:**
"Alice made better arguments and seemed more convincing overall. Bob didn't really address the main concerns effectively."
*Why poor: Focuses on debate performance rather than claim validity; no reference to rubrics or specific evidence.*

### Common Pitfalls to Avoid

**Don't Judge Debate Performance**
- You are judging whether **Alice's claim is valid**, not who argued better
- Bob conceding or providing a fix **confirms** Alice was right, not that she "won" rhetorically
- Focus on: Was Alice's original claim correct? Not: Who presented arguments more persuasively?

**Don't Solve Independently**
- Your role is to judge the claim's validity using debate evidence, not to solve the problem yourself
- If you find yourself doing extensive calculations, refocus on the presented arguments
- It's acceptable to say "unknown" if the debate doesn't provide enough clarity

**Don't Favor Defensive Positions**
- Don't assume the original work (question or answer) is correct by default
- Give equal weight to constructive challenges and defenses
- Bob's initial position gets no presumption of correctness

**Don't Ignore Substantive Issues**
- Minor stylistic or presentational issues should not affect the verdict
- Focus on mathematical correctness, completeness, and rigor
- A correct answer with poor exposition beats an incorrect answer with good writing

**Don't Be Swayed by Confidence or Persuasiveness**
- Assertive language or certainty doesn't make a claim correct or incorrect
- Evaluate the mathematical validity of Alice's claim, not the rhetorical style
- A tentative but correct claim beats a confident but invalid one

**What if there are multiple claims?**
- If there are multiple claims and **all are correct** (e.g. Alice correctly spots multiple mistakes), use "claimant_wins" - focus on the strongest successful claim
- For example, if Alice correctly spots a substantial and a stylistic error, this is "claimant_wins" because at least one claim is substantive
- If there are multiple claims and **only some are correct** (mixed validity), use "mixed"
- For example, if Alice makes three claims and only one is correct, this is "mixed" not "claimant_wins"

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
