from pathlib import Path
from typing import List, Optional


def read_guidance(path: str) -> str:
    return Path(path).read_text().strip()


def load_question_guidance(base: str = "guidance") -> str:
    return read_guidance(str(Path(base) / "question_quality.md"))


def load_answer_guidance(base: str = "guidance") -> str:
    return read_guidance(str(Path(base) / "answer_quality.md"))


def load_critique_guidance(base: str = "guidance") -> str:
    return read_guidance(str(Path(base) / "critique_quality.md"))


def build_question_prompt(topic: str, guidance_text: str, previous_questions: Optional[List[str]]) -> str:
    """
    Prompt for generating a challenging mathematics question with solution.
    This implements the tester/questioner role in the benchmarking framework.
    """
    extra = ""
    if previous_questions:
        attempts = "\n".join(f"- {q}" for q in previous_questions)
        extra = (
            "\n\n**Previous Attempts:**\n"
            "The following questions on this topic failed the self-solve gate or meaningfulness check.\n"
            "Generate a materially different, well-posed replacement that avoids these issues:\n"
            f"{attempts}\n"
        )
    return (
        "# Task: Generate a Challenging Mathematics Problem\n\n"
        "You are acting as a **tester** in a benchmarking framework. Your goal is to create a single, "
        "challenging but solvable mathematics problem along with a complete, verifiable solution.\n\n"
        "## Requirements\n\n"
        "- The question should be at an advanced mathematical level, suitable for testing strong reasoning capabilities\n"
        "- It must be fully self-contained, well-posed, and solvable with the information provided\n"
        "- Your answer must be complete, rigorous, and demonstrate that the question is actually solvable\n"
        "- Use standard LaTeX notation for mathematical expressions where appropriate, delineated with $ or $$\n\n"
        "## Quality Rubric (CRITICAL)\n\n"
        f"{guidance_text}\n\n"
        "## Topic\n\n"
        f"Generate a problem in the following domain: **{topic}**\n"
        f"{extra}\n"
        "## Output Format\n\n"
        "Use exactly this structure:\n\n"
        "[QUESTION]\n"
        "<Your problem statement here>\n\n"
        "[ANSWER]\n"
        "<Your complete solution here>\n\n"
        "**Important**: Your solution will be verified first (self-solve gate). If it fails verification, "
        "the question will be rejected without being used to test other models."
    )


def build_answer_prompt(question: str, guidance_text: str) -> str:
    """
    Prompt for answering a mathematics question.
    This implements the testee/answerer role in the benchmarking framework.
    """
    return (
        "# Task: Solve the Mathematics Question\n\n"
        "You are acting as a **testee** in a benchmarking framework. Provide a complete, rigorous answer "
        "to the question below.\n\n"
        "## Answer Quality Requirements\n\n"
        f"{guidance_text}\n\n"
        "## The Question\n\n"
        f"{question}\n\n"
        "## Your Response\n\n"
        "Provide your complete answer below. Use LaTeX notation for mathematical expressions where appropriate, delineated with $ or $$.\n"
        "If the question is ill-posed, explicitly state this and explain why rather than attempting to answer."
    )


def build_self_check_prompt(question: str, answer: str, answer_guidance: str) -> str:
    """
    Prompt for self-critique of an answer during the self-improvement loop.
    """
    return (
        "# Task: Evaluate Your Own Answer\n\n"
        "Review the answer you provided to the question below and assess whether it meets the quality standards.\n\n"
        "## Question\n\n"
        f"{question}\n\n"
        "## Your Answer\n\n"
        f"{answer}\n\n"
        "## Evaluation Rubric\n\n"
        f"{answer_guidance}\n\n"
        "## Required Output Format\n\n"
        "Return ONLY a JSON object with this exact schema (no additional text):\n\n"
        "```json\n"
        "{\n"
        '  "verdict": "pass" | "fail",\n'
        '  "ill_posed": true | false,\n'
        '  "issues": ["<specific issue 1>", "<specific issue 2>", ...],\n'
        '  "improvements": "<actionable guidance for improvement>"\n'
        "}\n"
        "```\n\n"
        "**Field Descriptions:**\n"
        '- `verdict`: "pass" if the answer is correct and complete, "fail" otherwise\n'
        '- `ill_posed`: true if the question itself is unanswerable as stated, false otherwise\n'
        '- `issues`: List of specific problems with the answer (empty list if none)\n'
        '- `improvements`: Short, concrete guidance on how to fix the answer (empty string if verdict is "pass")'
    )


def build_refine_prompt(question: str, answer: str, feedback: str, guidance: str) -> str:
    """
    Prompt for refining an answer based on self-critique feedback.
    """
    return (
        "# Task: Improve Your Answer\n\n"
        "Revise your answer to address the issues identified in the feedback below.\n\n"
        "## Question\n\n"
        f"{question}\n\n"
        "## Current Answer\n\n"
        f"{answer}\n\n"
        "## Feedback\n\n"
        f"{feedback}\n\n"
        "## Quality Standards\n\n"
        f"{guidance}\n\n"
        "## Your Revised Answer\n\n"
        "Provide only the improved answer below (no meta-commentary). "
        "Use LaTeX notation for mathematical expressions where appropriate, delineated with $ or $$."
    )


def build_critique_prompt(question: str, author: str, answer: str, guidance: str) -> str:
    return (
        f"You are critiquing an answer.\nFollow the critique guidance:\n{guidance}\n\n"
        f"Question (by {author}):\n{question}\n\nAnswer:\n{answer}\n\n"
        "Respond ONLY with a JSON object following this schema (no extra text):\n"
        "```\n"
        "{\n"
        '  "verdict": "correct" | "incorrect" | "insufficient" | "obscure",\n'
        '  "notes": "<short text>",\n'
        '  "suggestions": "<optional text>"\n'
        "}\n"
        "```\n"
        'Where "verdict" indicates if the answer is correct, incorrect, insufficient (partially correct), or obscure (unclear),\n'
        "Where necessary, use standard mathematical notation (LaTeX) to express formulas."
    )


def build_critique_self_check(question: str, answer: str, critique: str, guidance: str) -> str:
    return (
        "Review your critique for accuracy and clarity.\n"
        f"Question:\n{question}\n\nAnswer under review:\n{answer}\n\nYour critique:\n{critique}\n\n"
        f"Guidance:\n{guidance}\n\n"
        "Return ONLY a JSON object following this schema (no extra text):\n"
        "```\n"
        "{\n"
        '  "verdict": "pass" | "fail",\n'
        '  "issues": [<string>, ...],\n'
        '  "improvements": "<short text>"\n'
        "}\n"
        "```\n"
        "Where necessary, use standard mathematical notation (LaTeX) to express formulas.\n"
    )


def build_critique_refine(question: str, answer: str, critique: str, feedback: str) -> str:
    return (
        "Rewrite the critique to address the feedback.\n"
        f"Question:\n{question}\n\nAnswer under review:\n{answer}\n\nCurrent critique:\n{critique}\n\n"
        f"Feedback:\n{feedback}\n\nReturn only the improved critique."
    )
