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
    extra = ""
    if previous_questions:
        attempts = "\n".join(f"- {q}" for q in previous_questions)
        extra = (
            "\nPrevious questions on this topic failed self-answering; generate a materially different, well-posed replacement."
            f"\nEarlier attempts:\n{attempts}\n"
        )
    return (
        "Create a single very challenging but solvable mathematics problem and provide a full solution.\n"
        "This question should be at the level of an advanced mathematician, but still solvable with clear reasoning\n"
        "Where necessary, use standard mathematical notation (LaTeX) to express formulas.\n"
        "\nFollow the question quality rules:\n"
        f"{guidance_text}\n"
        f"\nThe topic of the question to generate is: {topic}\n"
        f"{extra}"
        "\nOutput using the tags:\n[QUESTION]\n<problem statement>\n\n[ANSWER]\n<complete answer>"
    )


def build_answer_prompt(question: str, guidance_text: str) -> str:
    return (
        "Solve the mathematics question below.\n"
        "Use a clear chain of reasoning and include the final answer.\n"
        "Where necessary, use standard mathematical notation (LaTeX) to express formulas.\n"
        f"Follow the answer quality rules:\n{guidance_text}\n\n"
        f"[QUESTION]\n{question}\n"
    )


def build_self_check_prompt(question: str, answer: str, answer_guidance: str) -> str:
    return (
        "You will grade your own answer to the given question.\n"
        f"Question:\n{question}\n\nAnswer:\n{answer}\n\n"
        f"Use this rubric:\n{answer_guidance}\n\n"
        "Where necessary, use standard mathematical notation (LaTeX) to express formulas.\n"
        "Return ONLY a JSON object following this schema (no extra text):\n"
        "```\n"
        "{\n"
        '  "verdict": "pass" | "fail",\n'
        '  "ill_posed": true | false,\n'
        '  "issues": [<string>, ...],\n'
        '  "improvements": "<short text>"\n'
        "}\n"
        "```"
        'Where "verdict" indicates if the answer is correct, "ill_posed" indicates if the question was unanswerable as posed,\n'
        ' "issues" is a list of specific problems with the answer, and "improvements" is guidance on how to improve it.\n'
        
    )


def build_refine_prompt(question: str, answer: str, feedback: str, guidance: str) -> str:
    return (
        "Improve the answer to the question given the feedback.\n"
        f"Question:\n{question}\n\nCurrent answer:\n{answer}\n\nFeedback:\n{feedback}\n\n"
        f"Apply the rubric:\n{guidance}\n\n"
        "Where necessary, use standard mathematical notation (LaTeX) to express formulas.\n"
        "Return only the revised answer."
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
