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
        "Create a single very challenging but solvable mathematics problem and provide a full solution."
        "\nFollow the question quality rules:\n"
        f"{guidance_text}\n"
        f"\nTopic: {topic}\n"
        f"{extra}"
        "\nOutput using the tags:\n[QUESTION]\n<problem statement>\n\n[ANSWER]\n<complete answer>"
    )


def build_answer_prompt(question: str, guidance_text: str) -> str:
    return (
        "Solve the mathematics question below.\n"
        "Use a clear chain of reasoning and include the final answer.\n"
        f"Follow the answer quality rules:\n{guidance_text}\n\n"
        f"[QUESTION]\n{question}\n"
    )


def build_self_check_prompt(question: str, answer: str, answer_guidance: str) -> str:
    return (
        "You will grade your own answer to the given question.\n"
        f"Question:\n{question}\n\nAnswer:\n{answer}\n\n"
        f"Use this rubric:\n{answer_guidance}\n\n"
        "Return a JSON object with keys verdict (pass/fail), ill_posed (true/false),"
        " issues (list of strings), and improvements (short text). The last two are optional."
    )


def build_refine_prompt(question: str, answer: str, feedback: str, guidance: str) -> str:
    return (
        "Improve the answer to the question given the feedback.\n"
        f"Question:\n{question}\n\nCurrent answer:\n{answer}\n\nFeedback:\n{feedback}\n\n"
        f"Apply the rubric:\n{guidance}\n\n"
        "Return only the revised answer."
    )


def build_critique_prompt(question: str, author: str, answer: str, guidance: str) -> str:
    return (
        f"You are critiquing an answer.\nFollow the critique guidance:\n{guidance}\n\n"
        f"Question (by {author}):\n{question}\n\nAnswer:\n{answer}\n\n"
        "Respond ONLY with a JSON object containing the keys verdict (correct/incorrect/insufficient/obscure),"
        " notes (short text string, not a list), and suggestions (optional). Do not include any extra text."
    )


def build_critique_self_check(question: str, answer: str, critique: str, guidance: str) -> str:
    return (
        "Review your critique for accuracy and clarity.\n"
        f"Question:\n{question}\n\nAnswer under review:\n{answer}\n\nYour critique:\n{critique}\n\n"
        f"Guidance:\n{guidance}\n\n"
        "Return JSON with keys verdict (pass/fail), issues (list), and improvements (text)."
    )


def build_critique_refine(question: str, answer: str, critique: str, feedback: str) -> str:
    return (
        "Rewrite the critique to address the feedback.\n"
        f"Question:\n{question}\n\nAnswer under review:\n{answer}\n\nCurrent critique:\n{critique}\n\n"
        f"Feedback:\n{feedback}\n\nReturn only the improved critique."
    )
