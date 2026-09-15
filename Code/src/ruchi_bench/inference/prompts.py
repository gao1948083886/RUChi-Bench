"""Prompt Suite v3 — current five-dataset pilot prompts.

All five tasks share a single system prompt and use task-specific user prompts.
Task routing is strictly by benchmark_task_type, never by payload type.

Naming convention:
  prompt_template_version encodes both the task and the version.
  Changing any prompt string or parameter MUST increment the version.
  Old responses are NEVER reused when version changes.
"""

from __future__ import annotations

from collections.abc import Sequence

from ruchi_bench.schema.enums import BenchmarkTaskType
from ruchi_bench.schema.payloads import MRCPayload, PairPayload, SingleTextPayload
from ruchi_bench.schema.sample import Sample

__all__ = [
    "SYSTEM_PROMPT",
    "PromptTemplate",
    "PromptSuiteV2",
    "prompt_for_sample",
    "prompt_for_standardized_row",
    "MAX_NEW_TOKENS",
]

# ── Shared system prompt ────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "你是一个严格的文本分类器。\n"
    "请严格遵守以下规则：\n"
    "1. 只输出一个允许标签。\n"
    "2. 不要解释理由。\n"
    "3. 不要复述输入。\n"
    "4. 不要添加\"答案是\"\"Label\"\"Answer\"等前缀。\n"
    "5. 不要添加句号、引号、括号或其他标点。\n"
    "6. 不要输出多个标签。\n"
    "7. 即使不确定，也必须从允许标签中选择最符合的一项。"
)

# ── Per-task max_new_tokens ────────────────────────────────────────────────

MAX_NEW_TOKENS: dict[str, int] = {
    "pawsx-binary-v2": 4,
    "xnli-nli-v2": 8,
    "lcqmc-binary-v2": 4,
    "asap-polarity-v2": 4,
    "asap-rating-v1": 4,
    "c3-letter-v2": 4,
}

# ── Prompt template definitions ─────────────────────────────────────────────

_PROMPT_TEMPLATES: dict[str, str] = {
    # PAWS-X — binary paraphrase (1=same, 0=different)
    "pawsx-binary-v2": (
        "任务：判断下面两个句子的核心语义是否相同。\n\n"
        "标签定义：\n"
        "1 = 语义相同\n"
        "0 = 语义不同\n\n"
        "句子A：\n"
        "{sentence1}\n\n"
        "句子B：\n"
        "{sentence2}\n\n"
        "允许输出：0 或 1\n\n"
        "答案："
    ),
    # XNLI — natural language inference (entailment/neutral/contradiction)
    "xnli-nli-v2": (
        "任务：自然语言推断。\n\n"
        "请根据\"前提\"判断\"假设\"与它的逻辑关系。\n\n"
        "标签定义：\n"
        "entailment = 前提能够推出假设\n"
        "neutral = 前提既不能推出也不能否定假设\n"
        "contradiction = 前提与假设矛盾\n\n"
        "前提：\n"
        "{premise}\n\n"
        "假设：\n"
        "{hypothesis}\n\n"
        "允许输出：entailment、neutral 或 contradiction\n\n"
        "答案："
    ),
    # LCQMC — binary question matching (1=same, 0=different)
    "lcqmc-binary-v2": (
        "任务：判断下面两个问题表达的意思是否相同。\n\n"
        "标签定义：\n"
        "1 = 两个问题表达相同或基本相同的意思\n"
        "0 = 两个问题表达不同的意思\n\n"
        "问题A：\n"
        "{question1}\n\n"
        "问题B：\n"
        "{question2}\n\n"
        "允许输出：0 或 1\n\n"
        "答案："
    ),
    # ASAP-Polarity — sentiment (positive/negative)
    "asap-polarity-v2": (
        "任务：判断下面评论的整体情感极性。\n\n"
        "标签定义：\n"
        "positive = 整体评价为正面\n"
        "negative = 整体评价为负面\n\n"
        "评论：\n"
        "{text}\n\n"
        "允许输出：positive 或 negative\n\n"
        "答案："
    ),
    # ASAP — original five-way star rating (labels preserved)
    "asap-rating-v1": (
        "任务：判断下面评论的整体星级评分。\n\n"
        "请根据评论内容，从 1 到 5 星中选择一个最符合的评分。\n"
        "1 = 1 星，2 = 2 星，3 = 3 星，4 = 4 星，5 = 5 星\n\n"
        "评论：\n"
        "{text}\n\n"
        "允许输出：1、2、3、4 或 5\n\n"
        "答案："
    ),
    # C3 — multiple choice (letter only, dynamic options)
    "c3-letter-v2": (
        "任务：根据材料回答一道单项选择题。\n\n"
        "请阅读材料和问题，从给出的选项中选择唯一正确答案。\n\n"
        "材料：\n"
        "{context}\n\n"
        "问题：\n"
        "{question}\n\n"
        "选项：\n"
        "{options_block}\n\n"
        "只输出正确选项的一个大写字母。\n"
        "不要输出选项文本。\n"
        "不要解释理由。\n\n"
        "允许输出：{allowed_letters}\n\n"
        "答案："
    ),
}


def _build_c3_options(options: Sequence[str]) -> tuple[str, str]:
    """Build C3 options block and allowed-letters string.

    Returns (options_block, allowed_letters).
    options is a list of strings, e.g. ["大床房", "双人间", "套房"].
    """
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lines = []
    for i, opt in enumerate(options):
        letter = letters[i]
        lines.append(f"{letter}. {opt}")
    options_block = "\n".join(lines)
    allowed_letters = " ".join(letters[i] for i in range(len(options)))
    return options_block, allowed_letters


class PromptTemplate:
    """Frozen prompt template with version and max_new_tokens."""

    __slots__ = ("version", "system_prompt", "user_template", "max_new_tokens")

    def __init__(self, version: str) -> None:
        if version not in _PROMPT_TEMPLATES:
            raise ValueError(
                f"Unknown prompt_template_version {version!r}. "
                f"Known: {list(_PROMPT_TEMPLATES)}"
            )
        self.version = version
        self.system_prompt = SYSTEM_PROMPT
        self.user_template = _PROMPT_TEMPLATES[version]
        self.max_new_tokens = MAX_NEW_TOKENS[version]

    def format(self, **kwargs: object) -> list[dict[str, str]]:
        """Return a list of OpenAI-compatible message dicts."""
        user_content = self.user_template.format(**kwargs)
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]


class PromptSuiteV2:
    """Factory for PromptTemplate by task type."""

    __slots__ = ()

    # ── Per-task versions ──────────────────────────────────────────────────
    PAWSX = "pawsx-binary-v2"
    XNLI = "xnli-nli-v2"
    LCQMC = "lcqmc-binary-v2"
    ASAP = "asap-polarity-v2"
    ASAP_RATING = "asap-rating-v1"
    C3 = "c3-letter-v2"

    @staticmethod
    def from_task_type(task_type: BenchmarkTaskType) -> PromptTemplate:
        """Resolve a BenchmarkTaskType enum value to its PromptTemplate.

        Raises ValueError if the task type is unknown.
        """
        VERSION_MAP = {
            BenchmarkTaskType.PAIR_PARAPHRASE: PromptSuiteV2.PAWSX,
            BenchmarkTaskType.NLI: PromptSuiteV2.XNLI,
            BenchmarkTaskType.QUESTION_MATCHING: PromptSuiteV2.LCQMC,
            BenchmarkTaskType.SENTIMENT_POLARITY: PromptSuiteV2.ASAP,
            BenchmarkTaskType.MULTIPLE_CHOICE_MRC: PromptSuiteV2.C3,
        }
        version = VERSION_MAP.get(task_type)
        if version is None:
            raise ValueError(
                f"Unknown benchmark_task_type {task_type!r}. "
                f"Valid values: {list(VERSION_MAP)}"
            )
        return PromptTemplate(version)


def prompt_for_sample(
    sample: object, *, is_clean: bool = True
) -> tuple[list[dict[str, str]], str, int]:
    """Build messages for a sample using Prompt Suite v2.

    Returns
    -------
    (messages, prompt_template_version, max_new_tokens)

    Raises
    ------
    TypeError
        If the sample is not a Sample instance.
    ValueError
        If the sample's benchmark_task_type is unknown, or a corrupted request
        has no corrupted payload.

    The sample's benchmark_task_type field is the SOLE routing criterion.
    Payload type is never consulted.
    """
    if not isinstance(sample, Sample):
        raise TypeError(f"Expected Sample, got {type(sample).__name__!r}")

    payload = sample.clean_payload if is_clean else sample.corrupted_payload
    if payload is None:
        raise ValueError(
            f"Sample {sample.sample_id!r} has no corrupted_payload for a "
            "corrupted inference request"
        )

    task_type = sample.benchmark_task_type

    # Route by task type — the ONLY permitted routing criterion
    if task_type is BenchmarkTaskType.PAIR_PARAPHRASE:
        if not isinstance(payload, PairPayload):
            raise TypeError("PAIR_PARAPHRASE requires a PairPayload")
        pt = PromptSuiteV2.from_task_type(task_type)
        messages = pt.format(sentence1=payload.text_a, sentence2=payload.text_b)
        return messages, pt.version, pt.max_new_tokens

    elif task_type is BenchmarkTaskType.NLI:
        if not isinstance(payload, PairPayload):
            raise TypeError("NLI requires a PairPayload")
        pt = PromptSuiteV2.from_task_type(task_type)
        messages = pt.format(premise=payload.text_a, hypothesis=payload.text_b)
        return messages, pt.version, pt.max_new_tokens

    elif task_type is BenchmarkTaskType.QUESTION_MATCHING:
        if not isinstance(payload, PairPayload):
            raise TypeError("QUESTION_MATCHING requires a PairPayload")
        pt = PromptSuiteV2.from_task_type(task_type)
        messages = pt.format(question1=payload.text_a, question2=payload.text_b)
        return messages, pt.version, pt.max_new_tokens

    elif task_type is BenchmarkTaskType.SENTIMENT_POLARITY:
        if not isinstance(payload, SingleTextPayload):
            raise TypeError("SENTIMENT_POLARITY requires a SingleTextPayload")
        pt = PromptSuiteV2.from_task_type(task_type)
        messages = pt.format(text=payload.text_a)
        return messages, pt.version, pt.max_new_tokens

    elif task_type is BenchmarkTaskType.MULTIPLE_CHOICE_MRC:
        if not isinstance(payload, MRCPayload):
            raise TypeError("MULTIPLE_CHOICE_MRC requires an MRCPayload")
        pt = PromptSuiteV2.from_task_type(task_type)
        options_block, allowed_letters = _build_c3_options(payload.options)
        messages = pt.format(
            context=payload.context,
            question=payload.question,
            options_block=options_block,
            allowed_letters=allowed_letters,
        )
        return messages, pt.version, pt.max_new_tokens

    else:
        # Explicitly fail — do NOT silently fall back
        raise ValueError(
            f"Unrecognized benchmark_task_type {task_type!r} "
            f"for sample {sample.sample_id!r}. "
            f"Cannot fall back to paraphrase or any other prompt."
        )


def prompt_for_standardized_row(
    row: dict[str, object], *, is_clean: bool = True
) -> tuple[list[dict[str, str]], str, int]:
    """Build a prompt directly from the current pilot JSON envelope.

    The current pilot intentionally preserves ASAP's official five-way star
    labels. This adapter therefore avoids the historical ``Sample`` model,
    whose ASAP branch describes the retired derived polarity projection.
    """
    task = str(row.get("task_type", ""))
    payload_value = row.get("clean_payload" if is_clean else "corrupted_payload")
    if not isinstance(payload_value, dict):
        raise ValueError("standardized row is missing the requested payload")
    payload = payload_value

    if task == "pair_paraphrase":
        pt = PromptTemplate(PromptSuiteV2.PAWSX)
        return (
            pt.format(sentence1=str(payload["text_a"]), sentence2=str(payload["text_b"])),
            pt.version,
            pt.max_new_tokens,
        )
    if task == "nli":
        pt = PromptTemplate(PromptSuiteV2.XNLI)
        return (
            pt.format(premise=str(payload["text_a"]), hypothesis=str(payload["text_b"])),
            pt.version,
            pt.max_new_tokens,
        )
    if task == "question_matching":
        pt = PromptTemplate(PromptSuiteV2.LCQMC)
        return (
            pt.format(question1=str(payload["text_a"]), question2=str(payload["text_b"])),
            pt.version,
            pt.max_new_tokens,
        )
    if task == "multiple_choice_mrc":
        options = payload.get("options")
        if not isinstance(options, (list, tuple)):
            raise ValueError("standardized C3 row has invalid options")
        pt = PromptTemplate(PromptSuiteV2.C3)
        options_block, allowed_letters = _build_c3_options(
            [str(option) for option in options]
        )
        return (
            pt.format(
                context=str(payload["context"]),
                question=str(payload["question"]),
                options_block=options_block,
                allowed_letters=allowed_letters,
            ),
            pt.version,
            pt.max_new_tokens,
        )
    if task == "sentiment_rating":
        pt = PromptTemplate(PromptSuiteV2.ASAP_RATING)
        return pt.format(text=str(payload["text_a"])), pt.version, pt.max_new_tokens

    raise ValueError(f"Unknown standardized task_type: {task!r}")
