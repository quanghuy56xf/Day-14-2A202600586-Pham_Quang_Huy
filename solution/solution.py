"""
Day 14 — AI Evaluation & Benchmarking Pipeline
AICB-P1: AI Practical Competency Program, Phase 1

Key concepts from lecture:
    - Evaluation = Scientific Method for AI (Hypothesis → Experiment → Measure → Conclude → Iterate)
    - 4 nhóm metrics: Task Completion, Answer Quality, RAG-Specific, Business
    - RAG pipeline metrics: Context Recall → Context Precision → Faithfulness → Answer Relevancy
    - LLM-as-Judge: rubric scoring 1-5, detect bias (positional, verbosity, self-preference)
    - Golden dataset: stratified sampling (5 Easy + 7 Medium + 5 Hard + 3 Adversarial)
    - Failure taxonomy: hallucination, irrelevant, incomplete, off_topic, refusal
    - 5 Whys method for root cause analysis
    - CI/CD integration: eval as quality gate (score < threshold = block deploy)
    - Continuous Improvement Loop: Evaluate → Analyze → Improve → Augment → Repeat

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change class/function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Task 1 — Data Models (Golden Dataset + Evaluation Results)
# ---------------------------------------------------------------------------

@dataclass
class QAPair:
    """
    A question-answer pair for evaluation (part of the Golden Dataset).

    From lecture: Golden dataset cần có:
        - question: câu hỏi user
        - ground_truth (expected_answer): expert-written expected answer
        - context: source documents cần retrieve
        - metadata: difficulty (easy/medium/hard), category, source_docs

    Fields:
        question:        The question to answer.
        expected_answer: The reference/ground-truth answer (expert-written).
        context:            Source context (may be empty string if not applicable).
        metadata:           Optional metadata dict (difficulty, category, etc.).
        retrieved_contexts: List of retrieved chunks (ORDER = retriever rank).
                            Used by the retrieval-side metrics (Task 2b).
    """
    question: str
    expected_answer: str
    context: str = ""
    metadata: dict = field(default_factory=dict)
    retrieved_contexts: list = field(default_factory=list)


@dataclass
class EvalResult:
    """
    Evaluation result for a single Q&A pair.

    From lecture - RAG metrics pipeline:
        Question → Retriever → Context → Generator → Answer
        Each step has a metric: Context Recall, Context Precision, Faithfulness, Answer Relevancy

    From lecture - Score interpretation:
        0.8-1.0: Good (Monitor, maintain)
        0.6-0.8: Needs work (Analyze failures, iterate)
        < 0.6: Significant issues (Deep investigation required)

    Fields:
        qa_pair:        The original QAPair.
        actual_answer:  What the agent actually returned.
        faithfulness:   Float 0-1, how grounded the answer is in context.
        relevance:      Float 0-1, how relevant the answer is to the question.
        completeness:   Float 0-1, how complete the answer is vs expected.
        passed:         True if all three scores >= 0.5.
        failure_type:   None if passed, otherwise one of:
                        "hallucination", "irrelevant", "incomplete", "off_topic".
        context_precision: Float 0-1 or None — quality of retrieval ranking.
        context_recall:    Float 0-1 or None — coverage of expected by context.
                        (Both stay None unless retrieved chunks are supplied;
                         they are NOT part of overall_score().)
    """
    qa_pair: QAPair
    actual_answer: str
    faithfulness: float
    relevance: float
    completeness: float
    passed: bool
    failure_type: str | None = None
    context_precision: float | None = None
    context_recall: float | None = None

    def overall_score(self) -> float:
        """Compute the average of faithfulness, relevance, and completeness.

        Returns:
            (faithfulness + relevance + completeness) / 3.0
        """
        return (self.faithfulness + self.relevance + self.completeness) / 3.0


# ---------------------------------------------------------------------------
# Task 2 — RAGAS Evaluator (Simplified word-overlap heuristic)
# ---------------------------------------------------------------------------

STOPWORDS: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "as", "by", "and", "or",
    "it", "its", "this", "that", "these", "those", "from", "into", "than",
}


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokenization, ignoring punctuation and stopwords."""
    if not text:
        return set()
    tokens = re.findall(r"\b\w+\b", text.lower())
    return {t for t in tokens if t not in STOPWORDS}


def _overlap_score(numerator_tokens: set[str], denominator_tokens: set[str]) -> float:
    """Compute |num ∩ denom| / |denom|, clamped to [0, 1]."""
    if not denominator_tokens:
        return 1.0
    if not numerator_tokens:
        return 0.0
    overlap = len(numerator_tokens & denominator_tokens) / len(denominator_tokens)
    return max(0.0, min(1.0, overlap))


class RAGASEvaluator:
    """
    Evaluates RAG pipeline outputs using RAGAS-inspired heuristics.

    All metrics use word overlap rather than LLM calls for simplicity.
    Replace with actual LLM-based evaluation in production.
    """

    def evaluate_faithfulness(self, answer: str, context: str) -> float:
        """
        Measure how grounded the answer is in the context.

        Heuristic:
            faithfulness = |answer_tokens ∩ context_tokens| / |answer_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if answer is empty.
        """
        if not answer.strip():
            return 1.0
        answer_tokens = _tokenize(answer)
        context_tokens = _tokenize(context)
        if not answer_tokens:
            return 1.0
        overlap = len(answer_tokens & context_tokens) / len(answer_tokens)
        return max(0.0, min(1.0, overlap))

    def evaluate_relevance(self, answer: str, question: str) -> float:
        """
        Measure how relevant the answer is to the question.

        Heuristic:
            relevance = |answer_tokens ∩ question_tokens| / |question_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if question is empty.
        """
        if not question.strip():
            return 1.0
        answer_tokens = _tokenize(answer)
        question_tokens = _tokenize(question)
        if not question_tokens:
            return 1.0
        overlap = len(answer_tokens & question_tokens) / len(question_tokens)
        return max(0.0, min(1.0, overlap))

    def evaluate_completeness(self, answer: str, expected: str) -> float:
        """
        Measure how well the answer covers the expected answer.

        Heuristic:
            completeness = |answer_tokens ∩ expected_tokens| / |expected_tokens|
            Clamp to [0.0, 1.0]. Return 1.0 if expected is empty.
        """
        if not expected.strip():
            return 1.0
        answer_tokens = _tokenize(answer)
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0
        overlap = len(answer_tokens & expected_tokens) / len(expected_tokens)
        return max(0.0, min(1.0, overlap))

    def evaluate_context_recall(self, contexts: list[str], expected: str) -> float:
        """Context Recall — union coverage of expected answer by retrieved chunks."""
        if not expected.strip():
            return 1.0
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0
        union_tokens: set[str] = set()
        for chunk in contexts:
            union_tokens |= _tokenize(chunk)
        return _overlap_score(expected_tokens & union_tokens, expected_tokens)

    def evaluate_context_precision(
        self,
        contexts: list[str],
        expected: str,
        relevance_threshold: float = 0.1,
    ) -> float:
        """Context Precision — rank-aware Average Precision (AP@K)."""
        if not expected.strip():
            return 1.0
        expected_tokens = _tokenize(expected)
        if not expected_tokens:
            return 1.0
        if not contexts:
            return 0.0

        relevant_flags: list[bool] = []
        for chunk in contexts:
            chunk_tokens = _tokenize(chunk)
            coverage = len(chunk_tokens & expected_tokens) / len(expected_tokens)
            relevant_flags.append(coverage >= relevance_threshold)

        num_relevant = sum(relevant_flags)
        if num_relevant == 0:
            return 0.0

        ap_sum = 0.0
        relevant_so_far = 0
        for k, is_relevant in enumerate(relevant_flags, start=1):
            if is_relevant:
                relevant_so_far += 1
                ap_sum += relevant_so_far / k

        return ap_sum / num_relevant

    def run_full_eval(
        self,
        answer: str,
        question: str,
        context: str,
        expected: str,
    ) -> EvalResult:
        """
        Run all three evaluations and combine into an EvalResult.

        passed = True if all three scores >= 0.5.

        failure_type determination (first match wins):
            faithfulness < 0.3  → "hallucination"
            relevance < 0.3     → "irrelevant"
            completeness < 0.3  → "incomplete"
            otherwise if failed → "off_topic"
        """
        faithfulness = self.evaluate_faithfulness(answer, context)
        relevance = self.evaluate_relevance(answer, question)
        completeness = self.evaluate_completeness(answer, expected)

        passed = faithfulness >= 0.5 and relevance >= 0.5 and completeness >= 0.5

        failure_type: str | None = None
        if not passed:
            if faithfulness < 0.3:
                failure_type = "hallucination"
            elif relevance < 0.3:
                failure_type = "irrelevant"
            elif completeness < 0.3:
                failure_type = "incomplete"
            else:
                failure_type = "off_topic"

        qa_pair = QAPair(
            question=question,
            expected_answer=expected,
            context=context,
        )

        return EvalResult(
            qa_pair=qa_pair,
            actual_answer=answer,
            faithfulness=faithfulness,
            relevance=relevance,
            completeness=completeness,
            passed=passed,
            failure_type=failure_type,
        )


# ---------------------------------------------------------------------------
# Reranking helper (used by Exercise 3.5 — boosting Context Precision)
# ---------------------------------------------------------------------------

def rerank_by_overlap(contexts: list[str], query: str) -> list[str]:
    """Sort chunks by word overlap with the query, most-overlapping first."""
    query_tokens = _tokenize(query)
    return sorted(
        contexts,
        key=lambda c: len(_tokenize(c) & query_tokens),
        reverse=True,
    )


# ---------------------------------------------------------------------------
# Task 3 — LLM Judge
# ---------------------------------------------------------------------------

class LLMJudge:
    """
    Uses an LLM to score AI responses according to a rubric.
    """

    def __init__(self, judge_llm_fn: Callable[[str], str]) -> None:
        self.judge_llm_fn = judge_llm_fn

    def score_response(
        self,
        question: str,
        answer: str,
        rubric: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Score an AI response using the judge LLM.

        Returns:
            {"scores": dict[str, float], "reasoning": str}
        """
        rubric_lines = "\n".join(
            f"- {criterion}: {description}"
            for criterion, description in rubric.items()
        )
        prompt = (
            f"Question: {question}\n"
            f"Answer: {answer}\n\n"
            f"Score the answer on each criterion (0.0 to 1.0):\n"
            f"{rubric_lines}\n\n"
            f"Respond with JSON mapping criterion names to scores."
        )

        raw_response = self.judge_llm_fn(prompt)
        default_scores = {criterion: 0.5 for criterion in rubric}

        try:
            parsed = json.loads(raw_response)
            if isinstance(parsed, dict):
                scores = {
                    criterion: float(parsed.get(criterion, 0.5))
                    for criterion in rubric
                }
                return {"scores": scores, "reasoning": raw_response}
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        return {"scores": default_scores, "reasoning": raw_response}

    def detect_bias(self, scores_batch: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Detect potential bias patterns in a batch of judge scores.

        Returns:
            {"positional_bias": bool, "leniency_bias": bool, "severity_bias": bool}
        """
        if not scores_batch:
            return {
                "positional_bias": False,
                "leniency_bias": False,
                "severity_bias": False,
            }

        all_scores: list[float] = []
        first_scores: list[float] = []
        rest_scores: list[float] = []

        for i, entry in enumerate(scores_batch):
            criterion_scores = list(entry.get("scores", {}).values())
            all_scores.extend(criterion_scores)
            if i == 0:
                first_scores.extend(criterion_scores)
            else:
                rest_scores.extend(criterion_scores)

        avg_all = sum(all_scores) / len(all_scores) if all_scores else 0.0
        avg_first = sum(first_scores) / len(first_scores) if first_scores else 0.0
        avg_rest = sum(rest_scores) / len(rest_scores) if rest_scores else 0.0

        positional_bias = (
            len(scores_batch) > 1
            and avg_first > avg_rest + 0.1
        )
        leniency_bias = avg_all > 0.8
        severity_bias = avg_all < 0.3

        return {
            "positional_bias": positional_bias,
            "leniency_bias": leniency_bias,
            "severity_bias": severity_bias,
        }


# ---------------------------------------------------------------------------
# Task 4 — Benchmark Runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """
    Runs a full evaluation benchmark.
    """

    def run(
        self,
        qa_pairs: list[QAPair],
        agent_fn: Callable[[str], str],
        evaluator: RAGASEvaluator,
    ) -> list[EvalResult]:
        """Run all QA pairs through the agent and evaluate each result."""
        results: list[EvalResult] = []
        for pair in qa_pairs:
            actual_answer = agent_fn(pair.question)
            result = evaluator.run_full_eval(
                actual_answer,
                pair.question,
                pair.context or "",
                pair.expected_answer,
            )
            result.qa_pair = pair
            result.actual_answer = actual_answer
            if pair.retrieved_contexts:
                result.context_recall = evaluator.evaluate_context_recall(
                    pair.retrieved_contexts, pair.expected_answer
                )
                result.context_precision = evaluator.evaluate_context_precision(
                    pair.retrieved_contexts, pair.expected_answer
                )
            results.append(result)
        return results

    def generate_report(self, results: list[EvalResult]) -> dict[str, Any]:
        """Generate an aggregate report from evaluation results."""
        if not results:
            return {
                "total": 0,
                "passed": 0,
                "pass_rate": 0.0,
                "avg_faithfulness": 0.0,
                "avg_relevance": 0.0,
                "avg_completeness": 0.0,
                "failure_types": {},
            }

        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failure_types: dict[str, int] = {}
        for r in results:
            if r.failure_type:
                failure_types[r.failure_type] = failure_types.get(r.failure_type, 0) + 1

        return {
            "total": total,
            "passed": passed,
            "pass_rate": passed / total,
            "avg_faithfulness": sum(r.faithfulness for r in results) / total,
            "avg_relevance": sum(r.relevance for r in results) / total,
            "avg_completeness": sum(r.completeness for r in results) / total,
            "failure_types": failure_types,
        }

    def run_regression(self, new_results: list, baseline_results: list) -> dict:
        """Compare new evaluation results against a baseline."""
        def _avg(results: list, metric: str) -> float:
            if not results:
                return 0.0
            return sum(getattr(r, metric) for r in results) / len(results)

        metrics = ["faithfulness", "relevance", "completeness"]
        new_avgs = {m: _avg(new_results, m) for m in metrics}
        baseline_avgs = {m: _avg(baseline_results, m) for m in metrics}

        regressions = [
            m for m in metrics
            if baseline_avgs[m] - new_avgs[m] > 0.05
        ]

        return {
            "new_avg_faithfulness": new_avgs["faithfulness"],
            "new_avg_relevance": new_avgs["relevance"],
            "new_avg_completeness": new_avgs["completeness"],
            "baseline_avg_faithfulness": baseline_avgs["faithfulness"],
            "baseline_avg_relevance": baseline_avgs["relevance"],
            "baseline_avg_completeness": baseline_avgs["completeness"],
            "regressions": regressions,
            "passed": len(regressions) == 0,
        }

    def identify_failures(
        self,
        results: list[EvalResult],
        threshold: float = 0.5,
    ) -> list[EvalResult]:
        """Return EvalResults where any score is below threshold."""
        return [
            r for r in results
            if r.faithfulness < threshold
            or r.relevance < threshold
            or r.completeness < threshold
        ]


# ---------------------------------------------------------------------------
# Task 5 — Failure Analyzer
# ---------------------------------------------------------------------------

class FailureAnalyzer:
    """
    Analyzes failed evaluation results to identify patterns and suggest fixes.
    """

    def categorize_failures(
        self, failures: list[EvalResult]
    ) -> dict[str, int]:
        """Count failures by failure_type."""
        categories: dict[str, int] = {}
        for failure in failures:
            if failure.failure_type:
                categories[failure.failure_type] = (
                    categories.get(failure.failure_type, 0) + 1
                )
        return categories

    def find_root_cause(self, failure: EvalResult) -> str:
        """Suggest a root cause based on which score is lowest."""
        scores = {
            "faithfulness": failure.faithfulness,
            "relevance": failure.relevance,
            "completeness": failure.completeness,
        }
        lowest = min(scores, key=scores.get)  # type: ignore[arg-type]
        min_score = scores[lowest]

        tied = [k for k, v in scores.items() if v == min_score]
        if len(tied) > 1 and min_score < 0.5:
            return "Multiple issues detected — review full pipeline"

        if lowest == "faithfulness":
            return "Context is missing or irrelevant — improve retrieval"
        if lowest == "relevance":
            return "Answer does not address the question — improve prompt clarity"
        return "Answer is missing key information — increase context window or improve generation"

    def generate_improvement_log(self, failures: list, suggestions: list[str]) -> str:
        """Generate a Markdown table logging failures and improvement actions."""
        header = (
            "| Failure ID | Type | Root Cause | Suggested Fix | Status |\n"
            "|------------|------|------------|---------------|--------|"
        )
        rows: list[str] = []
        for i, failure in enumerate(failures):
            fix = suggestions[i] if i < len(suggestions) else "Review and address root cause"
            failure_type = failure.failure_type or "unknown"
            root_cause = self.find_root_cause(failure)
            rows.append(
                f"| F{i + 1:03d} | {failure_type} | {root_cause} | {fix} | Open |"
            )
        return header + "\n" + "\n".join(rows)

    def generate_improvement_suggestions(
        self, failures: list[EvalResult]
    ) -> list[str]:
        """Generate a prioritized list of improvement suggestions."""
        if not failures:
            return []

        categories = self.categorize_failures(failures)
        suggestions_map = {
            "hallucination": "Implement hallucination checker to filter unsupported claims",
            "irrelevant": "Add few-shot examples showing on-topic answers to improve relevance",
            "incomplete": "Increase chunk size in RAG pipeline to reduce context fragmentation",
            "off_topic": "Improve intent detection and routing to keep answers on topic",
            "refusal": "Relax overly strict guardrails for in-scope questions",
        }

        suggestions: list[str] = []
        for failure_type, _count in sorted(
            categories.items(), key=lambda x: x[1], reverse=True
        ):
            if failure_type in suggestions_map:
                suggestions.append(suggestions_map[failure_type])

        default_suggestions = [
            "Increase chunk size in RAG pipeline to reduce context fragmentation",
            "Add few-shot examples showing complete answers to improve completeness",
            "Implement hallucination checker to filter unsupported claims",
        ]
        while len(suggestions) < 3:
            for s in default_suggestions:
                if s not in suggestions:
                    suggestions.append(s)
                if len(suggestions) >= 3:
                    break

        return suggestions


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    qa_pairs = [
        QAPair(
            question="What is RAG?",
            expected_answer="RAG stands for Retrieval-Augmented Generation, which combines retrieval with text generation.",
            context="RAG is a technique that retrieves relevant documents and uses them to ground LLM generation.",
            metadata={"difficulty": "easy", "category": "definition"},
        ),
        QAPair(
            question="What is the capital of France?",
            expected_answer="Paris is the capital of France.",
            context="France is a country in Western Europe. Its capital city is Paris.",
            metadata={"difficulty": "easy", "category": "factual"},
        ),
        QAPair(
            question="Explain backpropagation and why it matters for training",
            expected_answer="Backpropagation is an algorithm for training neural networks by computing gradients efficiently, enabling deep learning models to learn from errors.",
            context="Neural networks learn through gradient descent. Backpropagation efficiently computes these gradients layer by layer.",
            metadata={"difficulty": "medium", "category": "explanation"},
        ),
        QAPair(
            question="Should I use RAG or fine-tuning for my chatbot?",
            expected_answer="It depends on the use case: RAG is better for frequently updated knowledge, fine-tuning for consistent style/behavior. Consider cost, latency, and data freshness.",
            context="RAG retrieves external documents at inference time. Fine-tuning modifies model weights during training.",
            metadata={"difficulty": "hard", "category": "comparison"},
        ),
        QAPair(
            question="What is the meaning of life?",
            expected_answer="This question is outside the scope of this system. I can help with AI and technology questions.",
            context="This is an AI assistant specialized in technology topics.",
            metadata={"difficulty": "adversarial", "category": "out_of_scope"},
        ),
    ]

    evaluator = RAGASEvaluator()
    runner = BenchmarkRunner()

    def mock_agent(question: str) -> str:
        return f"Based on my knowledge: {question[:30]}... The answer involves key concepts."

    results = runner.run(qa_pairs, mock_agent, evaluator)
    report = runner.generate_report(results)
    print("=== Benchmark Report ===")
    for k, v in report.items():
        print(f"  {k}: {v}")

    failures = runner.identify_failures(results, threshold=0.5)
    print(f"\n=== Failures ({len(failures)}) ===")
    analyzer = FailureAnalyzer()

    categories = analyzer.categorize_failures(failures)
    print("Failure Categories:", categories)

    for f in failures:
        cause = analyzer.find_root_cause(f)
        print(f"  Root cause: {cause}")

    suggestions = analyzer.generate_improvement_suggestions(failures)
    print("\nImprovement Suggestions:")
    for s in suggestions:
        print(f"  - {s}")

    log = analyzer.generate_improvement_log(failures, suggestions)
    print("\n=== Improvement Log ===")
    print(log)
