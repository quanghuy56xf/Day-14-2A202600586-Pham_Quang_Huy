# Day 14 — Reflection
## Evaluation Report & Failure Analysis

**Domain:** AI / RAG Assistant  
**Benchmark date:** Day 14 Lab  
**Golden dataset:** 20 QA pairs (5E + 7M + 5H + 3A)

---

## 1. Benchmark Results Summary

**Overall pass rate:** 0%

**Average scores:**

| Metric | Average | Min | Max | Std Dev |
|--------|---------|-----|-----|---------|
| Faithfulness | 0.079 | 0.000 | 0.214 | 0.065 |
| Relevance | 0.842 | 0.500 | 1.000 | 0.175 |
| Completeness | 0.155 | 0.000 | 0.667 | 0.146 |
| Overall Score | 0.359 | 0.243 | 0.603 | 0.092 |

**Score interpretation (theo bài giảng)** — 60 metric observations (20 cases × 3 metrics):
- Metrics ở Good (0.8–1.0): **13**
- Metrics ở Needs Work (0.6–0.8): **5**
- Metrics ở Significant Issues (<0.6): **42**

**Insight:** Agent có relevance cao (echo keywords từ câu hỏi) nhưng faithfulness và completeness rất thấp — classic pattern của agent không dùng retrieved context.

**Failure type distribution:**

| Failure Type | Count | Percentage |
|--------------|-------|------------|
| hallucination | 20 | 100% |
| irrelevant | 0 | 0% |
| incomplete | 0 | 0% |
| off_topic | 0 | 0% |
| refusal | 0 | 0% |

> Tất cả failures bị classify `hallucination` vì faithfulness < 0.3 (first-match rule trong `run_full_eval`), dù completeness cũng rất thấp.

---

## 2. Top 3 Worst Failures — 5 Whys Analysis

### Failure 1 — H02

**Question:** My RAG system has high recall but low precision — what should I fix first?

**Agent Answer:** Based on my knowledge about My RAG system has high recall but low pr: the answer involves key AI concepts and related terminology.

**Scores:** Faithfulness: 0.158 | Relevance: 0.571 | Completeness: 0.000 | Overall: 0.243

**5 Whys Analysis:**

| Level | Question | Answer |
|-------|----------|--------|
| Symptom | Vấn đề là gì? | Agent trả lời generic, completeness = 0, không đề xuất reranking/filter |
| Why 1 | Tại sao xảy ra? | Agent không inject retrieved context vào câu trả lời |
| Why 2 | Tại sao Why 1 xảy ra? | `mock_agent` chỉ echo câu hỏi, không gọi retriever |
| Why 3 | Tại sao Why 2 xảy ra? | Pipeline thiếu bước "answer from context only" trong prompt |
| Why 4 | Root cause là gì? | Generation không bị ràng buộc bởi context — không có faithfulness guardrail |

**Root cause (from `find_root_cause()`):**
> Context is missing or irrelevant — improve retrieval

**Bạn có đồng ý với root cause suggestion không? Tại sao?**  
Đồng ý một phần. `find_root_cause()` chỉ ra faithfulness thấp nhất — đúng vì agent bỏ qua context. Root cause sâu hơn là **generation prompt không enforce grounding**, không chỉ retrieval.

**Proposed fix:**
1. Thêm system prompt: *"Answer ONLY using the provided context. If context is insufficient, say so."*
2. Thêm reranking step trước generation khi precision thấp (đúng với câu hỏi H02).

---

### Failure 2 — M02

**Question:** Explain the difference between embedding search and keyword search.

**Agent Answer:** Based on my knowledge about Explain the difference between embedding: the answer involves key AI concepts and related terminology.

**Scores:** Faithfulness: 0.000 | Relevance: 0.667 | Completeness: 0.067 | Overall: 0.244

**5 Whys Analysis:**

| Level | Question | Answer |
|-------|----------|--------|
| Symptom | Agent không giải thích BM25 vs embedding | Generic template response |
| Why 1 | Không dùng context có sẵn về search methods | faithfulness = 0 |
| Why 2 | Agent function stateless, không nhận context | Benchmark `agent_fn` chỉ nhận question |
| Why 3 | Kiến trúc tách retriever và generator chưa nối | RAG pipeline chưa hoàn chỉnh |
| Why 4 | Root cause: thiếu end-to-end RAG wiring | |

**Root cause:**
> Context is missing or irrelevant — improve retrieval *(faithfulness = 0)*

**Proposed fix:**
1. Đổi `agent_fn` thành `agent_fn(question, context)` và truyền retrieved context.
2. Thêm few-shot examples so sánh embedding vs keyword trong prompt.

---

### Failure 3 — M01

**Question:** How does RAG reduce hallucination compared to a plain LLM?

**Agent Answer:** Based on my knowledge about How does RAG reduce hallucination compar: the answer involves key AI concepts and related terminology.

**Scores:** Faithfulness: 0.059 | Relevance: 0.625 | Completeness: 0.077 | Overall: 0.254

**5 Whys Analysis:**

| Level | Question | Answer |
|-------|----------|--------|
| Symptom | Không giải thích cơ chế grounding của RAG | Completeness 7.7% |
| Why 1 | Answer không chứa từ khóa từ expected ("retrieved documents", "grounded") | |
| Why 2 | Template response cố định | |
| Why 3 | Không có instruction "cover all key points from reference" | |
| Why 4 | Root cause: thiếu completeness-aware generation | |

**Root cause:**
> Context is missing or irrelevant — improve retrieval

**Proposed fix:**
1. Chain-of-thought prompt: liệt kê claims từ context trước khi trả lời.
2. Post-generation check: nếu completeness < 0.5 → regenerate với "include missing points".

---

## 3. Failure Clustering

| Cluster | Root Cause | Failures in cluster | Priority |
|---------|-----------|--------------------:|----------|
| 1 | Agent ignores retrieved context (no grounding) | 18 | **High** |
| 2 | Generic template → low completeness | 20 | **High** |
| 3 | Adversarial handling missing (A01–A03) | 3 | Medium |

**Nếu chỉ fix 1 cluster, bạn chọn cluster nào? Tại sao?**  
Chọn **Cluster 1** — fix grounding (context → prompt → answer) sẽ cải thiện faithfulness cho 100% failures và gián tiếp tăng completeness. Một root cause, nhiều failures — đúng tinh thần failure clustering trong bài giảng.

---

## 4. Improvement Log (from `generate_improvement_log`)

```
| Failure ID | Type | Root Cause | Suggested Fix | Status |
|------------|------|------------|---------------|--------|
| F001 | hallucination | Context is missing or irrelevant — improve retrieval | Implement hallucination checker to filter unsupported claims | Open |
| F002 | hallucination | Context is missing or irrelevant — improve retrieval | Increase chunk size in RAG pipeline to reduce context fragmentation | Open |
| F003 | hallucination | Context is missing or irrelevant — improve retrieval | Add few-shot examples showing complete answers to improve completeness | Open |
| F004 | hallucination | Context is missing or irrelevant — improve retrieval | Review and address root cause | Open |
| F005 | hallucination | Context is missing or irrelevant — improve retrieval | Review and address root cause | Open |
```

*(Full log: 20 rows — tất cả failures cùng cluster grounding; chạy `python template.py` để regenerate.)*

**3 improvement suggestions từ `generate_improvement_suggestions()`:**
1. Implement hallucination checker to filter unsupported claims
2. Increase chunk size in RAG pipeline to reduce context fragmentation
3. Add few-shot examples showing complete answers to improve completeness

---

## 5. Regression Testing Strategy

### CI/CD Integration

**Câu 1: Khi nào chạy `run_regression()`?**
- Trước mỗi merge vào `main` (GitHub Actions on PR)
- Sau mỗi thay đổi system prompt, retriever config, hoặc chunk size
- Trước release tag / demo

**Câu 2: Threshold 0.05 có phù hợp không?**  
Phù hợp cho **faithfulness** và **completeness** trên domain factual RAG. Có thể **strict hơn (0.03)** cho faithfulness vì hallucination impact cao. **Loose hơn (0.08)** cho relevance nếu paraphrase nhiều.

**Câu 3: Regression → block hay alert?**
- **Block deploy** nếu faithfulness regression > 0.05 (quality gate cứng)
- **Alert only** nếu chỉ relevance giảm nhẹ (có thể do paraphrase)
- Trade-off: block = an toàn hơn nhưng chậm release; alert = nhanh nhưng rủi ro production

**Câu 4: Eval pipeline trong CI/CD flow**

```
Code change → [Unit tests] → [Offline golden-set eval + run_regression()] → [Deploy to staging + smoke eval] → Deploy
```

---

## 6. Continuous Improvement Loop

| Priority | Action | Metric sẽ improve | Expected impact |
|----------|--------|-------------------|-----------------|
| 1 | Wire context vào agent prompt + grounding instruction | Faithfulness | +0.4–0.6 avg |
| 2 | Add reranking (Exercise 3.5 pipeline) | Context Precision | +0.4 AP |
| 3 | Expand golden set với adversarial cases từ production logs | Pass rate on A* | Catch injection early |

**Failure cases thêm cho sprint tiếp theo:**
1. Multi-hop question cần 2+ chunks (test retrieval recall)
2. Câu hỏi tiếng Việt / mixed language (test tokenizer edge case)
3. Policy update scenario — context mới vs cũ conflict

---

## 7. Framework Reflection

**Framework đã dùng trong lab:** RAGAS-inspired heuristic (word overlap)

**Production choice:** **DeepEval** cho CI/CD (pytest-native) + **RAGAS** cho periodic deep audit

| Tiêu chí | Lý do chọn |
|----------|------------|
| Focus phù hợp vì... | DeepEval gắn trực tiếp pytest assertions; RAGAS chuẩn hóa RAG metrics |
| CI/CD integration vì... | `deepeval test run` trong GitHub Actions; RAGAS chạy weekly full report |
| Team workflow vì... | Dev dùng DeepEval mỗi PR; ML engineer dùng RAGAS để tune retriever |
