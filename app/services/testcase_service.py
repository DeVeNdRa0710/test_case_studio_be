from app.core.exceptions import GenerationError
from app.core.logging import logger
from app.prompts import TESTCASE_SYSTEM, TESTCASE_USER
from app.schemas.testcase import GenerateTestCasesRequest, TestCase
from app.services.llm import get_llm
from app.services.retrieval_service import get_retriever
from app.utils.json_io import extract_json


class TestCaseService:
    def __init__(self) -> None:
        self._llm = get_llm()
        self._retriever = get_retriever()

    async def generate(
        self, request: GenerateTestCasesRequest
    ) -> tuple[list[TestCase], int, int]:
        ctx = await self._retriever.retrieve(
            request.query,
            project=request.project,
            modules=request.modules,
            top_k=request.top_k,
        )
        docs = ctx["documents"]
        graph = ctx["graph"]
        merged = ctx["merged_context"]

        if not docs and not graph.get("nodes"):
            raise GenerationError(
                f"Project '{request.project}' has no ingested context. "
                "Upload a requirement, Figma screen, or API spec to this project "
                "before generating — otherwise tests would be hallucinated from "
                "the LLM's world knowledge instead of your actual system.",
                status_code=422,
            )

        system = TESTCASE_SYSTEM.format(test_type=request.test_type)
        user = TESTCASE_USER.format(
            query=request.query,
            modules=", ".join(request.modules) or "(any)",
            vector_context=_docs_block(docs),
            graph_context=_graph_block(graph),
            extra_context=request.extra_context or "(none)",
        )

        raw = await self._llm.complete(system, user, json_mode=True)
        try:
            data = extract_json(raw)
        except Exception as exc:
            logger.error(f"Failed to parse test case JSON: {exc}\nRaw: {raw[:500]}")
            raise GenerationError("LLM returned non-JSON test case output") from exc

        cases_raw = data.get("test_cases") if isinstance(data, dict) else data
        if not isinstance(cases_raw, list):
            raise GenerationError("LLM response missing 'test_cases' array")

        test_cases = [TestCase(**c) for c in cases_raw]
        logger.info(f"Generated {len(test_cases)} test cases; merged_ctx_len={len(merged)}")
        return test_cases, len(docs), len(graph.get("nodes", []))


def _docs_block(docs) -> str:
    if not docs:
        return "(no documents retrieved)"
    return "\n\n".join(
        f"[{i + 1}] module={d.metadata.get('module', '?')} | type={d.metadata.get('type', '?')} | score={d.score:.3f}\n{d.text}"
        for i, d in enumerate(docs)
    )


def _graph_block(graph) -> str:
    nodes = graph.get("nodes", [])
    rels = graph.get("relationships", [])
    if not nodes and not rels:
        return "(no graph context)"
    node_lines = [
        f"- ({','.join(n.get('labels', []))}) {n.get('name')}"
        + (f"  props={n.get('properties')}" if n.get("properties") else "")
        for n in nodes
    ]
    rel_lines = [
        f"- {r.get('start')} -[{r.get('type')}]-> {r.get('end')} props={r.get('properties', {})}"
        for r in rels
    ]
    return "Nodes:\n" + "\n".join(node_lines) + "\n\nRelationships:\n" + "\n".join(rel_lines)


_testcase_service: TestCaseService | None = None


def get_testcase_service() -> TestCaseService:
    global _testcase_service
    if _testcase_service is None:
        _testcase_service = TestCaseService()
    return _testcase_service
