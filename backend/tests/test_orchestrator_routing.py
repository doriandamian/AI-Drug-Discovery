import json
import socket
from urllib.parse import urlparse

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agents import specialists, trace
from agents.specialists import SpecialistError
from agents import orchestrator


def _tool_call(name: str, task: str, call_id: str = "call_1") -> AIMessage:
    """An assistant message that calls one specialist tool."""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": {"task": task}, "id": call_id, "type": "tool_call"}],
    )

_ROUTES = [
    (orchestrator.cheminformatics_agent, specialists._cheminformatics, "_run_cheminformatics_agent"),
    (orchestrator.safety_agent, specialists._safety, "_run_safety_agent"),
    (orchestrator.literature_agent, specialists._literature, "_run_agent"),
    (orchestrator.graph_agent, specialists._graph, "_run_graph_agent"),
    (orchestrator.molecular_design_agent, specialists._molecular_design, "_run_design_agent"),
]


@pytest.mark.parametrize("wrapper, expected_agent, runner", _ROUTES)
def test_specialist_tool_dispatches_to_its_own_agent(wrapper, expected_agent, runner, monkeypatch):
    captured = {}

    def fake_run(agent, task):
        captured["agent"] = agent
        captured["task"] = task
        return "ok"

    monkeypatch.setattr(specialists, runner, fake_run)
    out = wrapper.invoke({"task": "analyze aspirin"})

    assert out == "ok"
    assert captured["agent"] is expected_agent, "tool routed to the wrong specialist"
    assert captured["task"] == "analyze aspirin"


def test_all_five_specialists_are_bound_to_the_supervisor():
    names = {t.name for t in orchestrator.tools}
    assert names == {
        "cheminformatics_agent", "safety_agent", "literature_agent",
        "graph_agent", "molecular_design_agent",
    }


def test_tool_node_resolves_each_name_to_its_specialist_tool():
    by_name = orchestrator.tool_node.tools_by_name
    assert {t.name for t in orchestrator.tools} == set(by_name)
    for t in orchestrator.tools:
        assert by_name[t.name] is t


def test_tool_node_dispatches_by_name(monkeypatch):
    captured = {}

    def fake_run(agent, task):
        captured["agent"] = agent
        return f"ran for: {task}"

    monkeypatch.setattr(specialists, "_run_safety_agent", fake_run)

    tool = orchestrator.tool_node.tools_by_name["safety_agent"]
    out = tool.invoke({"task": "is caffeine toxic?"})

    assert captured["agent"] is specialists._safety
    assert "is caffeine toxic?" in out


def test_design_runner_renders_from_json_not_llm_text(monkeypatch):
    payload = json.dumps({
        "status": "ok", "compound": "Aspirin", "safety_optimized": True, "n_scored": 3,
        "seed": {"smiles": "CC(=O)Oc1ccccc1C(=O)O", "fitness": 0.82, "qed": 0.82,
                 "tox": 0.06, "sa": 2.19, "alerts": 0, "ad_similarity": 1.0,
                 "similarity_to_seed": 1.0},
        "candidates": [{"rank": 1, "smiles": "CCO", "fitness": 0.89, "qed": 0.89,
                        "tox": 0.09, "sa": 1.51, "alerts": 0, "ad_similarity": 0.82,
                        "similarity_to_seed": 0.55,
                        "vs_seed": {"tox": {"delta": 0.03, "direction": "worsened"},
                                    "fitness": {"delta": 0.07, "direction": "improved"}}}],
        "generations_best": [0.85, 0.89], "caveats": ["unvalidated"],
    })

    class FakeDesignAgent:
        name = "molecular_design"

        def stream(self, *a, **k):
            yield {"tools": {"messages": [
                ToolMessage(content=payload, name="design_analogs", tool_call_id="1")]}}
            yield {"agent": {"messages": [
                AIMessage(content="Candidate CCO: tox=1.51 SA=0.09, much safer!")]}}

    trace.reset()
    out = specialists._run_design_agent(FakeDesignAgent(), "make aspirin safer")

    assert "tox=0.09" in out and "tox worsened (+0.03)" in out
    assert "tox=1.51" not in out and "much safer" not in out
    assert "<smiles>CCO</smiles>" in out


def _fake_agent(name, messages):
    class _Fake:
        def stream(self, *a, **k):
            for m in messages:
                node = "tools" if isinstance(m, ToolMessage) else "agent"
                yield {node: {"messages": [m]}}
    _Fake.name = name
    return _Fake()


def test_safety_runner_renders_both_profiles_from_json(monkeypatch):
    def tox(compound, clin):
        return ToolMessage(name="predict_toxicity", tool_call_id=compound, content=json.dumps({
            "status": "ok", "model": "multitask", "compound": compound, "n_endpoints": 2,
            "endpoints": [{"id": "ClinTox", "description": "clinical", "probability": clin,
                           "cutoff": 0.12, "flagged": clin >= 0.12}],
            "flagged": ([{"id": "ClinTox", "description": "clinical", "probability": clin,
                          "cutoff": 0.12, "flagged": True}] if clin >= 0.12 else []),
            "highest_probability": {"id": "ClinTox", "description": "clinical", "probability": clin},
            "domain": None,
        }))

    agent = _fake_agent("safety", [
        tox("Aspirin", 0.67), tox("Caffeine", 0.05),
        AIMessage(content="Caffeine is totally safe at 0%, much worse than aspirin."),
    ])
    trace.reset()
    out = specialists._run_safety_agent(agent, "compare safety of aspirin and caffeine")

    assert "Toxicity profile for 'Aspirin'" in out and "Toxicity profile for 'Caffeine'" in out
    assert "67%" in out and "Model scope:" in out
    assert "totally safe" not in out


def test_cheminformatics_runner_prefers_calculated_properties(monkeypatch):
    agent = _fake_agent("cheminformatics", [
        ToolMessage(name="fetch_pubchem_properties", tool_call_id="1", content=json.dumps(
            {"status": "ok", "compound": "Aspirin", "molecular_weight": 180.16, "logp": 1.31,
             "smiles_stored": True})),
        ToolMessage(name="calculate_properties", tool_call_id="2", content=json.dumps(
            {"status": "ok", "compound": "Aspirin", "molecular_weight": 180.16, "logp": 1.31,
             "hbd": 1, "hba": 3, "tpsa": 63.6, "rotatable_bonds": 2, "rings": 1, "qed": 0.55,
             "lipinski_violations": 0, "lipinski_pass": True})),
        AIMessage(content="aspirin has 4 lipinski violations"),  # wrong on purpose
    ])
    trace.reset()
    out = specialists._run_cheminformatics_agent(agent, "does aspirin pass Lipinski?")

    assert "Lipinski Rule of Five: PASS" in out and "QED Score" in out
    assert "4 lipinski violations" not in out


def test_tools_condition_routes_tool_calls_to_tools():
    from langgraph.prebuilt import tools_condition
    assert tools_condition({"messages": [_tool_call("graph_agent", "x")]}) == "tools"


def test_tools_condition_ends_on_a_plain_answer():
    from langgraph.prebuilt import tools_condition
    assert tools_condition({"messages": [AIMessage(content="Aspirin's MW is 180.")]}) != "tools"


def test_run_agent_raises_specialist_error_on_crash(monkeypatch):
    class Boom:
        name = "safety"

        def stream(self, *a, **k):
            raise ValueError("ollama exploded")

    trace.reset()
    with pytest.raises(SpecialistError) as ei:
        specialists._run_agent(Boom(), "task")
    assert "safety" in str(ei.value)
    assert "ValueError" in str(ei.value)


def test_run_agent_records_trace_even_when_it_crashes(monkeypatch):
    class Boom:
        name = "graph"

        def stream(self, *a, **k):
            raise RuntimeError("down")

    trace.reset()
    with pytest.raises(SpecialistError):
        specialists._run_agent(Boom(), "task")
    assert trace.get() == []


def test_tool_node_is_wired_to_surface_specialist_errors():
    assert orchestrator.tool_node._handle_tool_errors is orchestrator._surface_specialist_error


def test_specialist_error_formats_as_an_explicit_signal():
    msg = orchestrator._surface_specialist_error(
        SpecialistError("specialist 'safety' failed: ValueError: boom")
    )
    assert "SPECIALIST_ERROR" in msg
    assert "do NOT fabricate" in msg
    assert "boom" in msg


def test_supervisor_prompt_forbids_papering_over_a_specialist_error():
    prompt = orchestrator.SUPERVISOR_SYSTEM_PROMPT.content
    assert "SPECIALIST_ERROR" in prompt


def _ollama_reachable() -> bool:
    parsed = urlparse(orchestrator.OLLAMA_BASE_URL)
    host, port = parsed.hostname, parsed.port or 11434
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


requires_ollama = pytest.mark.skipif(
    not _ollama_reachable(),
    reason=f"Ollama not reachable at {orchestrator.OLLAMA_BASE_URL}; live routing test skipped",
)


@requires_ollama
@pytest.mark.parametrize("query, expected_tool", [
    ("What is the molecular weight of aspirin?", "cheminformatics_agent"),
    ("Is bisphenol A toxic?", "safety_agent"),
    ("What does the literature say about metformin's mechanism?", "literature_agent"),
    ("Which drugs target cyclooxygenase?", "graph_agent"),
    ("Design a safer, more drug-like analog of ibuprofen.", "molecular_design_agent"),
])
def test_live_supervisor_routes_query_to_expected_specialist(query, expected_tool):
    messages = [orchestrator.SUPERVISOR_SYSTEM_PROMPT, HumanMessage(content=query)]
    response = orchestrator.llm_supervisor_with_tools.invoke(messages)
    called = [tc["name"] for tc in (response.tool_calls or [])]
    assert expected_tool in called, f"{query!r} routed to {called}, expected {expected_tool}"
