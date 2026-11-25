"""Streamlit interface: a chat view (two models side by side) and a podium prediction view.

Run from the repository root:  .venv/Scripts/python -m streamlit run milestone3/app.py
The sidebar always shows the cached intent and entity slots, so every follow-up is traceable.
All conversation logic lives in pipeline.py; this file only draws it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # so `milestone3` imports work under streamlit

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from neo4j.exceptions import DriverError, Neo4jError

from milestone3 import config
from milestone3.llm.client import complete
from milestone3.llm.prompts import explanation_messages
from milestone3.nlu.embed_classifier import EmbeddingClassifier
from milestone3.nlu.entities import EntityExtractor
from milestone3.pipeline import GRAPH_DOWN, handle_turn, log_prediction
from milestone3.prediction import features
from milestone3.prediction.explain import pre_race_facts
from milestone3.prediction.predictor import PodiumPredictor
from milestone3.retrieval.graph import Graph
from milestone3.session import Session

VIEWS = ["Chat", "Podium prediction"]
MAX_DRAWN_NODES = 60
KEPT_WIDGETS = ["model_a", "model_b", "explain_model", "sel_driver", "sel_race", "sel_year"]
MUTED = "color: #7a7a7a; font-size: 0.8rem;"   # gray for "default value" notes and small captions


# ---------------------------------------------------------------------------------------
# Heavy objects, created once per server process
# ---------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Connecting to the knowledge graph...")
def load_backend():
    """Neo4j connection, the look-up tables and the entity extractor built from them."""
    graph = Graph()
    try:
        tables = graph.lookup_tables()
    except Exception:   # do not leak the connection when the first query fails
        graph.close()
        raise
    return graph, tables, EntityExtractor(tables)


@st.cache_resource(show_spinner="Loading the intent classifier...")
def load_classifier():
    """Sentence-embedding model and the embedded intent examples."""
    return EmbeddingClassifier()


@st.cache_resource(show_spinner="Loading the podium model...")
def load_predictor():
    """The saved podium network and its SHAP explainer."""
    return PodiumPredictor()


@st.cache_data
def cached_prefill(driver_id, race_name, year):
    """`features.prefill`, cached so the CSVs are read once per selection."""
    return features.prefill(driver_id, race_name, year)


def muted(text: str):
    """A small gray note."""
    st.markdown(f'<span style="{MUTED}">{text}</span>', unsafe_allow_html=True)


def show(value) -> str:
    """Entity slot value as text: {'driverId', 'name'} -> name."""
    return value["name"] if isinstance(value, dict) else str(value)


# ---------------------------------------------------------------------------------------
# Sidebar: navigation and the entity cache
# ---------------------------------------------------------------------------------------
def render_sidebar(session: Session):
    """View switch, the cached intent and entity slots, and the reset button."""
    with st.sidebar:
        st.title("F1 Knowledge Assistant")
        st.radio("View", VIEWS, key="view", label_visibility="collapsed")
        st.divider()
        st.subheader("Entity cache")
        st.markdown(f"Intent: `{session.intent or 'none yet'}`")
        if session.entities:
            st.table(pd.DataFrame([{"slot": k, "value": show(v)} for k, v in session.entities.items()]).set_index("slot"))
        else:
            muted("No entities cached yet.")
        if session.pending:
            muted("Waiting for an answer to the clarifying question.")
        if st.button("Reset conversation"):
            session.clear()
            st.rerun()


# ---------------------------------------------------------------------------------------
# Chat view
# ---------------------------------------------------------------------------------------
def subgraph_dot(subgraph: dict) -> str:
    """Graphviz DOT text for the nodes and edges of a retrieved subgraph (first MAX_DRAWN_NODES nodes)."""
    nodes = subgraph["nodes"][:MAX_DRAWN_NODES]
    ids = {n["id"] for n in nodes}
    clean = lambda text: str(text).replace('"', "'")
    lines = ["digraph G {", "rankdir=LR;",
             'node [shape=box, style=filled, fillcolor="#f1efea", color="#8a8a8a", fontname="Helvetica", fontsize=10];',
             'edge [color="#8a8a8a", fontname="Helvetica", fontsize=9];']
    lines += [f'"{n["id"]}" [label="{n["label"]}\\n{clean(n["name"])}"];' for n in nodes]
    lines += [f'"{e["source"]}" -> "{e["target"]}" [label="{e["rel"]}"];'
              for e in subgraph["edges"] if e["source"] in ids and e["target"] in ids]
    return "\n".join(lines + ["}"])


def render_subgraph(subgraph: dict):
    """Drawing, node table, edge table and relationship-type statements of a graphical answer."""
    if len(subgraph["nodes"]) > MAX_DRAWN_NODES:
        muted(f"Drawing limited to the first {MAX_DRAWN_NODES} of {len(subgraph['nodes'])} nodes.")
    st.graphviz_chart(subgraph_dot(subgraph), width="stretch")
    label = {n["id"]: n["label"] for n in subgraph["nodes"]}
    nodes = pd.DataFrame(subgraph["nodes"])[["label", "name"]]
    edges = pd.DataFrame([{"from": label[e["source"]], "relationship": e["rel"], "to": label[e["target"]]}
                          for e in subgraph["edges"]])
    left, right = st.columns(2)
    left.markdown("**Nodes**")
    left.dataframe(nodes, hide_index=True, width="stretch")
    right.markdown("**Edges**")
    right.dataframe(edges, hide_index=True, width="stretch")
    # one statement per relationship type, with how many edges of that type were retrieved
    counts = edges.value_counts().reset_index(name="n")
    st.markdown("**Relationships**")
    st.code("\n".join(f"(:{r['from']})-[:{r['relationship']}]->(:{r['to']})   x{r['n']}"
                      for _, r in counts.iterrows()), language="text")


def render_trace(turn):
    """Retrieval trace under an in-domain answer: intent, entities, graph, Cypher, parameters, rows."""
    r = turn.retrieval
    with st.expander("Retrieval trace"):
        st.markdown(f"**Intent** `{turn.intent}` (decided by: {turn.how}), template `{r.template}`")
        st.markdown("**Entities** " + (", ".join(f"`{k}` = {show(v)}" for k, v in turn.entities.items()) or "none"))
        if r.subgraph:
            render_subgraph(r.subgraph)
        elif len(r.rows) == 1:   # a scalar result: only the value(s)
            st.markdown("**Value** " + ", ".join(f"{k}: {v}" for k, v in r.rows[0].items()))
        st.markdown("**Cypher**")
        st.code(r.cypher.strip(), language="cypher")
        st.markdown("**Parameters**")
        st.json(r.params, expanded=False)
        st.markdown("**Rows**")
        st.dataframe(pd.DataFrame(r.rows), hide_index=True, width="stretch")


def render_turn(turn):
    """One exchange: the user's message, then the assistant's columns or message, then the trace."""
    with st.chat_message("user"):
        st.write(turn.text)
    with st.chat_message("assistant"):
        if turn.answers:
            for column, answer in zip(st.columns(len(turn.answers)), turn.answers):
                with column:
                    st.markdown(f"**{answer.model}**")
                    st.write(answer.text)
                    if answer.note:
                        muted(answer.note)
                    muted(f"{answer.result.latency_s:.1f} s, {answer.result.input_tokens} in / "
                          f"{answer.result.output_tokens} out tokens")
        else:
            st.write(turn.message)
        if turn.retrieval:
            render_trace(turn)


def render_chat(session: Session, graph, extractor):
    """Model pickers, the conversation so far and the input box (a prediction question switches view)."""
    st.header("Ask about Formula 1")
    left, right = st.columns(2)
    models = [left.selectbox("Model A", config.LLM_MODELS, index=0, key="model_a"),
              right.selectbox("Model B", config.LLM_MODELS, index=1, key="model_b")]
    for turn in session.history:
        render_turn(turn)
    if text := st.chat_input("Ask a question, e.g. How many wins did Lewis Hamilton have in 2019?"):
        with st.spinner("Working on it..."):
            turn = handle_turn(text, session, graph, extractor, load_classifier(), models)
        if turn.kind == "predict":   # hand over to the prediction view with the details found
            st.session_state.arrive = turn.prefill
            st.session_state.goto = "Podium prediction"
        st.rerun()


# ---------------------------------------------------------------------------------------
# Prediction view
# ---------------------------------------------------------------------------------------
def apply_selection(stated: dict | None = None):
    """Fill every feature field from the chosen driver / race / year (plus values stated in chat)."""
    s = st.session_state
    values, filled, info = {f.name: f.default for f in features.FEATURES}, set(), None
    if s.sel_driver and s.sel_race:
        values, filled, info = cached_prefill(s.sel_driver, s.sel_race, s.sel_year)
    values = {**values, **(stated or {})}
    s.start_values, s.filled, s.race_info, s.result = values, filled | set(stated or {}), info, None
    s.version = s.get("version", 0) + 1   # new widget keys, so the inputs restart from `values`


def take_arrival():
    """Consume a hand-over from the chat view (it must run before the widgets are created)."""
    s = st.session_state
    if arrival := s.pop("arrive", None):
        s.sel_driver = arrival.get("driver", {}).get("driverId")
        s.sel_race, s.sel_year = arrival.get("race_name"), arrival.get("year")
        apply_selection(arrival["values"])
    elif "filled" not in s:
        s.sel_driver = s.sel_race = s.sel_year = None
        apply_selection()


def fallback_explanation(prediction: dict) -> str:
    """Deterministic explanation from the strongest SHAP contributions (used when the LLM call fails)."""
    describe = lambda cs: ", ".join(f"{f} = {v:g} ({s:+.2f})" for f, v, s in cs[:3]) or "none"
    up = [c for c in prediction["contributions"] if c[2] > 0]
    down = [c for c in prediction["contributions"] if c[2] < 0]
    verdict = "a podium" if prediction["predicted_podium"] else "no podium"
    return (f"The model gives a {prediction['probability']:.0%} podium probability and predicts {verdict}. "
            f"Strongest pushes towards a podium (log-odds): {describe(up)}. "
            f"Strongest pushes away: {describe(down)}.")


def run_prediction(values, model, graph, predictor):
    """Validate, predict, build the SHAP figure and ask the LLM to explain; stores the result."""
    s = st.session_state
    errors = features.validate(values)
    if errors:
        s.result = {"errors": errors}
        return
    prediction = predictor.predict(values)
    info, facts = s.race_info, []
    if info:   # facts from the graph, strictly before the race date
        try:   # the explanation still works without graph facts if Neo4j drops out
            facts = pre_race_facts(graph, s.sel_driver, info["constructorId"], info["circuitId"], str(info["date"]))
        except (DriverError, Neo4jError, OSError):
            pass
    messages = explanation_messages(prediction, facts, config.EXPLAIN_VARIANT)
    llm = complete(model, messages)
    log_prediction(values, prediction, facts, messages, llm)
    s.result = {"prediction": prediction, "figure": predictor.waterfall_figure(prediction), "llm": llm, "info": info}


def render_result(result):
    """Probability, class, actual finish, then the SHAP waterfall beside the explanation."""
    if "errors" in result:
        for error in result["errors"]:
            st.error(error)
        return
    p, info = result["prediction"], result["info"]
    a, b, c = st.columns(3)
    a.metric("Podium probability", f"{p['probability']:.1%}")
    b.metric("Predicted class", "Podium" if p["predicted_podium"] else "No podium")
    b.caption(f"threshold {p['threshold']:.2f}")
    if info and info.get("actual_position"):
        c.metric("Actual finish", f"P{info['actual_position']}")
        c.caption("podium" if info["actual_podium"] else "no podium")
    left, right = st.columns([1.2, 1])   # the waterfall needs the extra width to stay legible
    with left:
        st.markdown("**Local SHAP contributions**")
        st.pyplot(result["figure"])
        plt.close(result["figure"])
    with right:
        st.markdown("**Explanation**")
        if result["llm"].error:
            muted(result["llm"].error)
            st.write(fallback_explanation(p))
        else:
            st.write(result["llm"].text)
            muted(f"{result['llm'].model}, {result['llm'].latency_s:.1f} s")


def render_prediction(graph, tables):
    """Selection of driver / race / season, the feature inputs, the Predict button and the result."""
    st.header("Podium prediction")
    take_arrival()
    try:
        predictor = load_predictor()
    except (FileNotFoundError, OSError):
        predictor = None
        st.error("The saved podium model is missing: export it from the notebook into milestone3/models.")

    # driver / race / year selection prefills the fields below from the Milestone 1 table
    drivers = {d["driverId"]: f'{d["forename"]} {d["surname"]}' for d in sorted(tables["drivers"], key=lambda d: d["surname"])}
    a, b, c = st.columns([2, 2, 1])
    a.selectbox("Driver", [None, *drivers], format_func=lambda i: "not selected" if i is None else drivers[i],
                key="sel_driver", on_change=apply_selection)
    b.selectbox("Race", [None, *tables["races"]], format_func=lambda r: "not selected" if r is None else r,
                key="sel_race", on_change=apply_selection)
    c.selectbox("Season", [None, *range(2024, 1949, -1)], format_func=lambda y: "any" if y is None else str(y),
                key="sel_year", on_change=apply_selection)
    s = st.session_state
    if s.sel_driver and s.sel_race and s.race_info is None:
        muted("No pre-race record for this driver and race in the model table; default values are shown.")
    elif s.race_info:
        muted(f"Values taken from the pre-race record of the {s.race_info['race_name']} {s.race_info['year']}.")

    # one input per feature; a value neither supplied by the data nor changed by the user gets a
    # gray "default value" note (Streamlit cannot style a single input, so the note carries the cue)
    st.markdown("**Pre-race features**")
    values = {}
    for column, spec in zip(st.columns(3) * 3, features.FEATURES):
        with column:
            help_text = f"{spec.help} Unit: {spec.unit}. Valid range: {spec.min:g} to {spec.max:g}."
            key, start = f"f_{spec.name}_{s.version}", s.start_values[spec.name]
            if spec.kind == "bool":
                values[spec.name] = int(st.checkbox(spec.label, value=bool(start), key=key, help=help_text))
            elif spec.kind == "int":
                values[spec.name] = st.number_input(spec.label, value=int(start), step=1, key=key, help=help_text)
            else:
                values[spec.name] = st.number_input(spec.label, value=float(start), step=0.01, format="%.2f",
                                                    key=key, help=help_text)
            if spec.name not in s.filled and values[spec.name] == spec.default:
                muted("default value")

    d, e = st.columns([2, 1])
    default = config.LLM_MODELS.index("openai/gpt-oss-20b") if "openai/gpt-oss-20b" in config.LLM_MODELS else 0
    model = d.selectbox("Explanation model", config.LLM_MODELS, index=default, key="explain_model")
    e.write("")
    if e.button("Predict", type="primary", disabled=predictor is None):
        with st.spinner("Predicting..."):
            run_prediction(values, model, graph, predictor)
    if s.result:
        render_result(s.result)


# ---------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="F1 Knowledge Assistant", layout="wide")
    st.markdown("<style>.block-container {padding-top: 2rem;}</style>", unsafe_allow_html=True)
    s = st.session_state
    if "session" not in s:
        s.session = Session()
    if "goto" in s:   # a chat turn asked for another view; must be applied before the radio exists
        s.view = s.pop("goto")
    try:
        graph, tables, extractor = load_backend()
    except (DriverError, Neo4jError, OSError):
        st.error(GRAPH_DOWN)
        st.stop()
    # Streamlit forgets widgets that are not drawn in a run, so the selections are copied aside
    for key, value in s.get("kept", {}).items():
        s.setdefault(key, value)
    render_sidebar(s.session)
    if s.get("view", VIEWS[0]) == "Chat":
        render_chat(s.session, graph, extractor)
    else:
        render_prediction(graph, tables)
    s.kept = {key: s[key] for key in KEPT_WIDGETS if key in s}


main()
