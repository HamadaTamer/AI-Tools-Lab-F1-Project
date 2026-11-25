# Milestone 3 study guide: Graph-RAG Formula 1 assistant

For the oral exam. Every member must explain the whole flow (section 3) and defend their own part (section 4).
References are written `file.py:function`. All paths are relative to `milestone3/` unless stated.
Evaluation numbers are not copied here (except the finished intent evaluation): open the file named
"see eval/results/..." and read the numbers there right before the exam.

---

## 1. What the system does and how to run it

1. The user types a Formula 1 question. The system classifies the intent (one of 12), extracts the entities
   (driver, team, circuit, race, years) and fills a fixed, parameterised Cypher template.
2. The template runs on the Milestone 2 Neo4j graph. Only the returned rows (plus a one-sentence scope) go to the LLM,
   which turns them into 1-3 sentences. The LLM never writes Cypher and never sees the graph.
3. Two Groq models answer the same prompt in parallel and are shown side by side. If a call fails the column shows
   "API connection error" plus a deterministic reading of the rows.
4. A session cache lets follow-ups work ("And at Monza?"). Out-of-scope questions are refused with no retrieval.
5. A prediction question ("Predict Norris at the 2024 Miami Grand Prix") opens a second view. The saved Milestone 1
   shallow neural network predicts the podium probability, live SHAP explains it, and an LLM explains the SHAP numbers
   and pre-race graph facts in plain language, without calling anything a cause.

### Prerequisites

| Need | Where |
| --- | --- |
| Neo4j running with the M2 graph loaded | built by `milestone2/Create_kg.py` from `milestone2/*.csv` |
| Neo4j credentials | `milestone2/config.txt` (`URI`, `USERNAME`, `PASSWORD`, optional `DATABASE`), read by `config.py:neo4j_settings` |
| Groq key | `.env` at the repo root: `GROQ_API_KEY=...` (git-ignored), loaded by `config.py` with `python-dotenv` |
| Saved podium model | `milestone3/models/` (written by notebook section 18) |
| M1 model table | `milestone3/models/model_table.csv` (copy of `outputs/model_table.csv`, saved by the notebook export cell) |

### Commands (run from the repo root)

```
.venv/Scripts/python -m streamlit run milestone3/app.py        # the application

.venv/Scripts/python -m milestone3.tests.check_pipeline        # conversation logic, fake graph + fake LLM, no network
.venv/Scripts/python -m milestone3.eval.eval_intents           # intent comparison -> eval/results/intent_*
.venv/Scripts/python -m milestone3.eval.eval_llm               # model x {zero,few}-shot -> eval/results/llm_*
.venv/Scripts/python -m milestone3.eval.eval_llm --variant strict   # the improved prompt (before/after)
.venv/Scripts/python -m milestone3.eval.qualitative_sheet      # 40-item rating sheet
.venv/Scripts/python -m milestone3.retrieval.graph             # one example per intent against Neo4j
.venv/Scripts/python -m milestone3.prediction.predictor        # predicts every explanation case, saves a waterfall
.venv/Scripts/python -m milestone3.prediction.features         # rewrites eval/results/feature_ranges.csv
.venv/Scripts/python -m milestone3.nlu.entities                # entity extraction demo
```

`eval_llm` options: `--variant base|strict`, `--models M ...`, `--limit N`, `--skip-explain`. It resumes
(`eval_llm.py:run_all` skips calls already stored in `eval/results/llm_runs.jsonl` and retries failed ones).

---

## 2. Repo map

```
milestone3/
  CONTRACTS.md          shared interfaces, schema, intent table, test-set formats (the plan all four parts were written against)
  STUDY.md              this file
  config.py             paths, Groq key, model ids, embedding model and its two thresholds, API_ERROR text, Neo4j settings reader
  pipeline.py           handle_turn: the whole life of one chat message; Turn/Answer dataclasses; logging
  session.py            Session: cached intent, entity cache, pending clarification, history; merge rules
  app.py                Streamlit UI: chat view, prediction view, sidebar entity cache, retrieval trace, subgraph drawing
  nlu/
    intents.py          the 12 intents: description, example phrasings, required/optional slots
    rule_classifier.py  approach 1: ordered regex rules, first match wins
    embed_classifier.py approach 2: sentence-embedding nearest-example classifier with min score, margin, out_of_scope guard
    entities.py         EntityExtractor: names and years -> slot dict; ambiguity detection
  retrieval/
    templates.py        the Cypher templates, REQUIRED slots, param builder, template selection
    graph.py            Neo4j access: lookup_tables, retrieve, run, subgraph building, missing_slots
  llm/
    client.py           complete(): one Groq call, never raises, returns latency and tokens
    prompts.py          persona / context / task prompts, few-shot examples, "strict" variant, explanation prompt
    fallback.py         read_rows(): deterministic reading of rows when the call fails; NO_DATA text
  prediction/
    features.py         the 9 feature specs, valid ranges, validate(), prefill() from the M1 model table
    predictor.py        PodiumPredictor: loads the saved NN, predicts, live SHAP, waterfall figure
    explain.py          pre_race_facts(): two Cypher queries, strictly before the race date
  models/               nn.pt, preprocessor.joblib, shap_background.npy, global_shap.csv, meta.json  (exported from the notebook)
  eval/
    intent_testset.csv  labelled questions for the intent comparison (94 rows at the moment)
    qa_testset.json     question-answering items (normal, follow_up, ambiguous, empty, out_of_scope)
    explanation_cases.json  driver/race pairs for explanation evaluation
    eval_intents.py     rule vs embedding vs hybrids, 3 embedding models, threshold sweep, confusion matrices
    eval_llm.py         model comparison, zero vs few-shot, metrics, before/after prompt variant
    qualitative_sheet.py  prints/writes the 1-5 fluency rating sheet
    results/            everything the scripts write (intent_*, llm_runs.jsonl, llm_comparison.csv, llm_summary.md, feature_ranges.csv, waterfall_example.png)
  tests/check_pipeline.py  asserts the conversation logic without Neo4j or Groq
  logs/turns.jsonl      one JSON line per chat turn (git-ignored)
  .streamlit/config.toml  light theme
```

### What is reused from Milestones 1 and 2

| Reused item | Used by | Why |
| --- | --- | --- |
| `milestone2/Create_kg.py` schema (Driver, Constructor, Circuit, Season, Status, Race, Result, Qualifying, PitStop; `AT_CIRCUIT`, `IN_SEASON`, `FOR_RACE`, `BY_DRIVER`, `FOR_CONSTRUCTOR`, `HAS_STATUS`) | every template | the spec forbids renaming or removing anything; M3 adds no label, relationship or property |
| `milestone2/config.txt` | `config.py:neo4j_settings`, `Graph.__init__` | same credentials file as M2 |
| `milestone2/races.csv`, `results.csv` | `prediction/features.py:prefill` | race lookup and the actual finishing position (the same data the graph is built from) |
| `milestone2/*.csv` | `nlu/entities.py` `__main__` demo only | builds look-up tables without Neo4j |
| `milestone3/models/model_table.csv` | `features.py:prefill`, `write_feature_ranges` | the M1 leakage-safe driver-race rows: pre-race feature values and standings, which the graph lacks |
| `F1_podium_prediction.ipynb` section 18 (last cell) | writes `models/*` | exports the fitted preprocessor, network, SHAP background, global SHAP and metadata. The app never retrains |

---

## 3. The life of a question

### 3.1 Main example and its follow-up

Start-up (once per server process, `app.py`): `load_backend` opens the `Graph`, calls `Graph.lookup_tables`
(four Cypher queries: drivers with first/last season, constructors, circuits, distinct race names) and builds the
`EntityExtractor`. `load_classifier` builds the `EmbeddingClassifier` (embeds all example phrasings once).
`main()` creates a `Session` in `st.session_state`.

**Turn 1: "How many wins did Lewis Hamilton have at Silverstone before 2020?"**

| # | Step | What happens |
| --- | --- | --- |
| 1 | `app.py:render_chat` | `st.chat_input` returns the text; calls `pipeline.handle_turn(text, session, graph, extractor, classifier, models)` with the two selected model ids |
| 2 | `pipeline.handle_turn` | creates `Turn(text=...)`, calls `run_turn` inside `try`; only `DriverError`, `Neo4jError`, `OSError` become the "Neo4j is not reachable" turn |
| 3 | `run_turn` step 1 | `session.pending` is empty, so it is a fresh message: `extractor.extract(text)` |
| 4 | `EntityExtractor.extract` | `fold` strips accents/punctuation; `extract_years`: pattern "before 2020" gives `{"year_to": 2019}`; `_find_spans` finds "Lewis Hamilton" (driver, 2 words, beats the 1-word "Hamilton" because longest span wins) and "Silverstone" (circuit). Result: `{"year_to": 2019, "driver": {"driverId": 1, "name": "Lewis Hamilton"}, "circuit": {"circuitId": 9, "name": "Silverstone Circuit"}}`, no ambiguity |
| 5 | `pipeline.choose_intent` | `rule_classifier.classify_rule` lowers the text and tries rules in order. out_of_scope, podium_prediction, head_to_head, pit_stops, grid_gains, qualifying, standings, race_result, circuit_info do not match. `driver_circuit_record` matches `\b(at\|around\|on\|in)\b (the )?(silverstone\|monza\|...)`. Returns `("driver_circuit_record", "rule")`. The follow-up rule and the embedding classifier are never reached |
| 6 | `run_turn` step 2-3 | intent is not None / not out_of_scope; `extraction.ambiguous` is empty |
| 7 | `run_turn` step 4 | not a head-to-head follow-up; `changed = True` (session intent was None); `session.merge(...)` stores the three slots; `session.intent = "driver_circuit_record"`; `turn.found` = the slots named now |
| 8 | `pipeline.usable_slots` | intent changed, so only required slots (`driver`, `circuit` from `templates.REQUIRED`) are carried from the cache, plus everything named in this message (`year_to`) |
| 9 | `graph.missing_slots` | nothing missing |
| 10 | `pipeline.answer_from_graph` | `graph.retrieve(intent, entities)` |
| 11 | `Graph.retrieve` | `select_template` picks the single `driver_circuit_record` variant; `build_params` turns slots into `{"driverId": 1, "circuitId": 9, "year_from": None, "year_to": 2019, "not_started": [...], ...}`; `Graph.run` executes read-only (`routing_="r"`); `build_subgraph` splits rows from graph columns (none here: scalar template); returns `Retrieval(template, cypher, params, rows, subgraph=None, scope)` |
| 12 | empty-rows check | if no rows: `kind = "no_data"`, message `NO_DATA`, no LLM call |
| 13 | `prompts.answer_messages` | system persona, 3 few-shot user/assistant pairs, then the real context + task (details in 3.2) |
| 14 | `ThreadPoolExecutor.map(complete, models)` | both models are called at the same time with the identical prompt; `client.complete` returns an `LLMResult` and never raises |
| 15 | fallback | per model: if `result.error` is set, the text becomes `fallback.read_rows(intent, retrieval)` and `Answer.note = "API connection error"` |
| 16 | `pipeline.log_turn` | appends one JSON line to `logs/turns.jsonl` (fields in 4.4) |
| 17 | UI | `st.rerun()`; `render_turn` draws user bubble, one column per model (model id, text, note, latency, tokens), then `render_trace`; the sidebar `render_sidebar` shows intent and cached slots |

The template (`retrieval/templates.py:DRIVER_CIRCUIT`), exactly as stored:

```cypher
MATCH (d:Driver {driverId: $driverId})<-[:BY_DRIVER]-(res:Result)-[:FOR_RACE]->(ra:Race)
      -[:AT_CIRCUIT]->(c:Circuit {circuitId: $circuitId})
WHERE ($year_from IS NULL OR ra.year >= $year_from)
  AND ($year_to   IS NULL OR ra.year <= $year_to)
MATCH (res)-[:HAS_STATUS]->(s:Status)
WHERE NOT s.status IN $not_started
WITH d, c, count(DISTINCT ra) AS starts,
     sum(CASE WHEN res.position = 1 THEN 1 ELSE 0 END) AS wins,
     sum(CASE WHEN res.position <= 3 THEN 1 ELSE 0 END) AS podiums,
     min(res.positionOrder) AS best_finish,
     avg(res.positionOrder) AS avg_finish,
     min(ra.year) AS first_year, max(ra.year) AS last_year
RETURN d.forename + ' ' + d.surname AS driver, c.circuit_name AS circuit, first_year, last_year,
       starts, wins, podiums, best_finish, avg_finish
LIMIT 1
```

Points to say aloud: three hops from `Result` (to Driver, to Race then Circuit, to Status); the optional year filter
is handled inside the query with `IS NULL`, so one template serves "at Silverstone", "before 2020", "since 2015";
user text never touches the string; named columns; ends with an aggregation and `LIMIT 1`; entries that never started
(`Did not qualify`, `Did not prequalify`, `Withdrew`) are excluded.

**Turn 2: "And at Monza?"**

| # | Step | What happens |
| --- | --- | --- |
| 1 | `extract` | only `{"circuit": {"circuitId": 14, "name": "Autodromo Nazionale di Monza"}}`; no driver, no year |
| 2 | `choose_intent` | in this example the **rule** still fires first (`at monza`), so `how = "rule"`, intent `driver_circuit_record`. The follow-up rule is only reached when no rule matches |
| 3 | `session.merge` | replaces `circuit`; no season slot named, so `driver` and `year_to = 2019` stay in the cache |
| 4 | `usable_slots` | intent unchanged (`changed = False`), so required and optional slots are carried: driver, circuit (new), year_to |
| 5 | rest | same as turn 1 with `circuitId = 14`. The user sees "wins at Monza before 2020" for Hamilton |

A follow-up that really goes through the follow-up rule: **"And in 2015?"**. No rule matches; `choose_intent` sees
`session.intent` is set, `FOLLOW_UP` (`^(and|what about|how about|also|then|same for)\b`) matches, so it returns
`(session.intent, "follow-up")`. `Session.merge` sees a season slot in the message, deletes `year`, `year_from`,
`year_to` from the cache (they are one group) and stores `year = 2015`. Driver and circuit are reused.
A short message (6 words or fewer) that names an entity also counts as a follow-up ("2021?", "What about Norris?").
"What about Norris?" replaces `driver` and keeps the year.

Note: the spec's wording uses "Hamilton" only. The graph has two drivers with that surname (Lewis and Duncan), so
"How many wins did Hamilton have at Silverstone before 2020?" produces the clarification
"Which Hamilton do you mean: Lewis or Duncan?" (this is test item q54 in `eval/qa_testset.json`).

### 3.2 The prompt (`llm/prompts.py:answer_messages`)

Message list sent to Groq:

1. `system`: `PERSONA` ("a meticulous Formula 1 statistician working on the pit wall ... never fill a gap from memory").
2. Three few-shot pairs (`EXAMPLES`): `user` = "EXAMPLE (made-up data)" + context + task, `assistant` = the model answer.
   The names (Anna Example, Ben Sample, Carl Demo) are invented on purpose so they cannot be mistaken for facts;
   the third example has empty rows and shows the "data is insufficient" answer.
3. Final `user`: **context** (`_context`: question, detected intent, extracted entities as JSON, scope sentence, rows as
   JSON lines, whole floats printed as integers, nulls dropped) followed by the **task** (`ANSWER_TASK`: use only the
   context, quote numbers and names exactly, say when insufficient, state the scope in one clause, never add a count,
   year or name, one to three sentences).

`variant="strict"` appends `STRICT_EXTRA` (do not calculate anything; mention only names and numbers from the rows).
The app always uses `few_shot=True` and `variant="base"` (`pipeline.answer_from_graph`).

`client.complete`: Groq SDK with `max_retries=4` (exponential backoff on 429), `temperature=0`,
`max_completion_tokens=1024`, `reasoning_effort="low"` for `openai/gpt-oss*`; reads only `message.content` (the
final answer); an empty answer counts as a failure; any exception becomes `LLMResult(error="API connection error")`
and the real exception type goes only to the Python logger.

### 3.3 Ambiguous name: "How many wins did Schumacher have?"

1. `extract`: the phrase "schumacher" maps to three driver records; `_active` keeps those whose first/last season
   overlap the years named (none named, so all three). More than one candidate: `ambiguous["driver"] = [Michael, Ralf, Mick Schumacher]`.
2. `choose_intent`: rule `\bwins?\b` (the broad last rule) gives `driver_season_stats`.
3. `run_turn` step 3: `extraction.ambiguous` is non-empty. `session.pending = {intent, extraction, ambiguous}`;
   `pipeline.clarification` builds "Which Schumacher do you mean: Michael, Ralf or Mick?" (first names only when all share a surname);
   `turn.kind = "clarify"`. No retrieval, cache not merged, no trace.
4. Reply "Ralf": at the start of the next `run_turn`, `session.pending` is set so `resolve_pending` runs first. It
   matches the reply words against the candidates (all reply words must be inside one candidate's name, plus "the/one/i/mean"),
   re-extracts "Ralf Schumacher", builds a complete `Extraction`, returns the saved intent. `how = "clarification"`. Then the normal path
   continues (merge, retrieve, answer).
5. If the reply names none of the candidates ("Hamilton podiums 2019") `resolve_pending` returns None, the question is dropped and the
   message is processed as a fresh one (checked in `tests/check_pipeline.py`).

A missing required slot works the same way: "How many podiums did he get?" with an empty cache gives
`missing_slots = ["driver"]` and "Which driver do you mean?". The intent was already stored in the session (step 4 runs before
the missing-slot check), so the next message ("Norris", short and naming an entity) goes through the follow-up rule, keeps
that intent and fills the slot (checked in `tests/check_pipeline.py`).

### 3.4 Out-of-scope: "What is the capital of France?"

`classify_rule`: the first rule group (`out_of_scope`) contains `capital of`, so it returns `out_of_scope` immediately.
`run_turn` step 2: `turn.kind = "refuse"`, message `REFUSAL` ("I don't have the knowledge to answer that..."). It returns before the merge, so
**the session cache and cached intent are untouched** (a later "And in 2018?" still works). No `Retrieval`, so `render_turn` draws no trace.
The turn is still logged (with null template/cypher/rows).

If no rule matches, `EmbeddingClassifier.decide` may still pick `out_of_scope`, but only if the text has no F1 vocabulary
(`_F1_WORDS`) or the rules also say out of scope; otherwise the result is `None` and the user is asked to rephrase.

If the embedding classifier returns `None` (score under 0.45 or margin under 0.005): `turn.kind = "rephrase"`, `REPHRASE` message.

### 3.5 Prediction: "Predict Norris at the 2024 Miami Grand Prix"

| # | Step | What happens |
| --- | --- | --- |
| 1 | `extract` | `driver` Lando Norris, `race_name` "Miami Grand Prix" (3-word race phrase beats the 1-word circuit "Miami"), `year` 2024 |
| 2 | `choose_intent` | rule `predict` in the podium_prediction group. Intent `podium_prediction` |
| 3 | `run_turn` step 5 | merge into cache; `turn.entities` = driver, race_name, year from the cache; `turn.prefill = {driver, race_name, year, "values": feature_values(text)}`; `pipeline.feature_values` reads feature values stated in the text with regexes ("starting P3" gives `grid_pos 3`, "qualified 2nd" gives `quali_pos 2`, "from pole" gives `grid_pos 1`, "pit lane start" sets `pit_lane_start`); `kind = "predict"` |
| 4 | `app.render_chat` | stores `st.session_state.arrive = turn.prefill` and `goto = "Podium prediction"`, reruns; `main()` applies `goto` before the radio widget exists |
| 5 | `render_prediction` -> `take_arrival` | sets `sel_driver`, `sel_race`, `sel_year`; `apply_selection(stated)` calls `cached_prefill` -> `features.prefill(driverId, race_name, year)` |
| 6 | `features.prefill` | starts from defaults (training medians); finds the race in `milestone2/races.csv`; finds the driver's row in `models/model_table.csv`; fills every feature that is not NaN; reads the actual finish from `milestone2/results.csv`; returns `(values, filled_names, race_info)` with `race_info` = raceId, date, circuitId, constructorId, actual position, actual podium. Values the user stated in chat override. Fields not filled and still at the default show a gray "default value" note |
| 7 | user presses Predict | `run_prediction(values, model, graph, predictor)` |
| 8 | validate | `features.validate(values)`: missing, non-numeric, non-whole (int/bool) or out-of-range values give error messages; if any, the UI shows `st.error` for each and **no probability is computed** |
| 9 | `PodiumPredictor.predict` | validates again; `pd.DataFrame([values])[features]` in the saved feature order; `prep.transform` (the saved median imputation + scaling); `sigmoid(net(z))` = probability; `predicted_podium = probability >= threshold` (0.69 from `meta.json`); `explainer.shap_values(z)` with the live `shap.DeepExplainer` on the saved 200-row background; contributions sorted by absolute size, in log-odds; also `base_value` and the M1 global importance |
| 10 | `explain.pre_race_facts` | only if `race_info` exists: two Cypher queries with `race.date < date($raceDate)`: the driver's earlier starts/wins/podiums/best finish at this circuit, and the team's not-finished and podium counts in the three years before; if Neo4j fails the explanation still runs without facts |
| 11 | `prompts.explanation_messages` | system persona + one user message: probability, threshold, class, top 3 positive and top 3 negative contributions (`feature = value: +x.xx`), top 8 global features, the facts (or "none available"), then `EXPLAIN_TASK` |
| 12 | `client.complete` | one model (default `openai/gpt-oss-20b`, user can change it) |
| 13 | `render_result` | metrics (probability, class + threshold, actual finish if known); SHAP waterfall (`PodiumPredictor.waterfall_figure`) beside the explanation; if the call failed, shows "API connection error" and `app.fallback_explanation` (deterministic text from the top contributions) |

`EXPLAIN_TASK` rules: 4-6 sentences, for a fan with no statistics background, say which features pushed up/down and
roughly how much (small/moderate/strong), do not explain log-odds maths, never use causal language ("relies on",
"gives weight to"), use only features/values/facts in the context, mention pre-race facts only if listed.

---

## 4. One section per team member

The spec says each member is assessed mainly on their own part and everyone must explain the whole flow.

### 4.1 Member 1: intent classification and entity extraction

Files: `nlu/intents.py`, `nlu/rule_classifier.py`, `nlu/embed_classifier.py`, `nlu/entities.py`,
`eval/intent_testset.csv`, `eval/eval_intents.py`, `eval/results/intent_*`.

**The 12 intents.** 10 retrieval intents (driver_season_stats, driver_circuit_record, constructor_stats, head_to_head,
race_result, circuit_info, qualifying_record, grid_gains, pit_stops, season_standings) + `podium_prediction` +
`out_of_scope`. Each has a description, 10-12 example phrasings (used only by the embedding classifier), and
required/optional slots. The authoritative "which slots are really needed, including either-or" is
`templates.py:REQUIRED` (for example `race_name or circuit`, plus `year`).

**Approach 1, rules (`rule_classifier.py:RULES`, `classify_rule`).** Ordered list of (intent, regex list); the text is
lower-cased; the first intent with any matching pattern wins; `None` if nothing matches.
Order matters and is deliberate:
- `out_of_scope` first (weather, salary, tyres, telemetry, news, opinions, non-F1 topics), so "weather at Silverstone" is refused
  before the track name routes it to a circuit intent.
- then `podium_prediction`, `head_to_head`, `pit_stops`, `grid_gains` (before qualifying, which also says "grid"),
  `qualifying_record`, `season_standings`, `race_result`, `circuit_info`, `driver_circuit_record` (needs "at <track>"),
  `constructor_stats`, and the broad `driver_season_stats` (wins, points, podiums, career) last as the fallback.
- Paraphrase patterns, e.g. head_to_head covers "vs", "versus", "beat", "ahead of", "faster teammate", "compare ... and".
Strength: instant, explainable, no model. Weakness: only knows listed wordings; a new phrasing returns `None` or the wrong intent.

**Approach 2, embedding similarity (`embed_classifier.py:EmbeddingClassifier`).**
- At start-up all example phrasings are embedded once with `normalize_embeddings=True`, so a dot product is the cosine similarity.
- `classify`: embed the question, similarities to all examples, `intent_scores` takes the **max** per intent (one close paraphrase
  is enough; a mean would punish intents with deliberately diverse examples), `decide` applies the rules below.
- `decide`: if best score < `MIN_SIMILARITY` (0.45) or best minus second-best < `MIN_MARGIN` (0.005), `intent = None` (ask to rephrase).
  Otherwise, if the winner is `out_of_scope` and the text contains F1 vocabulary and the rule classifier does not also say out_of_scope,
  `intent = None` too. So out_of_scope is never decided by similarity alone (spec requirement).
- Model: `sentence-transformers/all-mpnet-base-v2`.

**Why that model and those thresholds (`eval/eval_intents.py`).**
- Three models benchmarked on the same test set: `all-MiniLM-L6-v2`, `all-mpnet-base-v2`, `BAAI/bge-small-en-v1.5`.
- Thresholds are **not** tuned on the test set. `leave_one_out_sweep` classifies each classifier example against all the *other* examples
  (diagonal set to -1) for every (min_similarity, min_margin) pair in a grid; `best_thresholds` takes the most precise pair whose
  rephrase rate is at most 10% (`MAX_REPHRASE`), ties go to the more permissive pair.
- Honest caveat for the examiner: the *model* winner is chosen by test accuracy (then LOO precision); only the thresholds are kept off the test set.
- Results: see `eval/results/intent_summary.md`, `intent_model_comparison.csv`, `intent_confusion_{rule,embedding,hybrid}.csv/.png`,
  `intent_threshold_sweep.csv`, `intent_predictions.csv` (per-question predictions for error analysis).
  At the time of writing the summary lists rule-first hybrid best, rules second, embeddings third; re-read the file before the exam.
  "rephrase" counts as wrong and is its own confusion-matrix column.

**Why the deployed classifier is the rule-first hybrid.** It scored best on the labelled set (the hybrid rows in `intent_summary.md`):
rules catch the common wordings with certainty, and embeddings only handle what the rules miss. In the app this is
`pipeline.choose_intent`: rules, then the follow-up rule, then embeddings. (The offline hybrid in `eval_intents.py` is stateless, so it has no follow-up step.)

**Follow-up handling** (spec: a follow-up that names no new intent keeps the cached intent): `pipeline.choose_intent`, after the rules fail:
if a session intent exists and the message starts with and / what about / how about / also / then / same for, or is at most 6 words
(`SHORT_WORDS`) and names an entity (found or ambiguous), return the cached intent with `how = "follow-up"`.

**Entity extraction (`entities.py:EntityExtractor`).**
- Look-up tables come from the graph (`Graph.lookup_tables`), loaded once; a phrase dictionary maps lower-case word tuples to (kind, records).
- Matching: accent- and case-insensitive on whole words, phrases up to 6 words (`MAX_PHRASE_WORDS`).
  `_find_spans` collects all matches, sorts longest first (ties: race, circuit, constructor, driver via `KIND_PRIORITY`), keeps non-overlapping ones.
  So "Red Bull Ring" is a circuit, not the team Red Bull; "Miami Grand Prix" is a race, not the circuit Miami.
- Drivers: full name and surname (surname must be 3+ letters and not in `STOP_WORDS`); upper-case 3-letter codes ("HAM") only if unique.
  The first driver found fills `driver`, the second `driver2`, a third is ignored; the same driver mentioned twice is counted once.
- Constructors: official name, name without "F1 Team", compact form ("redbull"), hand aliases (`CONSTRUCTOR_ALIASES`: "mercedes amg", "rb", "merc", ...).
- Circuits: official name, name without generic words (circuit, international, autodromo, ...), city, aliases (COTA); a country counts as a circuit only if the country has exactly one circuit.
- Races: "British Grand Prix", "British GP", a few aliases (US GP). Output is `race_name` exactly as in the graph.
- Years (`extract_years`): "2014 to 2021" range; "before 2020" gives `year_to = 2019`; "after 2020" gives `year_from = 2021`; "since/from 2015" gives `year_from`;
  "until/up to" gives `year_to`; one bare year gives `year`; several bare years give a span.
- Ambiguity: a name with several records goes to `Extraction.ambiguous[slot]`. For drivers, `_active` first keeps only drivers whose career overlaps the years named
  (assumption: a question about a season concerns a driver active then); "Schumacher in 2022" resolves to Mick, "Schumacher" alone or "in 2001" still asks. The pipeline never guesses.
- It returns only what *this* message names; merging with the cache is the session's job.

**Likely examiner questions**

1. *Why two classifiers and which is deployed?* The spec asks for a rule-based and a neural approach compared on one label set. Rules are exact on known wordings; embeddings generalise to new wordings but confuse close intents. The rule-first hybrid scored best on the held-out set, so it is deployed (numbers: `intent_summary.md`).
2. *How do you stop the embedding classifier from guessing?* Minimum cosine 0.45 and minimum margin 0.005 between the best two intents; below either, the user is asked to rephrase. Thresholds were chosen by leave-one-out on the classifier's own examples, not the test set.
3. *How can a question be marked out of scope?* Rules for known unsupported topics, or the embedding winner is out_of_scope AND no F1 vocabulary appears (or the rules agree). Similarity alone never refuses.
4. *What if the name is shared ("Schumacher")?* The extractor returns all candidates in `ambiguous`; the pipeline asks "Which Schumacher do you mean: Michael, Ralf or Mick?" and stores the question in `session.pending`.
5. *Why is the confusion matrix not square?* The columns have one extra label, `rephrase` (classifier unsure); it counts as an error.
6. *Why max and not mean similarity per intent?* One close paraphrase should be enough; the examples within an intent are intentionally diverse, so a mean would punish them.

### 4.2 Member 2: graph retrieval, prediction, QA test set, error analysis

Files: `retrieval/templates.py`, `retrieval/graph.py`, `prediction/features.py`, `prediction/predictor.py`, `prediction/explain.py`,
`models/*` (notebook export), `eval/qa_testset.json`, `eval/explanation_cases.json`, `eval/results/feature_ranges.csv`, `waterfall_example.png`.

**Templates (`templates.py:TEMPLATES`).** One entry per retrieval intent (pit_stops has two variants). Summary:

| Intent | Template name | Hops / path | Graphical | Notes |
| --- | --- | --- | --- | --- |
| driver_season_stats | driver_stats | Result->Driver, Result->Race, Result->Status | no | races, points, wins, podiums, classified; no year = career |
| driver_circuit_record | driver_circuit_record | Result->Race->Circuit (+ driver, status) | no | wins, podiums, best/avg finish |
| constructor_stats | constructor_stats | Result->Constructor, Race, Status | no | reliability = classified / entries |
| head_to_head | head_to_head | Driver<-Result->Race<-Result->Driver (+ constructor check) | yes | totals + 10 most recent shared races, paths for the subgraph |
| race_result | race_result | Circuit<-Race<-Result->Driver/Constructor/Status | yes | top 10 finishers; race by name or by circuit |
| circuit_info | circuit_info | Race->Circuit, aggregated | no | location, country, races held, first/last year |
| qualifying_record | qualifying_record | Qualifying->Driver, Qualifying->Race | no | poles, front rows, Q3 appearances, average position |
| grid_gains | grid_gains | Driver<-Result->Race->Circuit, Result->Status | yes | top 5 places gained, classified finishers with a grid slot |
| pit_stops | pit_stops_driver / pit_stops_fastest | Driver<-PitStop->Race->Circuit | yes | `when="driver"` selects the per-driver variant, else 5 fastest |
| season_standings | season_standings | Result->Race->Season, Result->Driver | no | top 10 by summed points + number of drivers |

Requirement checks: at least three intents cross two or more hops (driver_circuit_record, head_to_head, race_result, grid_gains, pit_stops, season_standings);
more than one entity: head_to_head (two drivers, optional constructor), driver_circuit_record, grid_gains/pit_stops (race or circuit + year + optional driver);
six or more new relative to the five M2 queries (CONTRACTS.md lists them).

**Design decisions and why.**
- **Parameterised only.** Every value is a `$parameter` (`graph.py:retrieve` passes `params` to `driver.execute_query`); optional slots use `$x IS NULL OR ...` inside the query. No string formatting of user text: prevents Cypher injection and makes the query deterministic.
- **One dynamic template per intent.** A single `build_params` produces every parameter any template may use; unset slots are `None`. `year` is expanded into `year_from = year_to = year`.
- **M2 directions respected**: `(Result)-[:FOR_RACE]->(Race)`, `(Result)-[:BY_DRIVER]->(Driver)`, `(Race)-[:AT_CIRCUIT]->(Circuit)`, etc., never reversed.
- **Small context**: every template ends with `LIMIT` or an aggregation (`circuit_info` is an aggregation without a LIMIT), so the LLM prompt stays short.
- **Not-started entries excluded** (`NOT_STARTED` statuses), as in M2.
- **Graphical vs scalar.** Templates that list individual entities (races, finishers, stops) return Neo4j `path` columns and are `graphical=True`. `graph.py:build_subgraph` removes Node/Relationship/Path values from each row (so the LLM only sees scalar columns) and collects nodes and edges for the UI (drawing, node table, edge table, relationship statements). Aggregations (counts, averages, a leader) are scalar: the UI shows the value only. `Retrieval.subgraph` is None for scalar results or empty rows.
- **Empty result handling**: `rows == []` gives `kind = "no_data"` with `NO_DATA` text and no LLM call, so the model is never asked to fill the gap from memory.
- **`scope` sentence per template** states what the figures cover and exclude (1950-2024 results, no sprint races, qualifying data only 1994-2024, pit stops only 2011-2024, ...). The prompt tells the model to repeat it.

**Prediction (spec section 4).**
- *Exported from the M1 notebook, not retrained.* Section 18 (last cell) saves `preprocessor.joblib` (median imputation + scaling fitted on training seasons), `nn.pt` (weights, n_inputs, hidden layers, dropout), `shap_background.npy` (the exact 200 transformed rows used for M1's SHAP), `global_shap.csv` (M1 mean |SHAP| per feature) and `meta.json` (feature order, threshold, hyperparameters, validation and test metrics). `predictor.py:build_net` rebuilds the architecture; `load_state_dict` loads weights; `net.eval()`. Nothing is fitted in the app.
- *Same explainer setup as M1:* `shap.DeepExplainer(net, background)`; SHAP is computed live for the one row. Values are in log-odds (the network's output before the sigmoid): `base_value + sum(contributions) = logit`; the sizes rank features, they are not percentage points.
- *Threshold 0.69 vs the M1 report's 0.78.* The export cell re-runs the network in the same notebook and stores `nn_inference_threshold`, which was selected on the validation seasons for *that* run. The M1 notebook itself says network results depend on the random seed and PyTorch version (`README.md` and notebook sections 12 and 17 mention differences around 0.01 PR-AUC), so a re-run gives slightly different weights and a different validation-optimal threshold (`meta.json`: 0.69; `report/report.tex` quotes 0.78 from the original run). The app uses the saved model and its own saved threshold consistently. Say this openly; do not claim the two runs are identical.
- *Features and leakage.* Nine features (`features.py:FEATURES`): grid_pos, pit_lane_start, quali_pos, q1_gap_pct, drv_standing, con_standing, drv_podium_last, drv_finish_last, drv_career_podium. They come from `models/model_table.csv` (the M1 `outputs/model_table.csv`), built in M1 with its leakage rules: the cutoff is after qualifying; known before the start are the grid, the qualifying result, standings after the previous round and form from earlier races. Standings are not in the graph, so (per spec) they are not added to it: the graph schema is unchanged and standings come from the M1 table.
- *Valid ranges and defaults.* Ranges are the observed minimum and maximum in the M1 model table (1950-2024), shares widened to 0-1 (outside means the model would extrapolate); defaults are training-season medians (what the M1 imputer fills). Evidence: `eval/results/feature_ranges.csv` (min, 1%, median, 99%, max). `validate` refuses missing, non-numeric, non-whole (int/bool) and out-of-range values with a message naming the field.
- *Facts strictly before the race date* (`explain.py`): `race.date < date($raceDate)` in both queries; the team query also limits to the preceding three years.

**QA test set and error analysis.** `eval/qa_testset.json`: items with id, question, intent, gold entities, category (normal / follow_up / ambiguous / empty / out_of_scope) and `follows` (the item it continues). Gold facts are not hand-written: `eval_llm.py:qa_items` runs the correct template on the gold entities at evaluation time. Ambiguous items list candidate names instead of one driver and are skipped in the LLM comparison (they never reach the answer prompt); prediction items likewise. Error analysis groups failures by stage: intent (`intent_predictions.csv`), entity (extractor output vs gold), query (empty or wrong rows), model misread the rows (correctness metric), hallucination (faithfulness metric); see `eval/results/llm_runs.jsonl` and `llm_summary.md` for the examples.

**Likely examiner questions**

1. *Why can the LLM not write Cypher?* Determinism, safety and testability: every intent has a reviewed template; user text only becomes parameters; wrong or injected queries are impossible.
2. *What does "grounded" mean for you?* The LLM receives only the retrieved rows, the entities and a scope sentence, and is told to use nothing else; if there are no rows it is not called at all.
3. *Scalar versus graphical templates?* Graphical ones return paths so the UI can draw the subgraph (nodes, edges, relationship statements); scalar ones return aggregates where a drawing would not add information, so only the value is shown.
4. *Why is the saved threshold 0.69, not 0.78?* See above: the exported network is a re-run; its validation-chosen threshold is stored in `meta.json` and used with exactly that network.
5. *How do you avoid leakage in the prediction view?* Features are the M1 pre-race features (nothing known only after the start); graph facts are filtered with `race.date < raceDate`. Prefill takes a past race's pre-race row, and the actual finish is displayed only afterwards for comparison, never fed to the model.
6. *Is a SHAP value a cause?* No. It is how much the fitted network relied on that feature for that row, in log-odds; the prompt forbids causal wording.

### 4.3 Member 3: LLM layer and evaluation

Files: `llm/client.py`, `llm/prompts.py`, `llm/fallback.py`, `eval/eval_llm.py`, `eval/qualitative_sheet.py`, `config.py` (`LLM_MODELS`, `API_ERROR`, `GROQ_API_KEY`).

**Models (`config.LLM_MODELS`)**: three Groq free-tier models of different families and sizes: `allam-2-7b` (SDAIA, 7B), `qwen/qwen3.8-27b` (Alibaba Qwen, 27B per the config comment), `openai/gpt-oss-20b` (OpenAI open-weight MoE, 20B). The chat lets the user pick two (defaults A = first, B = second); the explanation view defaults to gpt-oss-20b.

**Prompt design (`prompts.py`).** Three parts as the spec requires: persona (system message, a "meticulous statistician" so it reports and does not recall), context (question, intent, entities, scope, rows), task (rules). Few-shot = worked examples with made-up, labelled data in the same layout as the real turn; one of them shows the empty-rows refusal wording. Zero-shot = same prompt without the example pairs (`few_shot=False`). The key stays in `.env`, never in code or logs.

**Failure handling.** `complete` never raises; a failed call returns `error = "API connection error"` and zero tokens. `pipeline.answer_from_graph` swaps the text for `fallback.read_rows`: a single row becomes `Result: col: value, ...` + scope; several rows become a numbered list (max 10, id columns hidden, whole floats as integers). The UI shows only "API connection error" as the note.

**Metrics (`eval_llm.py`, all automatic, averaged per variant x model x style).**

| Metric | Definition |
| --- | --- |
| correctness | share of the gold numbers and names (surname) in the rows that appear in the answer (`correctness`, `gold_facts`; id columns skipped) |
| faithfulness | answer is faithful if it has no number and no capitalised name-like word that is missing from the context (`is_faithful`, `capitalised_names`; a lone name opening a sentence is not checked) |
| refusal | share of no-data / out-of-scope items (empty rows) whose answer says the data is insufficient (`NO_DATA_WORDS` regex); measured at the LLM level, although the app refuses out-of-scope earlier in the pipeline |
| explanation fidelity | share of the three strongest local contributions (by absolute SHAP) named in the explanation, counted 0 if the text names a feature not in the context (`explanation_scores`) |
| latency, tokens | mean latency and mean input/output tokens over the successful calls |

Results: see `eval/results/llm_comparison.csv` and `llm_summary.md` (rebuilt from `llm_runs.jsonl`; the summary also has few-shot minus zero-shot per model).
Qualitative sheet: `qualitative_sheet.py` takes the first 4 questions per intent (40 for 10 intents) from the few-shot base runs, writes `eval/results/qualitative_sheet.csv`
with empty columns `fluency_rater1`, `fluency_rater2`, `causal_language` for two raters to fill by hand (rubric 1-5 printed by the script).

**Zero- vs few-shot and the before/after improvement.** `eval_llm.py` runs every item in both styles with the same rows. The improvement experiment is the `strict` variant
(`STRICT_EXTRA`: no calculations, only names and numbers from the rows), aimed at the failure of "extra numbers and names that are not in the rows" (faithfulness). Run `--variant base`, then `--variant strict`;
both are stored side by side in `llm_runs.jsonl`. Record the before/after numbers from `llm_summary.md`.

**Why Groq/free tier:** the spec says paid APIs are not required; Groq serves open-weight models free with rate limits, hence `max_retries=4` and the 1 s pause between calls in the evaluation (`PAUSE_S`).

**Likely examiner questions**

1. *Why those three models?* Different families and sizes (7B Allam, 27B Qwen, 20B gpt-oss MoE) on one provider, same prompts and same retrieved rows, so differences come from the model.
2. *How is faithfulness measured?* Automatically: any number or capitalised name in the answer that is not in the prompt context makes it unfaithful. It is a heuristic (it can miss rephrased claims and flag harmless capitalised words), which is why there is also a manual qualitative sheet.
3. *Zero-shot vs few-shot, what changes?* Few-shot adds three worked examples in the same layout (made-up data); it costs more input tokens and shows the model the expected style and the "insufficient data" answer. Numbers: `llm_summary.md`.
4. *What does the user see if the API fails?* Only "API connection error", plus the facts via `read_rows`; the real exception is logged to the Python logger, not shown.
5. *How do you stop the model from adding facts?* Prompt rules (use only the context, never add a count, year or name), temperature 0, no rows means no call, and the strict variant forbids calculating. The metrics check it.
6. *Where does the persona help?* It frames the model as a statistician who reports data, which supports the "do not fill gaps from memory" rule.

### 4.4 Member 4: session, integration, UI

Files: `session.py`, `pipeline.py`, `app.py`, `tests/check_pipeline.py`, `config.py` (paths, log file).

**Session (`session.py:Session`).** Fields: `intent` (last in-domain intent), `entities` (slot cache), `pending` (open clarification), `history` (list of `Turn`, rendered as the chat thread). `merge(found)` replaces only the slots the new message names. The season slots (`year`, `year_from`, `year_to`) are one group: naming any of them removes all three first, so "since 2015?" replaces a cached single year and "2019?" replaces a cached range. `clear()` is the sidebar reset.

**Pipeline rules worth knowing (`pipeline.py:run_turn`).**
- **Order:** open clarification reply, extract + choose intent, rephrase / refuse, ambiguity question, merge, prediction hand-over, missing slots question, retrieve + answer.
- **Out of scope does not clear the cache** (spec): it returns before `merge`.
- **Stale optional slots:** `usable_slots` carries cached slots to retrieval, but after an explicit change of intent only the *required* slots of the new intent are carried (e.g. a driver from a previous question is not silently applied to "fastest pit stops at Monaco 2019").
- **Head-to-head opponent replacement:** if the intent stays `head_to_head` and a follow-up names exactly one driver ("And against Rosberg?"), the extractor's `driver` is moved to `driver2`, so the cached first driver stays and the opponent changes.
- **Pronouns:** there is no pronoun resolver. "How many points did he score in 2019?" works because the driver slot is simply missing from the message and the cached driver is reused.
- **Missing slots:** `graph.missing_slots` + `SLOT_NAMES` give "Which driver do you mean?" (kind `clarify`); only the first ambiguous slot is asked at a time.
- **Error containment:** `handle_turn` catches `DriverError`, `Neo4jError`, `OSError` (graph down) and shows `GRAPH_DOWN`; LLM errors never propagate (`complete` returns them as data).
- **Turn kinds:** `answer | no_data | clarify | refuse | rephrase | predict | error`; `Turn.how` records which stage decided the intent: `rule | follow-up | embedding | clarification`.

**Logging (`pipeline.py:log_turn`).** One JSON line per turn in `logs/turns.jsonl` (git-ignored):
`timestamp`, `user_message`, `kind`, `intent`, `classifier`, `entities`, `template`, `cypher`, `params`, `rows`, `prompt`, and `models` = list of
`{model, answer, latency_s, input_tokens, output_tokens, error}`. This covers the spec's list (message, intent, entities, template, Cypher, retrieved results, prompt, answer, model, latency, token counts).
Clarification, refusal and rephrase turns have nulls for the retrieval fields.

**UI (`app.py`).**
- Heavy objects (`Graph`, extractor, classifier, predictor) are created once with `st.cache_resource`.
- Sidebar: view switch (Chat / Podium prediction), the cached intent and the entity slot table (so "And in 2020?" is visibly using the previous driver), a note when a clarification is pending, reset button.
- Chat: two model pickers, history with clarifying questions in the same thread, one column per model with the model id, the answer, the "API connection error" note when failed, latency and tokens. `render_trace` (only for turns with a `Retrieval`) shows intent + decider + template, entities, then for graphical results the subgraph (Graphviz drawing capped at 60 nodes, node table, edge table, relationship statements with counts), for scalar results the value, then Cypher, parameters (JSON), raw rows. Refusals, clarifications and rephrase requests have no trace.
- Prediction view: driver / race / season selectors prefill the nine fields from the M1 table; one input per feature (checkbox for pit-lane start, integer or decimal number inputs with help text showing definition, unit and valid range); gray "default value" note for fields neither supplied nor changed; Predict button; errors from `validate` are shown and no probability is returned; results: probability, class + threshold, actual finish if known, SHAP waterfall beside the explanation text.
- Streamlit detail: a chat turn of kind `predict` sets `arrive` and `goto` in `st.session_state`; `main()` applies `goto` before the radio widget is created, and `take_arrival` runs before the input widgets are created. `apply_selection` bumps a `version` counter so the inputs get new widget keys and restart from the new values.
- Selections are copied into `st.session_state.kept` because Streamlit drops widget state for widgets not drawn in a run.

**Tests.** `tests/check_pipeline.py` swaps the graph and LLM for fakes and asserts follow-ups, season replacement, refusal keeping the cache, clarification flow, missing slot, empty rows, API-error fallback, the prediction hand-over (stated feature values) and the log. It needs neither Neo4j nor Groq.

**Likely examiner questions**

1. *Walk me through "And in 2020?"* No rule matches; a session intent exists and the text starts with "and", so the cached intent is kept (`how = follow-up`); the extractor finds `year = 2020`; `Session.merge` clears the season group and stores 2020; the driver stays in the cache; the same template runs with the new year.
2. *What happens when the user asks something off-topic in the middle?* The refusal is returned before the merge; the cache and cached intent are untouched, so the next follow-up still works.
3. *How is the clarification kept across turns?* `session.pending` stores the intent, the partial extraction and the candidates; the next `run_turn` tries `resolve_pending` first; if the reply does not name exactly one candidate the question is dropped.
4. *What is logged and why?* Everything needed to audit a turn (see above), including the prompt and token counts, for error analysis and the evaluation.
5. *How do you validate the manual feature input?* `features.validate` (per-feature type, whole-number and range rules, ranges from the observed M1 data); the UI refuses and no probability is returned; `PodiumPredictor.predict` validates again and raises `ValueError`.
6. *How do the two models run at the same time?* `ThreadPoolExecutor.map` over the selected models with the identical prompt and rows; each call returns an `LLMResult`; a failed one gets the deterministic fallback text.

---

## 5. Glossary

| Term | Definition (say it in one or two sentences) |
| --- | --- |
| Graph-RAG | Retrieval-augmented generation where the retrieved context comes from a knowledge graph via structured queries, and the LLM only verbalises it. Here: intent -> entities -> Cypher -> rows -> LLM |
| Grounding | Forcing the answer to rest on retrieved evidence: the LLM sees only the rows and is told to use nothing else; no rows means no LLM call |
| Intent | The type of question the user asks (12 here); it selects the template and the required slots |
| Slot / entity | A value the template needs (driver, circuit, year ...); entity extraction fills slots from text |
| Entity cache | The session's stored slots, reused by follow-ups; only slots named in the new message are replaced |
| Cosine similarity | Dot product of two unit vectors; 1 = same direction. Here between the question embedding and each example phrasing |
| Margin | Best intent score minus second-best; a tiny margin means two intents are almost equally likely, so ask the user |
| Minimum score | Lowest acceptable best similarity (0.45); below it the question is unlike all examples |
| Embedding | A vector from a sentence-transformer model (`all-mpnet-base-v2`) in which similar meanings are close |
| Rule-based classification | Ordered keyword/regex patterns; first match wins |
| Parameterised query | A Cypher string with `$placeholders` whose values are sent separately; user text is never concatenated into the query |
| Hop | One relationship traversal in a pattern (Result -> Race is one hop; Result -> Race -> Circuit is two) |
| Template | A stored, reviewed Cypher query with parameters; one template covers many questions |
| Subgraph | The nodes and edges returned by a graphical template, drawn in the UI |
| Scope sentence | One sentence stating what the figures cover and exclude, included in the context and answer |
| SHAP | Shapley-value attributions: how much each feature moved this prediction away from the average (base) prediction |
| DeepExplainer | SHAP's explainer for neural networks; uses a background sample (here 200 saved rows) as the reference |
| Log-odds (logit) | `ln(p / (1 - p))`; the network output before the sigmoid. SHAP values add up to it: base + contributions = logit |
| Threshold | Probability cut-off for predicting "podium"; chosen on validation seasons in M1 (0.69 in the exported run) |
| Zero-shot / few-shot | Prompt without / with worked example answers |
| Faithfulness | The answer contains nothing (numbers, names) beyond the retrieved context |
| Hallucination | A fluent statement not supported by the context or the data |
| Leakage | Using information that would not be known at prediction time (e.g. finishing results); M1 uses only pre-race information |
| Explanation vs cause | A high SHAP value means the model relies on a feature, not that the feature causes a podium |

---

## 6. Assumptions and limitations

### 6.1 Assumptions documented in code and CONTRACTS.md

Schema and Cypher
- The M2 graph is read-only: no new labels, relationships or properties; missing CSV values are absent properties (`CONTRACTS.md`).
- Parameters only, M2 directions, named columns, every query ends with `LIMIT` or an aggregation (`CONTRACTS.md`).
- Data covers World Championship race results 1950 to 2024; sprint races are not in the data and sprint points are missing from points (`templates.py:BASE_SCOPE`).
- Entries with status "Did not qualify", "Did not prequalify" or "Withdrew" never started and are excluded from counts; other unfinished or disqualified entries count as starts (`NOT_STARTED`, `SCOPE_RESULTS`).
- "Classified" means status `Finished` or starting with `+` (lapped); reliability = classified / entries, every car counts (`constructor_stats`).
- "Ahead" in head-to-head = better `positionOrder`; only races where both have a result count.
- Race results show the top 10 only; grid gains only classified finishers with a grid slot, top 5; standings are summed race points (no dropped-score rules, no sprint points), top 10.
- Qualifying data exists 1994-2024 (Q3 times since 2006); a pole is qualifying position 1. Pit-stop data 2011-2024, pit-lane time.
- A race can be given by name or by circuit (`race_name or circuit`), which is why `REQUIRED` differs from `Intent.required` for race_result, grid_gains and pit_stops.

NLU and session
- A question about a season concerns a driver active in that season (`entities.py:_active`).
- Several bare years in a message are treated as the span between the first and the last.
- A country names a circuit only if the country has exactly one circuit; very short or common-word names are never matched alone (`STOP_WORDS`).
- Renamed teams are different constructors (no lineage table in the data).
- Max-similarity per intent; thresholds from leave-one-out on the classifier's examples with at most 10% rephrase rate.
- A follow-up is a message starting with and / what about / how about / also / then / same for, or at most 6 words that names an entity (`FOLLOW_UP`, `SHORT_WORDS`).
- In a head-to-head follow-up a single new driver is the new opponent. An unrelated reply to a clarification drops the clarification.
- After an explicit intent change only required slots are carried from the cache (`usable_slots`).

Prediction (also M1 assumptions, section 17 of the notebook)
- Known at the cutoff: the grid, the qualifying result, standings after the previous round; the cutoff is after qualifying.
- A pit-lane start counts as the back of the field; the Indianapolis 500 is treated as a separate population; missing values are filled with training medians.
- Valid ranges = observed min/max in the M1 table (shares 0-1); defaults = training medians (`features.py`).
- Facts for the explanation use only races with `date < raceDate`; the team window is three years; "not finished" means status other than `Finished` or `+N Laps`.
- The threshold is whatever validation chose in the exported run (`meta.json`); probabilities are not calibrated and should be read as ranking scores (notebook section 17).
- SHAP background = the 200 sampled rows saved by the notebook; SHAP is on the log-odds scale.

LLM and evaluation
- Temperature 0, `max_completion_tokens=1024`, `reasoning_effort="low"` for gpt-oss; the same prompt and rows for every model.
- Few-shot examples use made-up names and numbers and are labelled so.
- Gold facts are the rows returned by the correct template at evaluation time, not hand-written numbers.
- Metrics are automatic heuristics (see 4.3); the refusal metric is tested at the LLM level with empty rows.

### 6.2 Known limitations

- Rules only know listed wordings; embeddings confuse near intents (for example constructor vs driver stats). See the confusion matrices and `intent_predictions.csv`.
- Out-of-scope detection relies on keyword rules and an F1-vocabulary check; unusual off-topic or on-topic questions can be refused wrongly or sent to the wrong intent.
- Entity extraction is dictionary-based: typos, nicknames not in the aliases, and two drivers with the same full name are not handled; pronouns are not resolved, only reused from the cache; only the first ambiguous slot is asked at a time.
- Counts are limited by the data: no sprint races, qualifying only from 1994, pit stops only from 2011, standings are summed points without dropped scores.
- Free-tier LLMs: rate limits, variable latency, and the answers may still contain small unfaithful phrases; the metrics are heuristic.
- Prediction: unseen races (no row in the model table) get default feature values and no graph facts; the model leans heavily on the grid position; probabilities are uncalibrated; SHAP describes the model, not causes; M1 limitations apply (qualifying coverage drift, era sensitivity, small test period).
- The saved network is a re-run of the M1 notebook, so its numbers (threshold 0.69, metrics in `models/meta.json`) differ slightly from the figures printed in the M1 report.
- Prediction-view explanations are not written to `logs/turns.jsonl` (only chat turns are).
