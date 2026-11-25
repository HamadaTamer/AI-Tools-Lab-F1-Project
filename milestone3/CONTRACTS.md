# Milestone 3 – shared plan and interface contracts

Every part of the system is written against the interfaces below so the four parts
(NLU, retrieval + prediction, LLM layer, integration) can be built in parallel and
plugged together without changes. The spec is `../../milestone 3.pdf` (one level above
the repo root: `C:\Users\ahmed\OneDrive\Desktop\Uni\Semester 9\ACL\milestone 3.pdf`).

## Ground rules (apply to every file)

- Python 3.13, run from the repo root with `.venv/Scripts/python`. Imports are
  package-relative to `milestone3/` (run as `python -m milestone3.<module>`; every
  sub-folder has an empty `__init__.py`).
- Succinct, human-readable code. A module docstring saying what the file does and how
  it fits the pipeline; a short comment above every non-obvious block. No speculative
  abstractions, no defensive code for things that cannot happen, no long banners.
- Settings come from `milestone3/config.py`. Never hard-code the Groq key or the Neo4j
  password; never print them.
- The Milestone 2 graph schema is read-only: no new labels, relationships or
  properties. Pre-race features that the graph lacks (standings) come from the
  Milestone 1 tables (`milestone3/models/model_table.csv`, saved by the notebook export cell).
- Cypher: parameters only (`$driverId`), never string-formatted user text. Respect the
  M2 directions. Named columns. Every query ends with `LIMIT` or an aggregation.

## Graph schema (Milestone 2, unchanged)

Nodes: `Driver{driverId, driverRef, number, code, forename, surname, dob, nationality}`,
`Constructor{constructorId, constructorRef, constructor_name, constructor_nationality}`,
`Circuit{circuitId, circuitRef, circuit_name, circuit_loc, circuit_country, lat, lng, alt}`,
`Season{year}`, `Status{statusId, status}`, `Race{raceId, year, round, name, date, time}`,
`Result{resultId, number, grid, position, positionText, positionOrder, points, laps, time,
milliseconds, fastestLap, rank, fastestLapTime, fastestLapSpeed}`,
`Qualifying{qualifyId, number, position, q1, q2, q3}`,
`PitStop{eventId, stop, lap, time, duration, milliseconds}`.

Relationships: `(Race)-[:AT_CIRCUIT]->(Circuit)`, `(Race)-[:IN_SEASON]->(Season)`,
`(Result|Qualifying|PitStop)-[:FOR_RACE]->(Race)`, `(Result|Qualifying|PitStop)-[:BY_DRIVER]->(Driver)`,
`(Result|Qualifying)-[:FOR_CONSTRUCTOR]->(Constructor)`, `(Result)-[:HAS_STATUS]->(Status)`.
Missing CSV values are absent properties (e.g. `Result.position` is null for non-classified).

## Intents (12)

| intent | family (spec table row) | required slots | optional slots |
| --- | --- | --- | --- |
| `driver_season_stats` | Driver performance | driver | year **or** year_from/year_to (none = career) |
| `driver_circuit_record` | Driver performance | driver, circuit | year_from/year_to (e.g. "before 2020") |
| `constructor_stats` | Constructor performance | constructor | year or year range (points, wins, reliability) |
| `head_to_head` | Head-to-head | driver, driver2 | constructor, year range |
| `race_result` | Race and circuit facts | race_name, year | – |
| `circuit_info` | Race and circuit facts | circuit | – (location, country, races held, first/last year) |
| `qualifying_record` | Qualifying and grid | driver | year or year range (poles, Q3 appearances, avg quali pos) |
| `grid_gains` | Qualifying and grid | race_name **or** circuit, year | – (top gainers grid → finish) |
| `pit_stops` | Pit stops and strategy | race_name, year | driver (fastest stops; stops per driver) |
| `season_standings` | Season and history | year | – (points leader / top 10, number of drivers) |
| `podium_prediction` | → prediction view | driver | race_name, year, any pre-race feature value |
| `out_of_scope` | refusal | – | – |

Six or more are new relative to the five M2 queries (race_result, circuit_info,
qualifying_record, grid_gains, pit_stops, season_standings, driver_season_stats per driver).
At least three cross ≥2 hops (Result→Race→Season / Circuit), and head_to_head,
driver_circuit_record, grid_gains take more than one entity.

## Entity slots (one dict, shared by every stage)

```python
entities = {
    "driver":      {"driverId": 1, "name": "Lewis Hamilton"},     # optional
    "driver2":     {"driverId": 822, "name": "Valtteri Bottas"},  # optional
    "constructor": {"constructorId": 131, "name": "Mercedes"},
    "circuit":     {"circuitId": 9, "name": "Silverstone Circuit"},
    "race_name":   "British Grand Prix",          # Race.name exactly as in the graph
    "year":        2019,                          # single season
    "year_from":   2014, "year_to": 2021,         # inclusive range; "before 2020" -> year_to=2019
}
```
Absent slots are simply not in the dict.

## Module interfaces

```python
# nlu/intents.py           (part 1)
@dataclass
class Intent: name: str; description: str; examples: list[str]; required: list[str]; optional: list[str]
INTENTS: dict[str, Intent]

# nlu/rule_classifier.py   (part 1)
def classify_rule(text: str) -> str | None          # None = no pattern matched

# nlu/embed_classifier.py  (part 1)
@dataclass
class IntentPrediction: intent: str | None; score: float; runner_up: str | None; margin: float
class EmbeddingClassifier:
    def __init__(self, model_name=config.EMBEDDING_MODEL): ...
    def classify(self, text: str) -> IntentPrediction  # intent None = ask to rephrase

# nlu/entities.py          (part 1)
@dataclass
class Extraction:
    entities: dict                 # slots found in this message (format above)
    ambiguous: dict[str, list[str]]  # slot -> candidate display names, e.g. {"driver": ["Michael Schumacher", "Ralf Schumacher", "Mick Schumacher"]}
class EntityExtractor:
    def __init__(self, tables: dict): ...   # tables = Graph.lookup_tables(), loaded once at start-up
    def extract(self, text: str) -> Extraction

# lookup_tables() format (part 2 produces it, part 1 consumes it):
# {"drivers":      [{"driverId", "forename", "surname", "code", "driverRef", "first_year", "last_year"}],
#  "constructors": [{"constructorId", "name", "constructorRef"}],
#  "circuits":     [{"circuitId", "name", "location", "country", "circuitRef"}],
#  "races":        ["Australian Grand Prix", ...]}          # distinct Race.name values

# retrieval/graph.py       (part 2)
@dataclass
class Retrieval:
    template: str; cypher: str; params: dict; rows: list[dict]
    subgraph: dict | None   # {"nodes": [{"id","label","name"}], "edges": [{"source","rel","target"}]} or None for scalar results
    scope: str              # one sentence: what the figures cover / exclude
class Graph:
    def lookup_tables(self) -> dict    # {"drivers": [...], "constructors": [...], "circuits": [...], "races": [names]}
    def retrieve(self, intent: str, entities: dict) -> Retrieval
    def run(self, cypher: str, params: dict) -> list[dict]

# retrieval/templates.py   (part 2)
TEMPLATES: dict[str, Template]   # intent -> Template(name, cypher, build_params(entities)->dict, graphical: bool, scope)

# llm/client.py            (part 3)
@dataclass
class LLMResult: model: str; text: str | None; error: str | None; latency_s: float; input_tokens: int; output_tokens: int
def complete(model: str, messages: list[dict], temperature: float = 0) -> LLMResult   # never raises

# llm/prompts.py           (part 3)
def answer_messages(question, intent, entities, retrieval, few_shot=True) -> list[dict]   # persona / context / task
def explanation_messages(prediction, graph_facts) -> list[dict]

# llm/fallback.py          (part 3)
def read_rows(intent: str, retrieval) -> str    # deterministic sentence(s) from the rows

# prediction/features.py   (part 2)
FEATURES: list[FeatureSpec]   # name, label, unit, help, kind(int/float/bool), min, max, default
def validate(values: dict) -> list[str]           # error messages; empty = valid
def prefill(driver_id, race_name=None, year=None) -> tuple[dict, set[str], dict | None]
    # (values, names_filled_from_data, race_info incl. actual finish if known)

# prediction/predictor.py  (part 2)
class PodiumPredictor:          # loads milestone3/models/ once; never refits
    def predict(self, values: dict) -> dict
    # {"probability", "predicted_podium", "threshold", "contributions": [(feature, value, shap)], "base_value", "global_importance": [...]}
    def waterfall_figure(self, prediction) -> matplotlib.figure.Figure

# prediction/explain.py    (part 2)
def pre_race_facts(graph, driver_id, constructor_id, circuit_id, race_date) -> list[str]   # strictly before race_date

# pipeline.py / session.py / app.py   (part 4)
# handle_turn(text, session) -> Turn ; session holds intent + entity cache; every turn logged to config.LOG_FILE
```

## Test-set formats

- `eval/intent_testset.csv` (part 1): `question,intent` – ≥50 rows, ≥4 per intent, phrased
  differently from the classifier examples in `nlu/intents.py`.
- `eval/qa_testset.json` (part 2): list of
  `{"id": "q01", "question": "...", "intent": "...", "entities": {...gold slots...},
    "category": "normal|ambiguous|follow_up|out_of_scope|empty", "follows": "q00" | null}`
  – ≥30 items. Gold facts are NOT written by hand: evaluation runs the gold template.
- `eval/explanation_cases.json` (part 2): list of `{"driver": "...", "driverId": 1,
  "race_name": "...", "year": 2023}` – ≥5, at least one predicted podium and one not.

## Folder layout

```
milestone3/
  config.py  CONTRACTS.md  pipeline.py  session.py  app.py
  nlu/         intents, rule + embedding classifiers, entity extraction
  retrieval/   Cypher templates, Neo4j access, subgraph building
  llm/         Groq client, prompts, fallback reading
  prediction/  feature specs + validation, saved-model predictor, explanation facts
  models/      artifacts exported by the M1 notebook (NN weights, preprocessor, SHAP)
  eval/        test sets, evaluation scripts, results/
```
