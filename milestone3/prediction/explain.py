"""Graph facts for the prediction explanation.

Part of the prediction view: next to the SHAP contributions of the NN, the explanation shows a
few facts from the knowledge graph about the driver and the team before the race. Every query
only looks at races strictly before the race date, so the facts could have been known beforehand.
`graph` is any object with `run(cypher, params) -> list[dict]` (retrieval/graph.py).
"""
from milestone3.retrieval.templates import NOT_STARTED

# the driver's record at this circuit before the race (OPTIONAL MATCH: a first visit still returns names).
# Same counting rules as the driver_circuit_record template: never-started entries are not starts.
DRIVER_AT_CIRCUIT = """
MATCH (d:Driver {driverId: $driverId}), (c:Circuit {circuitId: $circuitId})
OPTIONAL MATCH (r:Result)-[:BY_DRIVER]->(d), (r)-[:FOR_RACE]->(race:Race)-[:AT_CIRCUIT]->(c),
               (r)-[:HAS_STATUS]->(s:Status)
WHERE race.date < date($raceDate) AND NOT s.status IN $notStarted
RETURN d.forename + ' ' + d.surname AS driver, c.circuit_name AS circuit, count(r) AS starts,
       sum(CASE WHEN r.position = 1 THEN 1 ELSE 0 END) AS wins,
       sum(CASE WHEN r.position <= 3 THEN 1 ELSE 0 END) AS podiums,
       min(r.positionOrder) AS best
LIMIT 1
"""

# the team's starts, races not finished and podium finishes in the three years before the race
TEAM_RECENT = r"""
MATCH (k:Constructor {constructorId: $constructorId})
OPTIONAL MATCH (r:Result)-[:FOR_CONSTRUCTOR]->(k), (r)-[:FOR_RACE]->(race:Race), (r)-[:HAS_STATUS]->(s:Status)
WHERE race.date < date($raceDate) AND race.date >= date($raceDate) - duration({years: 3})
RETURN k.constructor_name AS constructor, count(r) AS starts,
       sum(CASE WHEN s IS NULL OR s.status = 'Finished' OR s.status =~ '\\+\\d+ Laps?' THEN 0 ELSE 1 END) AS not_finished,
       sum(CASE WHEN r.position <= 3 THEN 1 ELSE 0 END) AS podiums
LIMIT 1
"""


def plural(n, word) -> str:
    """'1 win', '3 wins'."""
    return f"{n} {word}" + ("" if n == 1 else "s")


def pre_race_facts(graph, driver_id, constructor_id, circuit_id, race_date) -> list[str]:
    """Short factual sentences about the driver and team from races strictly before `race_date` (YYYY-MM-DD)."""
    facts = []

    # driver at this circuit
    rows = graph.run(DRIVER_AT_CIRCUIT, {"driverId": driver_id, "circuitId": circuit_id, "raceDate": race_date,
                      "notStarted": NOT_STARTED})
    if rows:
        r = rows[0]
        if r["starts"] == 0:
            facts.append(f"Before this race {r['driver']} had never started at {r['circuit']}.")
        else:
            facts.append(f"At {r['circuit']} {r['driver']} had {plural(r['starts'], 'earlier start')} with "
                         f"{plural(r['wins'], 'win')}, {plural(r['podiums'], 'podium')} and a best finish of P{r['best']}.")

    # team in the previous three years
    rows = graph.run(TEAM_RECENT, {"constructorId": constructor_id, "raceDate": race_date})
    if rows and rows[0]["starts"] > 0:
        r = rows[0]
        facts.append(f"In the three years before this race {r['constructor']} did not finish {r['not_finished']} of "
                     f"{plural(r['starts'], 'car start')} ({r['not_finished'] / r['starts']:.0%}) and scored "
                     f"{plural(r['podiums'], 'podium')}.")
    return facts
