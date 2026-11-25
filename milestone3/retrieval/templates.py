"""Cypher templates for the ten retrieval intents of the assistant.

Pipeline role: once the NLU stage has an intent and its entity slots, `graph.py` picks a
template from here, fills its parameters from the slots and runs it on the Milestone 2
graph. User text never reaches the Cypher string: every value travels as a `$parameter`,
and optional slots are handled inside the query with `IS NULL` checks.

Graphical templates return Neo4j paths (columns named `path*`) next to their scalar
columns; `graph.py` turns those paths into the subgraph the UI draws.
"""
from dataclasses import dataclass
from typing import Callable

# Statuses of entries that never started a race; they are excluded from every count (as in M2).
NOT_STARTED = ["Did not qualify", "Did not prequalify", "Withdrew"]

# Shared wording for the scope sentence of every template.
BASE_SCOPE = ("World Championship race results from 1950 to the 2024 season; "
              "sprint races are not in the data")


@dataclass
class Template:
    name: str
    cypher: str
    build_params: Callable[[dict], dict]
    graphical: bool
    scope: str
    when: str | None = None   # slot that must be present for this variant to be chosen


def build_params(entities):
    """Every parameter any template may use; slots the user did not give become None."""
    year = entities.get("year")
    ids = lambda slot, key: (entities.get(slot) or {}).get(key)
    return {
        "driverId": ids("driver", "driverId"),
        "driver2Id": ids("driver2", "driverId"),
        "constructorId": ids("constructor", "constructorId"),
        "circuitId": ids("circuit", "circuitId"),
        "race_name": entities.get("race_name"),
        "year": year,
        "year_from": year if year is not None else entities.get("year_from"),
        "year_to": year if year is not None else entities.get("year_to"),
        "not_started": NOT_STARTED,
    }


# Slots each intent needs; a tuple means "any one of these is enough".
REQUIRED = {
    "driver_season_stats": [("driver",)],
    "driver_circuit_record": [("driver",), ("circuit",)],
    "constructor_stats": [("constructor",)],
    "head_to_head": [("driver",), ("driver2",)],
    "race_result": [("race_name", "circuit"), ("year",)],
    "circuit_info": [("circuit",)],
    "qualifying_record": [("driver",)],
    "grid_gains": [("race_name", "circuit"), ("year",)],
    "pit_stops": [("race_name", "circuit"), ("year",)],
    "season_standings": [("year",)],
}

# ---------------------------------------------------------------------------------------
# Driver performance
# ---------------------------------------------------------------------------------------

# "How many wins did Hamilton have in 2019?" / career totals when no year is given.
# Hops: Result -> Driver, Result -> Race, Result -> Status.
DRIVER_STATS = """
MATCH (d:Driver {driverId: $driverId})<-[:BY_DRIVER]-(res:Result)-[:FOR_RACE]->(ra:Race)
WHERE ($year_from IS NULL OR ra.year >= $year_from)
  AND ($year_to   IS NULL OR ra.year <= $year_to)
MATCH (res)-[:HAS_STATUS]->(s:Status)
WHERE NOT s.status IN $not_started
WITH d, count(DISTINCT ra) AS races,
     sum(res.points) AS points,
     sum(CASE WHEN res.position = 1 THEN 1 ELSE 0 END) AS wins,
     sum(CASE WHEN res.position <= 3 THEN 1 ELSE 0 END) AS podiums,
     sum(CASE WHEN s.status = 'Finished' OR s.status STARTS WITH '+' THEN 1 ELSE 0 END) AS classified,
     min(ra.year) AS first_year, max(ra.year) AS last_year
RETURN d.forename + ' ' + d.surname AS driver, first_year, last_year,
       races, points, wins, podiums, classified
LIMIT 1
"""

# "How many times has Hamilton won at Silverstone before 2020?"
# Hops: Result -> Driver, Result -> Race -> Circuit, Result -> Status.
DRIVER_CIRCUIT = """
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
"""

# ---------------------------------------------------------------------------------------
# Constructor performance
# ---------------------------------------------------------------------------------------

# "How reliable were Ferrari between 2014 and 2021?" Reliability = classified / entries.
# Hops: Result -> Constructor, Result -> Race, Result -> Status.
CONSTRUCTOR_STATS = """
MATCH (c:Constructor {constructorId: $constructorId})<-[:FOR_CONSTRUCTOR]-(res:Result)-[:FOR_RACE]->(ra:Race)
WHERE ($year_from IS NULL OR ra.year >= $year_from)
  AND ($year_to   IS NULL OR ra.year <= $year_to)
MATCH (res)-[:HAS_STATUS]->(s:Status)
WHERE NOT s.status IN $not_started
WITH c, count(res) AS entries, count(DISTINCT ra) AS races,
     sum(res.points) AS points,
     sum(CASE WHEN res.position = 1 THEN 1 ELSE 0 END) AS wins,
     sum(CASE WHEN res.position <= 3 THEN 1 ELSE 0 END) AS podiums,
     sum(CASE WHEN s.status = 'Finished' OR s.status STARTS WITH '+' THEN 1 ELSE 0 END) AS classified,
     min(ra.year) AS first_year, max(ra.year) AS last_year
RETURN c.constructor_name AS constructor, first_year, last_year, races, entries,
       points, wins, podiums, classified, toFloat(classified) / entries AS reliability
LIMIT 1
"""

# ---------------------------------------------------------------------------------------
# Head-to-head
# ---------------------------------------------------------------------------------------

# "Hamilton vs Bottas": totals over all shared races (optionally only as team-mates at one
# constructor) plus the 10 most recent shared races, with the paths for the subgraph.
# Hops: Driver <- Result -> Race <- Result -> Driver, optionally both -> Constructor.
HEAD_TO_HEAD = """
MATCH path = (d1:Driver {driverId: $driverId})<-[:BY_DRIVER]-(r1:Result)-[:FOR_RACE]->(ra:Race)
             <-[:FOR_RACE]-(r2:Result)-[:BY_DRIVER]->(d2:Driver {driverId: $driver2Id})
WHERE ($year_from IS NULL OR ra.year >= $year_from)
  AND ($year_to   IS NULL OR ra.year <= $year_to)
  AND ($constructorId IS NULL OR EXISTS {
        (r1)-[:FOR_CONSTRUCTOR]->(:Constructor {constructorId: $constructorId})<-[:FOR_CONSTRUCTOR]-(r2) })
WITH d1, d2, ra, r1, r2, path
ORDER BY ra.date DESC
WITH d1, d2, count(*) AS shared_races,
     sum(CASE WHEN r1.positionOrder < r2.positionOrder THEN 1 ELSE 0 END) AS driver1_ahead,
     sum(CASE WHEN r2.positionOrder < r1.positionOrder THEN 1 ELSE 0 END) AS driver2_ahead,
     collect({race: ra.name, year: ra.year, pos1: r1.positionOrder, pos2: r2.positionOrder,
              path: path}) AS recent
UNWIND recent[..10] AS r
RETURN d1.forename + ' ' + d1.surname AS driver1, d2.forename + ' ' + d2.surname AS driver2,
       shared_races, driver1_ahead, driver2_ahead,
       r.race AS race, r.year AS year, r.pos1 AS driver1_finish, r.pos2 AS driver2_finish,
       r.path AS path
LIMIT 10
"""

# ---------------------------------------------------------------------------------------
# Race and circuit facts
# ---------------------------------------------------------------------------------------

# "Who won the 2019 British Grand Prix?" / "Who won at Silverstone in 2019?" Top 10 finishers
# of one race, given by name or by circuit.
# Hops: Circuit <- Race <- Result -> Driver, Result -> Constructor, Result -> Status.
RACE_RESULT = """
MATCH (c0:Circuit)<-[:AT_CIRCUIT]-(ra:Race {year: $year})
WHERE ($race_name IS NULL OR ra.name = $race_name)
  AND ($circuitId IS NULL OR c0.circuitId = $circuitId)
MATCH path_driver = (ra)<-[:FOR_RACE]-(res:Result)-[:BY_DRIVER]->(d:Driver)
MATCH path_constructor = (res)-[:FOR_CONSTRUCTOR]->(c:Constructor)
MATCH (res)-[:HAS_STATUS]->(s:Status)
RETURN ra.name AS race, ra.year AS year, ra.date AS date,
       res.positionOrder AS finish, d.forename + ' ' + d.surname AS driver,
       c.constructor_name AS constructor, res.grid AS grid, res.points AS points,
       s.status AS status, path_driver, path_constructor
ORDER BY ra.date ASC, res.positionOrder ASC
LIMIT 10
"""

# "Where is Monza and how many Grands Prix were held there?"
# Hops: Race -> Circuit (aggregated over all races held there).
CIRCUIT_INFO = """
MATCH (c:Circuit {circuitId: $circuitId})<-[:AT_CIRCUIT]-(ra:Race)
RETURN c.circuit_name AS circuit, c.circuit_loc AS location, c.circuit_country AS country,
       c.lat AS lat, c.lng AS lng, c.alt AS altitude_m,
       count(ra) AS races_held, min(ra.year) AS first_year, max(ra.year) AS last_year,
       collect(DISTINCT ra.name) AS grand_prix
"""

# ---------------------------------------------------------------------------------------
# Qualifying and grid
# ---------------------------------------------------------------------------------------

# "How many poles did Verstappen get in 2022?"
# Hops: Qualifying -> Driver, Qualifying -> Race.
QUALIFYING = """
MATCH (d:Driver {driverId: $driverId})<-[:BY_DRIVER]-(q:Qualifying)-[:FOR_RACE]->(ra:Race)
WHERE ($year_from IS NULL OR ra.year >= $year_from)
  AND ($year_to   IS NULL OR ra.year <= $year_to)
WITH d, count(q) AS sessions,
     sum(CASE WHEN q.position = 1 THEN 1 ELSE 0 END) AS poles,
     sum(CASE WHEN q.position <= 2 THEN 1 ELSE 0 END) AS front_rows,
     sum(CASE WHEN q.q3 IS NOT NULL THEN 1 ELSE 0 END) AS q3_appearances,
     avg(q.position) AS avg_position,
     min(ra.year) AS first_year, max(ra.year) AS last_year
RETURN d.forename + ' ' + d.surname AS driver, first_year, last_year,
       sessions, poles, front_rows, q3_appearances, avg_position
LIMIT 1
"""

# "Who gained the most places in the 2021 Italian Grand Prix?" The race is given either by
# name or by circuit. Top 5 classified finishers by grid position minus finishing position.
# Hops: Driver <- Result -> Race -> Circuit, Result -> Status.
GRID_GAINS = """
MATCH path = (d:Driver)<-[:BY_DRIVER]-(res:Result)-[:FOR_RACE]->(ra:Race)-[:AT_CIRCUIT]->(c:Circuit)
WHERE ra.year = $year
  AND ($race_name IS NULL OR ra.name = $race_name)
  AND ($circuitId IS NULL OR c.circuitId = $circuitId)
MATCH (res)-[:HAS_STATUS]->(s:Status)
WHERE res.grid > 0 AND (s.status = 'Finished' OR s.status STARTS WITH '+')
RETURN ra.name AS race, ra.year AS year, d.forename + ' ' + d.surname AS driver,
       res.grid AS grid, res.positionOrder AS finish,
       res.grid - res.positionOrder AS places_gained, path
ORDER BY places_gained DESC, driver ASC
LIMIT 5
"""

# ---------------------------------------------------------------------------------------
# Pit stops (two variants: whole race, or one driver when a driver is given)
# ---------------------------------------------------------------------------------------

# "Who had the fastest pit stop at the 2023 Monaco Grand Prix?" Five fastest stops of a race,
# given by name or by circuit.
# Hops: Driver <- PitStop -> Race -> Circuit.
PIT_FASTEST = """
MATCH path = (d:Driver)<-[:BY_DRIVER]-(ps:PitStop)-[:FOR_RACE]->(ra:Race {year: $year})-[:AT_CIRCUIT]->(c:Circuit)
WHERE ($race_name IS NULL OR ra.name = $race_name)
  AND ($circuitId IS NULL OR c.circuitId = $circuitId)
RETURN ra.name AS race, ra.year AS year, d.forename + ' ' + d.surname AS driver,
       ps.stop AS stop, ps.lap AS lap, ps.milliseconds / 1000.0 AS seconds, path
ORDER BY ps.milliseconds ASC
LIMIT 5
"""

# "How many pit stops did Hamilton make at the 2021 Hungarian Grand Prix?" All stops of one driver.
# Hops: Driver <- PitStop -> Race -> Circuit.
PIT_DRIVER = """
MATCH path = (d:Driver {driverId: $driverId})<-[:BY_DRIVER]-(ps:PitStop)-[:FOR_RACE]->(ra:Race {year: $year})
             -[:AT_CIRCUIT]->(c:Circuit)
WHERE ($race_name IS NULL OR ra.name = $race_name)
  AND ($circuitId IS NULL OR c.circuitId = $circuitId)
RETURN ra.name AS race, ra.year AS year, d.forename + ' ' + d.surname AS driver,
       ps.stop AS stop, ps.lap AS lap, ps.milliseconds / 1000.0 AS seconds, path
ORDER BY ps.stop ASC
LIMIT 10
"""

# ---------------------------------------------------------------------------------------
# Season and history
# ---------------------------------------------------------------------------------------

# "Who won the 2021 championship?" Top 10 by summed race points, plus the season's driver count.
# Hops: Result -> Race -> Season, Result -> Driver.
SEASON_STANDINGS = """
MATCH (res:Result)-[:FOR_RACE]->(:Race)-[:IN_SEASON]->(:Season {year: $year})
MATCH (res)-[:BY_DRIVER]->(d:Driver)
WITH d.forename + ' ' + d.surname AS driver, sum(res.points) AS points,
     sum(CASE WHEN res.position = 1 THEN 1 ELSE 0 END) AS wins
ORDER BY points DESC, wins DESC, driver ASC
WITH collect({driver: driver, points: points, wins: wins}) AS table, count(*) AS drivers_in_season
UNWIND range(0, size(table) - 1) AS i      // rank = position in the sorted table; LIMIT keeps the top 10
RETURN i + 1 AS rank, table[i].driver AS driver, table[i].points AS points,
       table[i].wins AS wins, drivers_in_season
LIMIT 10
"""

SCOPE_RESULTS = BASE_SCOPE + "; unfinished or disqualified entries count as starts, never-started entries are excluded."

TEMPLATES = {
    "driver_season_stats": [Template(
        "driver_stats", DRIVER_STATS, build_params, False,
        SCOPE_RESULTS + " Points are race points only, so sprint points are missing.")],
    "driver_circuit_record": [Template(
        "driver_circuit_record", DRIVER_CIRCUIT, build_params, False, SCOPE_RESULTS)],
    "constructor_stats": [Template(
        "constructor_stats", CONSTRUCTOR_STATS, build_params, False,
        SCOPE_RESULTS + " Reliability is classified finishes divided by entries (every car counts).")],
    "head_to_head": [Template(
        "head_to_head", HEAD_TO_HEAD, build_params, True,
        BASE_SCOPE + "; counts only races where both drivers have a result, "
                     "'ahead' is the better official finishing order.")],
    "race_result": [Template(
        "race_result", RACE_RESULT, build_params, True,
        BASE_SCOPE + "; shows the top 10 finishers only.")],
    "circuit_info": [Template(
        "circuit_info", CIRCUIT_INFO, build_params, False,
        "Counts World Championship races held at the circuit between 1950 and 2024.")],
    "qualifying_record": [Template(
        "qualifying_record", QUALIFYING, build_params, False,
        "Qualifying data exists only from 1994 to 2024; Q3 times exist only since the three-part format (2006); "
        "a pole is qualifying position 1.")],
    "grid_gains": [Template(
        "grid_gains", GRID_GAINS, build_params, True,
        BASE_SCOPE + "; only classified finishers who started from a grid slot, top 5 places gained.")],
    "pit_stops": [
        Template("pit_stops_driver", PIT_DRIVER, build_params, True,
                 "Pit-stop data exists only from 2011 to 2024; lists every recorded stop of the driver "
                 "(pit-lane time).", when="driver"),
        Template("pit_stops_fastest", PIT_FASTEST, build_params, True,
                 "Pit-stop data exists only from 2011 to 2024; shows the five fastest stops (pit-lane time).")],
    "season_standings": [Template(
        "season_standings", SEASON_STANDINGS, build_params, False,
        BASE_SCOPE + "; points are summed race points (no dropped-score rules, no sprint points), top 10 only.")],
}


def select_template(intent, entities):
    """First variant whose `when` slot is present (or that has none)."""
    return next(t for t in TEMPLATES[intent] if t.when is None or t.when in entities)
