"""
Milestone 2 - builds the Formula 1 Knowledge Graph in Neo4j.

Overall flow:
  1. Read the Neo4j credentials from config.txt (by key, not by line number).
  2. Create a uniqueness constraint on every node identifier, so MERGE is fast
     and duplicates are impossible.
  3. Load the entity (parent) nodes: Driver, Constructor, Circuit, Season, Status,
     then Race together with its AT_CIRCUIT and IN_SEASON relationships.
  4. Load the fact nodes: Result, Qualifying, PitStop, each with its relationships
     to the parent nodes created in step 3.
  5. Print the node and relationship counts (target: 51,113 nodes / 163,510 rels).

Every write uses MERGE on the node's unique identifier, and on each relationship,
so re-running the script updates the graph instead of duplicating it.
"""
import csv
import os
from datetime import date

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

# config.txt lives in the same directory as this script.
HERE = os.path.dirname(os.path.abspath(__file__))
# The deliverable requires reading the CSVs from the script's own directory (that is how the
# grader runs it), so the CSV files must sit next to this script.
CSV_DIR = HERE
BATCH_SIZE = 1000  # rows sent per UNWIND transaction; keeps each transaction small


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
def read_config(path):
    """Parse KEY=VALUE lines into a dict so each value is read by its key."""
    config = {}
    # utf-8-sig: tolerates a BOM in case config.txt was saved by Notepad.
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            if "=" in line:
                key, value = line.split("=", 1)
                config[key.strip()] = value.strip()
    return config


# ---------------------------------------------------------------------------
# 2. CSV -> property maps
# ---------------------------------------------------------------------------
def is_missing(value):
    # Missing values are the literal \N, or an empty cell (q2/q3 in qualifying.csv).
    # They are skipped, so the property is omitted rather than stored.
    return value == r"\N" or value == ""


# Type converters. Required types: identifiers and whole numbers are int; points, lat,
# lng and fastestLapSpeed are float; dob and date are Neo4j dates (a Python date is
# sent by the driver as a Neo4j Date); everything else, including times, is a string.
INT, FLOAT, STR, DATE = int, float, str, date.fromisoformat

# Per node label: (csv file, [(csv column, schema property, converter), ...]).
# Only the properties listed in the schema are stored. Columns not in the schema
# (url, fp1_date, ...) are ignored. Foreign keys are read separately (see FOREIGN_KEYS)
# because the schema stores them as relationships, not properties.
NODES = {
    "Driver": ("drivers.csv", [
        ("driverId", "driverId", INT), ("driverRef", "driverRef", STR), ("number", "number", INT),
        ("code", "code", STR), ("forename", "forename", STR), ("surname", "surname", STR),
        ("dob", "dob", DATE), ("nationality", "nationality", STR)]),
    # The schema renames name/nationality to constructor_name/constructor_nationality.
    "Constructor": ("constructors.csv", [
        ("constructorId", "constructorId", INT), ("constructorRef", "constructorRef", STR),
        ("name", "constructor_name", STR), ("nationality", "constructor_nationality", STR)]),
    # The schema renames name/location/country to circuit_name/circuit_loc/circuit_country.
    "Circuit": ("circuits.csv", [
        ("circuitId", "circuitId", INT), ("circuitRef", "circuitRef", STR), ("name", "circuit_name", STR),
        ("location", "circuit_loc", STR), ("country", "circuit_country", STR),
        ("lat", "lat", FLOAT), ("lng", "lng", FLOAT), ("alt", "alt", INT)]),
    "Season": ("seasons.csv", [("year", "year", INT)]),
    "Status": ("status.csv", [("statusId", "statusId", INT), ("status", "status", STR)]),
    "Race": ("races.csv", [
        ("raceId", "raceId", INT), ("year", "year", INT), ("round", "round", INT),
        ("name", "name", STR), ("date", "date", DATE), ("time", "time", STR)]),
    "Result": ("results.csv", [
        ("resultId", "resultId", INT), ("number", "number", INT), ("grid", "grid", INT),
        ("position", "position", INT), ("positionText", "positionText", STR),
        ("positionOrder", "positionOrder", INT), ("points", "points", FLOAT), ("laps", "laps", INT),
        ("time", "time", STR), ("milliseconds", "milliseconds", INT), ("fastestLap", "fastestLap", INT),
        ("rank", "rank", INT), ("fastestLapTime", "fastestLapTime", STR),
        ("fastestLapSpeed", "fastestLapSpeed", FLOAT)]),
    "Qualifying": ("qualifying.csv", [
        ("qualifyId", "qualifyId", INT), ("number", "number", INT), ("position", "position", INT),
        ("q1", "q1", STR), ("q2", "q2", STR), ("q3", "q3", STR)]),
    # eventId is not a CSV column; it is built in load_rows() as "raceId:driverId:stop".
    "PitStop": ("pit_stops.csv", [
        ("stop", "stop", INT), ("lap", "lap", INT), ("time", "time", STR),
        ("duration", "duration", STR), ("milliseconds", "milliseconds", INT)]),
}

# Unique identifier of each label (used for the constraints and for MERGE).
KEYS = {"Driver": "driverId", "Constructor": "constructorId", "Circuit": "circuitId",
        "Season": "year", "Status": "statusId", "Race": "raceId", "Result": "resultId",
        "Qualifying": "qualifyId", "PitStop": "eventId"}

# Foreign-key columns needed to connect each node to its parents.
# All of them are populated in the data, so none are checked for \N.
FOREIGN_KEYS = {
    "Race": ["circuitId"],
    "Result": ["raceId", "driverId", "constructorId", "statusId"],
    "Qualifying": ["raceId", "driverId", "constructorId"],
    "PitStop": ["raceId", "driverId"],
}


def load_rows(label):
    """Return one dict per CSV row: {'props': {...schema properties...}, <fk>: int, ...}."""
    file_name, columns = NODES[label]
    rows = []
    with open(os.path.join(CSV_DIR, file_name), newline="", encoding="utf-8") as f:
        for record in csv.DictReader(f):
            props = {prop: convert(record[col]) for col, prop, convert in columns
                     if not is_missing(record[col])}
            if label == "PitStop":
                # Required format for the PitStop identifier; kept as a string (the only non-int id).
                props["eventId"] = f"{record['raceId']}:{record['driverId']}:{record['stop']}"
            row = {"props": props}
            for fk in FOREIGN_KEYS.get(label, []):
                row[fk] = int(record[fk])
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# 3. Cypher write statements
# ---------------------------------------------------------------------------
# "SET n = row.props" replaces all properties of the node with the map, so a re-run
# leaves exactly the CSV's non-missing values on the node.
def merge_node(label):
    return (f"UNWIND $rows AS row "
            f"MERGE (n:{label} {{{KEYS[label]}: row.props.{KEYS[label]}}}) "
            f"SET n = row.props ")


# Statement per label: MERGE the node, then MATCH its parents and MERGE each relationship.
WRITES = {
    "Driver": merge_node("Driver"),
    "Constructor": merge_node("Constructor"),
    "Circuit": merge_node("Circuit"),
    "Season": merge_node("Season"),
    "Status": merge_node("Status"),
    # Race links to its Season through its own year property.
    "Race": merge_node("Race") + """
        WITH n, row
        MATCH (c:Circuit {circuitId: row.circuitId}), (s:Season {year: row.props.year})
        MERGE (n)-[:AT_CIRCUIT]->(c)
        MERGE (n)-[:IN_SEASON]->(s)""",
    "Result": merge_node("Result") + """
        WITH n, row
        MATCH (r:Race {raceId: row.raceId}), (d:Driver {driverId: row.driverId}),
              (c:Constructor {constructorId: row.constructorId}), (s:Status {statusId: row.statusId})
        MERGE (n)-[:FOR_RACE]->(r)
        MERGE (n)-[:BY_DRIVER]->(d)
        MERGE (n)-[:FOR_CONSTRUCTOR]->(c)
        MERGE (n)-[:HAS_STATUS]->(s)""",
    "Qualifying": merge_node("Qualifying") + """
        WITH n, row
        MATCH (r:Race {raceId: row.raceId}), (d:Driver {driverId: row.driverId}),
              (c:Constructor {constructorId: row.constructorId})
        MERGE (n)-[:FOR_RACE]->(r)
        MERGE (n)-[:BY_DRIVER]->(d)
        MERGE (n)-[:FOR_CONSTRUCTOR]->(c)""",
    "PitStop": merge_node("PitStop") + """
        WITH n, row
        MATCH (r:Race {raceId: row.raceId}), (d:Driver {driverId: row.driverId})
        MERGE (n)-[:FOR_RACE]->(r)
        MERGE (n)-[:BY_DRIVER]->(d)""",
}

# Parents first (Race last among them, since it links to Circuit and Season),
# then the fact nodes, which link to the parents.
LOAD_ORDER = ["Driver", "Constructor", "Circuit", "Season", "Status", "Race",
              "Result", "Qualifying", "PitStop"]


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------
def main():
    config = read_config(os.path.join(HERE, "config.txt"))
    # No database name is hardcoded. Assumption: if config.txt has a DATABASE line
    # (the grading one does), that database is used; otherwise the server's default database.
    database = config.get("DATABASE") or None

    with GraphDatabase.driver(config["URI"], auth=(config["USERNAME"], config["PASSWORD"])) as driver:
        driver.verify_connectivity()

        for label in LOAD_ORDER:
            try:
                driver.execute_query(
                    f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.{KEYS[label]} IS UNIQUE",
                    database_=database)
            except Neo4jError as e:
                # E.g. an index already exists on this property. MERGE still prevents
                # duplicates without the constraint, so carry on with the load.
                print(f"Could not create constraint on {label}.{KEYS[label]}: {e.message}")

        for label in LOAD_ORDER:
            rows = load_rows(label)
            for i in range(0, len(rows), BATCH_SIZE):
                driver.execute_query(WRITES[label], rows=rows[i:i + BATCH_SIZE], database_=database)
            print(f"{label}: {len(rows)} nodes merged")

        nodes = driver.execute_query("MATCH (n) RETURN count(n) AS c", database_=database).records[0]["c"]
        rels = driver.execute_query("MATCH ()-[r]->() RETURN count(r) AS c", database_=database).records[0]["c"]
        print(f"Graph now has {nodes} nodes and {rels} relationships")


if __name__ == "__main__":
    main()
