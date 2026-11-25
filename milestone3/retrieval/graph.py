"""Neo4j access for the assistant: look-up tables for entity extraction and template retrieval.

Pipeline role: `Graph.lookup_tables()` feeds the entity extractor at start-up;
`Graph.retrieve(intent, entities)` is called once per question. It picks a Cypher template,
fills its parameters, runs it read-only and returns the rows plus, for graphical templates,
the subgraph (nodes and edges) the UI draws. An empty `rows` list means the knowledge graph
holds no data for the question; nothing else is consulted.

`retrieve` raises `ValueError` for a missing required slot; the pipeline avoids that by asking
`missing_slots(intent, entities)` first and turning the answer into a clarifying question.

Run `python -m milestone3.retrieval.graph` for a self-test with one example per intent.
"""
from dataclasses import dataclass

from neo4j import GraphDatabase
from neo4j.graph import Node, Path, Relationship

from milestone3 import config
from milestone3.retrieval.templates import REQUIRED, TEMPLATES, select_template


@dataclass
class Retrieval:
    template: str
    cypher: str
    params: dict
    rows: list[dict]
    subgraph: dict | None   # {"nodes": [{"id","label","name"}], "edges": [{"source","rel","target"}]}
    scope: str


def missing_slots(intent, entities):
    """Required slots absent from `entities`; alternatives are written as 'race_name or circuit'."""
    return [" or ".join(options) for options in REQUIRED.get(intent, [])
            if not any(slot in entities for slot in options)]


def node_name(node):
    """Human-readable caption of a node, chosen by its label."""
    p = dict(node)
    label = next(iter(node.labels))
    return {
        "Driver": lambda: f"{p['forename']} {p['surname']}",
        "Constructor": lambda: p["constructor_name"],
        "Circuit": lambda: p["circuit_name"],
        "Season": lambda: str(p["year"]),
        "Status": lambda: p["status"],
        "Race": lambda: f"{p['name']} {p['year']}",
        "Result": lambda: f"Result {p['resultId']} (P{p['positionOrder']})",
        "Qualifying": lambda: f"Qualifying {p['qualifyId']} (P{p.get('position', '-')})",
        "PitStop": lambda: f"Pit stop {p['stop']} (lap {p['lap']})",
    }[label]()


def parts(item):
    """(nodes, relationships) contained in a Node, Relationship or Path."""
    if isinstance(item, Path):
        return item.nodes, item.relationships
    if isinstance(item, Relationship):
        return [item.start_node, item.end_node], [item]
    return [item], []


def build_subgraph(records):
    """Collect every node, relationship and path found in the records into the UI format.
    Returns (rows without the graph columns, subgraph)."""
    nodes, edges, rows = {}, {}, []
    for record in records:
        row = {}
        for key, value in record.items():
            if not isinstance(value, (Node, Relationship, Path)):
                row[key] = value
                continue
            found_nodes, found_rels = parts(value)
            for n in found_nodes:
                nodes[n.element_id] = {"id": n.element_id, "label": next(iter(n.labels)),
                                       "name": node_name(n)}
            for r in found_rels:
                edge = {"source": r.start_node.element_id, "rel": r.type, "target": r.end_node.element_id}
                edges[tuple(edge.values())] = edge
        rows.append(row)
    return rows, {"nodes": list(nodes.values()), "edges": list(edges.values())}


def plain(value):
    """Neo4j dates become ISO strings so rows are JSON-friendly; everything else is kept."""
    return value.iso_format() if hasattr(value, "iso_format") else value


class Graph:
    """Read-only connection to the Milestone 2 Neo4j graph."""

    def __init__(self):
        settings = config.neo4j_settings()
        self.driver = GraphDatabase.driver(settings["URI"], auth=(settings["USERNAME"], settings["PASSWORD"]))
        self.database = settings.get("DATABASE") or None

    def run(self, cypher, params):
        """Execute a read-only query and return one dict per record."""
        records = self.driver.execute_query(cypher, parameters_=params, database_=self.database,
                                            routing_="r").records
        return [{k: plain(v) for k, v in r.items()} for r in records]

    def lookup_tables(self):
        """Names and ids the entity extractor matches user text against (loaded once)."""
        drivers = self.run("""
            MATCH (d:Driver)
            OPTIONAL MATCH (d)<-[:BY_DRIVER]-(res:Result)-[:FOR_RACE]->(ra:Race)
            RETURN d.driverId AS driverId, d.forename AS forename, d.surname AS surname,
                   d.code AS code, d.driverRef AS driverRef,
                   coalesce(min(ra.year), 0) AS first_year, coalesce(max(ra.year), 0) AS last_year,
                   count(res) AS entries
            """, {})
        constructors = self.run("""
            MATCH (c:Constructor)
            RETURN c.constructorId AS constructorId, c.constructor_name AS name,
                   c.constructorRef AS constructorRef
            ORDER BY constructorId
            """, {})
        circuits = self.run("""
            MATCH (c:Circuit)
            RETURN c.circuitId AS circuitId, c.circuit_name AS name, c.circuit_loc AS location,
                   c.circuit_country AS country, c.circuitRef AS circuitRef
            ORDER BY circuitId
            """, {})
        races = self.run("MATCH (ra:Race) RETURN DISTINCT ra.name AS name ORDER BY name", {})
        return {"drivers": drivers, "constructors": constructors, "circuits": circuits,
                "races": [r["name"] for r in races]}

    def retrieve(self, intent, entities):
        """Run the template of `intent` with the slots in `entities`; returns a `Retrieval`."""
        if intent not in TEMPLATES:
            raise ValueError(f"No retrieval template for intent '{intent}'")
        missing = missing_slots(intent, entities)
        if missing:
            raise ValueError("Missing slot: " + ", ".join(missing))
        template = select_template(intent, entities)
        params = template.build_params(entities)
        records = self.run(template.cypher, params)
        rows, subgraph = build_subgraph(records)
        return Retrieval(template.name, template.cypher, params, rows,
                         subgraph if template.graphical and rows else None, template.scope)

    def close(self):
        self.driver.close()


# ---------------------------------------------------------------------------------------
# Self-test: one example per intent
# ---------------------------------------------------------------------------------------
EXAMPLES = {
    "driver_season_stats": {"driver": {"driverId": 1, "name": "Lewis Hamilton"}, "year": 2019},
    "driver_circuit_record": {"driver": {"driverId": 1, "name": "Lewis Hamilton"},
                              "circuit": {"circuitId": 9, "name": "Silverstone Circuit"}, "year_to": 2019},
    "constructor_stats": {"constructor": {"constructorId": 6, "name": "Ferrari"},
                          "year_from": 2014, "year_to": 2021},
    "head_to_head": {"driver": {"driverId": 1, "name": "Lewis Hamilton"},
                     "driver2": {"driverId": 822, "name": "Valtteri Bottas"}},
    "race_result": {"race_name": "British Grand Prix", "year": 2019},
    "circuit_info": {"circuit": {"circuitId": 14, "name": "Autodromo Nazionale di Monza"}},
    "qualifying_record": {"driver": {"driverId": 830, "name": "Max Verstappen"}, "year": 2022},
    "grid_gains": {"race_name": "Italian Grand Prix", "year": 2021},
    "pit_stops": {"race_name": "Monaco Grand Prix", "year": 2023},
    "season_standings": {"year": 2021},
}


def main():
    graph = Graph()
    for intent, entities in EXAMPLES.items():
        result = graph.retrieve(intent, entities)
        graph_size = result.subgraph and (len(result.subgraph["nodes"]), len(result.subgraph["edges"]))
        print(f"{intent:22} {result.template:22} rows={len(result.rows):2} subgraph={graph_size}")
        print("   ", result.rows[0] if result.rows else None)

    # The spec's example: Hamilton's wins at Silverstone before 2020.
    row = graph.retrieve("driver_circuit_record", EXAMPLES["driver_circuit_record"]).rows[0]
    print("Hamilton at Silverstone before 2020: wins =", row["wins"])
    graph.close()


if __name__ == "__main__":
    main()
