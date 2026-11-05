# Milestone 2 – Evaluation Prep

Likely TA questions with short answers based on our implementation ([milestone2/](milestone2/)).
The numbers come from running the queries on our built graph.

---

## A. Knowledge graph design

**1. What is the difference between an entity node and a fact node? Why separate them?**
An entity (Driver, Constructor, Circuit, Season, Status, Race) holds attributes that don't change from race to race, such as a name, nationality or country. A fact (Result, Qualifying, PitStop) holds what happened in one specific race: grid slot, finishing position, constructor, lap times. A driver changes teams between seasons, so the constructor can't be a Driver property. It is a relationship from the fact to the Constructor.

**2. Why is the driver's constructor linked from Result/Qualifying instead of directly from Driver?**
A driver can drive for different constructors across seasons, and even within one race (shared drives). A direct Driver→Constructor edge would lose *when* that pairing happened. Linking through the fact keeps it tied to one race.

**3. Why is Race an entity and not a fact?**
A race is a single event with fixed attributes (year, round, name, date). Facts hang off it. It is a parent node: Result, Qualifying and PitStop all point *to* it.

**4. Which direction do the relationships go, and does it matter?**
Facts point to entities (`(Result)-[:BY_DRIVER]->(Driver)`), and Race points to Circuit and Season. Neo4j can traverse either way, but a query that writes the wrong direction returns **0 rows**, for example `(Driver)-[:BY_DRIVER]->(Result)`. We followed the schema exactly.

**5. Why do the expected counts come out exactly to 51,113 nodes and 163,510 relationships?**
Nodes: one per row of the 9 CSVs (861 + 212 + 77 + 75 + 139 + 1,125 + 26,759 + 10,494 + 11,371). Relationships: 2 per Race + 4 per Result + 3 per Qualifying + 2 per PitStop = 2,250 + 107,036 + 31,482 + 22,742. Every foreign key in the data resolves to a parent node, so no relationship is dropped.

**6. Why aren't lap_times, standings, constructor_results or sprint_results included?**
The milestone says so. As a result, every points total is Grand Prix only, with sprints excluded.

**7. What would you change to add sprint results?**
We would add a fact node, for example `SprintResult`, with the same four relationships as Result. We wouldn't merge sprint results into Result, because that would change what "points" means in existing queries.

## B. Create_kg.py

**8. Walk us through the script.**
- Read `config.txt`.
- Create a uniqueness constraint per label.
- Load the parents in order: Driver, Constructor, Circuit, Season, Status, Race. Race goes last among them because it links to Circuit and Season.
- Load the facts: Result, Qualifying, PitStop.
- Print the counts.

Each label is sent in batches of 1,000 rows with `UNWIND`.

**9. Why must parents exist before facts?**
A fact's relationships are created with `MATCH` on the parent, then `MERGE` on the edge. If the parent doesn't exist yet, `MATCH` finds nothing and that row's node gets no relationships. There is no error; the relationship is just missing.

**10. How do you guarantee that re-running the script doesn't create duplicates?**
- Nodes: `MERGE` on the unique identifier, backed by a uniqueness constraint.
- Relationships: `MERGE` on each edge.

We ran the script twice and the counts stayed at 51,113 / 163,510.

**11. Why `MERGE` and not `CREATE`?**
`CREATE` always adds a new node or relationship. Running the script twice would double the graph.

**12. Why `SET n = row.props` instead of `SET n += row.props`?**
`=` replaces the node's whole property map. After a re-run, the node holds exactly what the CSV has. `+=` would keep any stale properties left over from before.

**13. Why create uniqueness constraints?**
- **Correctness:** two nodes with the same id become impossible.
- **Speed:** each constraint creates an index, so the `MERGE`/`MATCH` lookups on ids are fast. Without the index, every lookup scans all nodes of that label.

We have 9 constraints, one per label.

**14. How do you handle missing values?**
`\N` and empty cells (the empty `q2`/`q3` cells in qualifying.csv) are skipped when building the property map. The property simply doesn't exist on that node. For example, 10,953 Result nodes have no `position`. In Cypher, a missing property reads as `null`.

**15. Why not store `\N` or `null` or `0`?**
- `"\N"` is a string. It would break numeric comparisons and give wrong counts.
- `0` is a real value. It would pull averages down and be wrong data.
- Neo4j can't store a null property; setting a property to null removes it.

**16. Which property types did you use and why?**
- Integers: every id except `eventId`, plus all whole numbers (year, round, grid, position, laps, …).
- Floats: `points`, `lat`, `lng`, `fastestLapSpeed`.
- Neo4j `Date`: `dob` and `Race.date`.
- Strings: everything else, including lap or qualifying times like `"1:27.452"`, which aren't plain numbers.

All of this follows the spec. Correct types matter for queries: `ra.year >= 2014` only works if `year` is an integer, and `ra.date < date('2020-01-01')` only works if `date` is a Date.

**17. What happens if `points` is stored as a string?**
`sum()` fails or gives wrong results, and sorting becomes alphabetical ("9" sorts after "10").

**18. How is the PitStop identifier built and why?**
`eventId = "raceId:driverId:stop"`, as a string. pit_stops.csv has no id column; this combination is unique (we checked there are no duplicates).

**19. How does the script read the config? Why by key?**
It parses `KEY=VALUE` lines into a dict and reads `URI`, `USERNAME` and `PASSWORD` by key. The grading `config.txt` has an extra `DATABASE` line, so reading by line number would break. We don't hardcode a database name. If `DATABASE` is present we use it; otherwise the server's default database. That last part is our assumption.

**20. Where does the script read the CSVs from?**
From its own directory, as the deliverable requires. If they aren't there (inside our repo), it falls back to `../data`.

**21. Why batch with `UNWIND`?**
One transaction per row would be very slow (tens of thousands of round trips). One transaction for everything could use too much memory. Batches of 1,000 keep it fast and safe; the full build takes about 19 s.

**22. Which columns are not stored, and why?**
- `url` and the `fp*`/`quali_*`/`sprint_*` columns: they aren't in the schema.
- Foreign keys (`raceId`, `driverId`, … on facts; `circuitId` on Race): they become relationships, not properties.

## C. The five queries

**23. Q1: what counts as a win? Why `sum(CASE …)` instead of `count`?**
A win is `position = 1`. `sum(CASE WHEN … THEN 1 ELSE 0 END)` counts the wins while keeping drivers who have 0 wins. A `WHERE position = 1` filter would remove those drivers from the result.

**24. Q1: why group by the node `d` and not by the name?**
Two different drivers could share a full name. Grouping by the node is always correct.

**25. Q2: what is a classified finish? Why `STARTS WITH '+'`?**
A classified finish is the status "Finished", or one of the 33 "+N Laps" statuses (lapped, but still classified). The `Entries >= 100` filter is a `WHERE` after the `WITH`, so it applies to the aggregated value (like SQL `HAVING`).

**26. Q2: why `toFloat(Classified) / Entries`?**
Integer divided by integer would truncate to 0. The spec asks for a 0–1 value, not a percentage.

**27. Q2: did you filter by `Race.year` or through `Season`?**
By `Race.year`. Both give the same result, since every Race has `IN_SEASON` to a Season with the same year. Q1 shows the Season traversal.

**28. Q3: how do you find races where both drove for the same constructor?**
Both Result nodes point `FOR_RACE` to the same Race, and `FOR_CONSTRUCTOR` to the same Constructor. "Ahead" means a lower `positionOrder`. Result: Mercedes, 100 shared races, Hamilton ahead 75, Bottas ahead 25.

**29. Why `positionOrder` and not `position`?**
`position` is missing for drivers who didn't finish. `positionOrder` is always present and ranks every entry, including retirements.

**30. Q4: why count `DISTINCT ra` for Starts?**
85 (driver, race) pairs in the data have more than one Result (shared drives), 6 of them at Monaco before 2020. A driver can only start a race once, so Starts counts distinct races. As the question says, the average finish and wins use all of the driver's remaining Result nodes.

**31. Q4: why exclude "Did not qualify", "Did not prequalify" and "Withdrew"?**
The driver never started the race. That removes 222 Results at Monaco before 2020.

**32. Q4: why `ra.date < date('2020-01-01')`?**
`Race.date` is a Neo4j Date, so it has to be compared with a Date, not with a string.

**33. Q5: explain the primary-constructor logic.**
- Count entries per (driver, year, constructor).
- Sort by entries descending, then constructorId ascending (the tie rule).
- `collect(...)[0]` takes the primary constructor for each driver and season.
- Keep drivers with both seasons (`size = 2`; 16 drivers).
- Compare the two constructorIds.

Result: 6 drivers. Ocon (Renault → Alpine) and Stroll (Racing Point → Aston Martin) count because the ids differ even though the teams were only renamed.

**34. Q5: why compare by constructorId and not by name?**
The spec says so. A rename is a different Constructor node with a different id.

**35. Q5: what if no driver changed?**
`RETURN count(d), collect(...)` still returns one row: `0, []`.

**36. Why must every query start with `MATCH` and use those exact column names?**
The autograder extracts queries by the `//N. Query N` / `MATCH` / `RETURN` format and reads answers by column name (`AS`).

## D. "Private query" style variations

Be ready to write these live:

**37. Points and wins for any other season, e.g. 2008.**
Change the year in Q1.

**38. Pole positions per driver in a season.**
```cypher
MATCH (q:Qualifying)-[:FOR_RACE]->(:Race {year: 2021})
WHERE q.position = 1
MATCH (q)-[:BY_DRIVER]->(d:Driver)
RETURN d.forename + ' ' + d.surname AS Driver, count(*) AS Poles
ORDER BY Poles DESC
```
Trap: for 2021 this gives Verstappen and Hamilton 8 each, but Verstappen's official count is 10. On sprint weekends qualifying.csv records the Friday session, while the official pole went to the sprint winner, and sprint results are not in our KG. Our query answers "fastest in qualifying", not "official pole".

**39. Fastest pit stop in a race.**
```cypher
MATCH (p:PitStop)-[:FOR_RACE]->(:Race {year: 2021, round: 1})
MATCH (p)-[:BY_DRIVER]->(d:Driver)
RETURN d.forename + ' ' + d.surname AS Driver, p.duration, p.milliseconds
ORDER BY p.milliseconds ASC
LIMIT 1
```
Sort by `milliseconds` (integer), not by `duration` (string).

**40. Which circuit hosted the most races?**
```cypher
MATCH (ra:Race)-[:AT_CIRCUIT]->(c:Circuit)
RETURN c.circuit_name AS Circuit, count(ra) AS Races
ORDER BY Races DESC
```

**41. Drivers who reached Q3 in a season.**
```cypher
MATCH (q:Qualifying)-[:FOR_RACE]->(:Race {year: 2021})
WHERE q.q3 IS NOT NULL
...
```
This works only because missing `q3` is omitted rather than stored as an empty string.

**42. Constructor points per season.**
```cypher
MATCH (res:Result)-[:FOR_RACE]->(:Race {year: Y})
MATCH (res)-[:FOR_CONSTRUCTOR]->(c:Constructor)
RETURN c.constructor_name AS Constructor, sum(res.points) AS Points
ORDER BY Points DESC
```

**43. Positions gained from grid to finish.**
Use `res.grid - res.position` where `res.position IS NOT NULL`. In this data, `grid = 0` means a pit-lane start; decide whether to exclude it.

**44. Drivers of a given nationality who won at a given circuit.**
A multi-hop path: Driver ← Result → Race → Circuit.

Coverage limits to know:
- Qualifying data starts in 1994 and pit stops in 2011, so questions about earlier years return nothing for those facts.
- Only 394 of 1,125 races have a `time`.

## E. Neo4j and Cypher concepts

**45. `MATCH` vs `OPTIONAL MATCH`?**
`OPTIONAL MATCH` keeps the row with nulls when the pattern isn't found, like a SQL LEFT JOIN.

**46. What does `WITH` do?**
It pipes results between query parts. It is needed to aggregate, then filter on the aggregate, then continue.

**47. How do aggregation groups form in Cypher?**
There is no `GROUP BY`. The non-aggregated expressions in `WITH`/`RETURN` are the grouping keys.

**48. How do `count(*)`, `count(x)` and `count(DISTINCT x)` differ?**
- `count(*)` counts rows.
- `count(x)` skips nulls.
- `count(DISTINCT x)` counts unique values.

**49. How do you check the counts and schema in the GUI?**
- Totals: `MATCH (n) RETURN count(n)` and `MATCH ()-[r]->() RETURN count(r)`.
- Schema picture: `CALL db.schema.visualization()`.
- Constraints: `SHOW CONSTRAINTS`.

**50. Why did Bloom show only a handful of nodes?**
Bloom draws only what you search for, up to a display limit. It never shows the whole database.

**51. Why a graph database for this rather than SQL?**
Multi-hop questions become path patterns, with no chains of joins. Examples: teammates in the same race for the same team (Q3), or driver → race → circuit. Relationships are stored directly, so traversing them is cheap. The graph is also the base for Milestone 3.
