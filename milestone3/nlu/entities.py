"""Entity extraction: finds drivers, constructors, circuits, races and years in a message.

Pipeline role: runs after intent classification and fills the slots of the Cypher
template. Names come from the lookup tables loaded from the graph at start-up.
Matching is accent- and case-insensitive on whole words; when two names overlap the
longest one wins ("Red Bull Ring" is a circuit, not the constructor Red Bull).
It returns only what *this* message names; the session merges it with its cache.
"""
import re
import unicodedata
from dataclasses import dataclass, field

# Common English words that are also surnames, teams or places: never matched alone.
STOP_WORDS = {
    "will", "may", "about", "race", "grand", "max", "most", "best", "win", "won", "who", "the",
    "ball", "bell", "fry", "hall", "hart", "love", "pace", "rose", "said", "ward", "ruby", "ide",
    "pic", "pon", "rol", "palm", "owen", "jean", "lang", "boyd", "grim", "gaze", "lees", "reed",
    "rees", "miles", "march", "life", "spirit", "token", "eagle", "wolf", "martini", "moore",
    "turner", "adams", "watson", "phillips", "rae", "team", "points", "lap", "gp", "driver", "drivers",
}

# Words dropped from official names to get the name people actually say.
GENERIC_NAME_WORDS = {"circuit", "international", "autodromo", "grand", "prix", "street", "de", "di",
                      "del", "do", "da", "motor", "speedway", "raceway", "f1", "team"}

# Extra names for constructors, keyed by the exact name in the graph.
CONSTRUCTOR_ALIASES = {
    "Red Bull": ["red bull racing", "rbr"],
    "Mercedes": ["mercedes amg", "mercedes benz", "mercedes amg petronas", "merc"],
    "Ferrari": ["scuderia ferrari"],
    "McLaren": ["mclaren racing", "maclaren"],
    "AlphaTauri": ["alpha tauri"],
    "RB F1 Team": ["rb", "racing bulls"],
    "Toro Rosso": ["str"],
    "Aston Martin": ["aston"],
    "Alfa Romeo": ["alfa"],
    "Force India": ["sahara force india"],
}

# Extra names for circuits, keyed by circuitRef.
CIRCUIT_ALIASES = {"americas": ["cota"], "interlagos": ["interlagos"]}

# Extra spellings for Grand Prix names, keyed by the exact Race.name.
RACE_ALIASES = {"United States Grand Prix": ["us grand prix", "us gp", "usgp", "american grand prix"]}

YEAR = r"(?<!\d)(19[5-9]\d|20[0-3]\d)(?!\d)"
# (pattern, slots) in priority order; the first pattern that matches decides the year slots.
YEAR_RULES = [
    (rf"{YEAR}\s*(?:-|–|—|to|until|through|and|&)\s*{YEAR}", lambda a, b: {"year_from": min(a, b), "year_to": max(a, b)}),
    (rf"(?:before|prior to|earlier than)\s+{YEAR}", lambda y: {"year_to": y - 1}),
    (rf"after\s+{YEAR}", lambda y: {"year_from": y + 1}),
    (rf"(?:since|from)\s+{YEAR}", lambda y: {"year_from": y}),
    (rf"(?:until|up to|through)\s+{YEAR}", lambda y: {"year_to": y}),
]

SLOTS_FOR_DRIVERS = ["driver", "driver2"]
KIND_PRIORITY = {"race": 0, "circuit": 1, "constructor": 2, "driver": 3}   # tie-break on equal length
MAX_PHRASE_WORDS = 6


@dataclass
class Extraction:
    entities: dict = field(default_factory=dict)    # slots found in this message
    ambiguous: dict = field(default_factory=dict)   # slot -> candidate display names


def fold(text: str) -> str:
    """Strip accents and punctuation, keep case: 'Pérez, Max-Verstappen' -> 'Perez Max Verstappen'."""
    plain = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return re.sub(r"[^A-Za-z0-9]+", " ", plain).strip()


def full_name(driver: dict) -> str:
    """"Lewis Hamilton" from a driver look-up record."""
    return f'{driver["forename"]} {driver["surname"]}'


def extract_years(text: str) -> dict:
    """Year slots: a range, an open-ended bound, or a single season."""
    text = text.lower()
    for pattern, build in YEAR_RULES:
        match = re.search(pattern, text)
        if match:
            return build(*map(int, match.groups()))
    years = sorted({int(y) for y in re.findall(YEAR, text)})
    if len(years) == 1:
        return {"year": years[0]}
    if len(years) > 1:   # several bare years ("2019, 2021"): treat as the span between them
        return {"year_from": years[0], "year_to": years[-1]}
    return {}


class EntityExtractor:
    """Matches look-up-table names in free text (built once from `Graph.lookup_tables()`)."""

    def __init__(self, tables: dict):
        # phrase (tuple of lower-case words) -> list of (kind, record); filled per entity type.
        self.phrases: dict[tuple, list] = {}
        codes: dict[str, list] = {}
        for d in tables["drivers"]:
            self._add("driver", full_name(d), d)
            self._add("driver", d["surname"], d, single_word_min=3)
            if d["code"]:
                codes.setdefault(d["code"], []).append(d)
        # Driver codes are optional shortcuts; a code shared by two drivers is not usable.
        self.codes = {code: ds[0] for code, ds in codes.items() if len(ds) == 1}
        for c in tables["constructors"]:
            # Official name, name without "F1 Team", its compact form ("redbull") ...
            short = " ".join(w for w in fold(c["name"]).lower().split() if w not in {"f1", "team"})
            for name in {c["name"], short, short.replace(" ", "")}:
                self._add("constructor", name, c, single_word_min=4)   # short codes like "bar" are too risky
            # ... and the hand-written aliases, which may be short ("rb", "merc").
            for alias in CONSTRUCTOR_ALIASES.get(c["name"], []):
                self._add("constructor", alias, c, single_word_min=2)
        for c in tables["circuits"]:
            words = [w for w in fold(c["name"]).lower().split() if w not in GENERIC_NAME_WORDS]
            for name in {c["name"], " ".join(words), c["location"], *CIRCUIT_ALIASES.get(c["circuitRef"], [])}:
                self._add("circuit", name, c, single_word_min=3)
        # A country names a circuit only when the country has exactly one circuit.
        by_country: dict[str, list] = {}
        for c in tables["circuits"]:
            by_country.setdefault(c["country"], []).append(c)
        for country, circuits in by_country.items():
            if len(circuits) == 1:
                self._add("circuit", country, circuits[0], single_word_min=4)
        for race in tables["races"]:
            names = [race] + RACE_ALIASES.get(race, [])
            if race.endswith(" Grand Prix"):   # "British Grand Prix" -> "British GP"
                names.append(race.replace(" Grand Prix", " GP"))
            for name in names:
                self._add("race", name, race, single_word_min=3)

    def _add(self, kind: str, name: str, record, single_word_min: int = 3):
        """Register one spelling of an entity (skipped if it is a stop word or too short)."""
        key = tuple(fold(name).lower().split())
        if not key or (len(key) == 1 and (len(key[0]) < single_word_min or key[0] in STOP_WORDS)):
            return
        entries = self.phrases.setdefault(key, [])
        for entry_kind, records in entries:   # same spelling, same kind: collect candidates
            if entry_kind == kind:
                if record not in records:
                    records.append(record)
                return
        entries.append((kind, [record]))

    def _find_spans(self, words: list[str]) -> list[tuple]:
        """All (start, end, kind, records) phrase matches, then keep the non-overlapping best."""
        lower = [w.lower() for w in words]
        found = []
        for i in range(len(words)):
            for n in range(1, MAX_PHRASE_WORDS + 1):
                for kind, records in self.phrases.get(tuple(lower[i:i + n]), []):
                    found.append((i, i + n, kind, records))
            # An upper-case three-letter word can be a driver code ("HAM").
            if words[i] in self.codes and words[i].lower() not in STOP_WORDS:
                found.append((i, i + 1, "driver", [self.codes[words[i]]]))
        # Longest phrase first, so "Red Bull Ring" beats "Red Bull"; ties by kind priority.
        found.sort(key=lambda s: (s[0] - s[1], KIND_PRIORITY[s[2]], s[0]))
        taken, kept = set(), []
        for start, end, kind, records in found:
            if taken.isdisjoint(range(start, end)):
                taken.update(range(start, end))
                kept.append((start, end, kind, records))
        return sorted(kept)

    @staticmethod
    def _active(drivers: list[dict], years: dict) -> list[dict]:
        """Drivers who raced in the years named in the message (all, if none did).
        Assumption: a question about a season concerns a driver active in that season."""
        low = years.get("year", years.get("year_from", 0))
        high = years.get("year", years.get("year_to", 9999))
        active = [d for d in drivers if d["first_year"] <= high and d["last_year"] >= low]
        return active or drivers

    def extract(self, text: str) -> Extraction:
        """Entity slots named in `text`; names shared by several records go to `ambiguous`."""
        words = fold(text).split()
        years = extract_years(text)
        result = Extraction(entities=dict(years))
        seen_drivers, driver_count = set(), 0
        for _, _, kind, records in self._find_spans(words):
            if kind == "race":
                result.entities.setdefault("race_name", records[0])   # first mention wins
            elif kind in ("constructor", "circuit"):
                if len(records) == 1:
                    result.entities.setdefault(kind, {f"{kind}Id": int(records[0][f"{kind}Id"]), "name": records[0]["name"]})
                else:
                    result.ambiguous[kind] = [r["name"] for r in records]
            else:
                if any(d["driverId"] in seen_drivers for d in records):
                    continue   # "Michael Schumacher ... Schumacher": same driver again
                if driver_count == len(SLOTS_FOR_DRIVERS):
                    continue
                slot = SLOTS_FOR_DRIVERS[driver_count]
                driver_count += 1
                candidates = self._active(records, years) if len(records) > 1 else records
                if len(candidates) == 1:
                    d = candidates[0]
                    seen_drivers.add(d["driverId"])
                    result.entities[slot] = {"driverId": int(d["driverId"]), "name": full_name(d)}
                else:
                    result.ambiguous[slot] = [full_name(d) for d in candidates]
        return result


if __name__ == "__main__":
    # Demo: build the lookup tables from the Milestone 2 CSVs and try a few questions.
    import pandas as pd
    from milestone3 import config

    def read(name):
        return pd.read_csv(config.REPO_DIR / "milestone2" / f"{name}.csv", na_values=r"\N", keep_default_na=False)

    results = read("results").merge(read("races")[["raceId", "year"]], on="raceId")
    active = results.groupby("driverId")["year"].agg(first_year="min", last_year="max")
    drivers = read("drivers").join(active, on="driverId").dropna(subset=["first_year"])
    tables = {
        "drivers": drivers.to_dict("records"),
        "constructors": read("constructors").to_dict("records"),
        "circuits": read("circuits").to_dict("records"),
        "races": sorted(read("races")["name"].unique()),
    }
    extractor = EntityExtractor(tables)
    for question in [
        "How many wins did Hamilton have at Silverstone before 2020?",
        "How reliable was Williams between 2014 and 2021?",
        "Who finished ahead more often, Norris or Ricciardo at McLaren?",
        "Who won the 2019 German Grand Prix?",
        "What about Schumacher in 2001?",
        "Michael Schumacher vs Hulkenberg since 2015",
        "Perez at the Abu Dhabi GP 2021",
        "redbull points from 2010 to 2012",
        "Will Max Verstappen podium at Monza?",
        "VER vs HAM at COTA",
    ]:
        out = extractor.extract(question)
        print(question, "->", out.entities, out.ambiguous or "")
