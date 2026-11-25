"""Rule-based intent classifier: ordered keyword / regex patterns, first match wins.

Pipeline role: approach 1 of the intent classification step (the embedding classifier is
approach 2). It needs no model and no entity list, so it is instant and explainable, but
it only knows the wordings listed below. Order matters: the specific intents come first
and the broad "driver stats" intent is the last fallback. Returns None when nothing matches.
"""
import re

# A few well-known circuits so "at Monza" can be told apart from "at Ferrari".
_TRACKS = (r"silverstone|monza|spa|monaco|suzuka|interlagos|imola|hungaroring|zandvoort|baku|"
           r"albert park|red bull ring|austin|sepang|catalunya|barcelona|bahrain|jeddah|"
           r"singapore|marina bay|yas marina|abu dhabi|shanghai|hockenheim|nurburgring|"
           r"montreal|mexico city|las vegas|miami|sochi|istanbul|magny.?cours|paul ricard|"
           r"circuit|track|speedway|autodromo|ring")

# Teams people actually name in questions (the entity extractor knows all of them).
_TEAMS = (r"ferrari|mercedes|red ?bull|mclaren|williams|renault|alpine|haas|sauber|alfa romeo|"
          r"aston martin|alphatauri|alpha tauri|toro rosso|force india|racing point|lotus|"
          r"brawn|benetton|jordan|tyrrell|brabham|cooper|merc|constructors?")

# (intent, patterns) in priority order. Each pattern is searched in the lower-cased text.
RULES = [
    # Things the graph cannot answer: refuse before any F1 keyword pulls the question in.
    ("out_of_scope", [
        r"weather|\brain\b|forecast for the (race|weekend)|temperature",
        r"salary|how much.*earn|net worth|contract value|sponsor|how rich|prize money",
        r"\btyres?\b|\btires?\b|compound|telemetry|lap[- ]by[- ]lap|sector time|top speed|horsepower|"
        r"\bhp\b|engine (spec|power)|downforce|car (spec|weight)|\bweight of\b|aerodynamic",
        r"\bnews\b|rumou?rs?|latest|trending|gossip|transfer market|regulations?",
        r"best driver|greatest|\bgoat\b|favou?rite|in your opinion|do you think|who do you like|"
        r"overrated|underrated",
        r"capital of|football|soccer|basketball|nba|\bnfl\b|bitcoin|stock market|recipe|"
        r"tell me a joke|\bjoke\b|your name|who are you|hello there|^(hi|hello|hey)\b",
    ]),
    # Asking for a model prediction.
    ("podium_prediction", [
        r"predict|forecast|estimate|\bodds\b|chances?\b|how likely|likelihood|probability|\bmodel\b",
        r"\blikely to\b|\b(reach|reaches|make|makes)\b.*\bpodium",
        r"\bwill\b.*\b(podium|top (three|3)|finish on the podium)",
        r"\b(going|expected) to (be on the podium|podium|get a podium|make the top)",
    ]),
    # Two drivers compared: paraphrases of "who finished ahead more often".
    ("head_to_head", [
        r"\bvs\b|\bvs\.|versus|\bagainst\b|head[- ]to[- ]head|h2h",
        r"\bbeat\b|\bbeats\b|finished? (ahead|in front|higher|before)|ahead of|in front of|came home",
        r"faster teammate|outperform|out-?finish|better teammate|\bcompare\b.*\b(and|with|to)\b",
    ]),
    # Pit stops.
    ("pit_stops", [
        r"pit[- ]?stops?|\bpits?\b|pitted|pitting|quickest stop|fastest stop|\bstops\b",
    ]),
    # Places gained between grid slot and finish (before qualifying, which also says "grid").
    ("grid_gains", [
        r"gain(ed|s|ers?)?\b.*\b(places?|positions?|spots?)|places? (gained|made up)|\bgainers\b",
        r"made up|overtook|overtak|climbed|moved up|improved the most|biggest mover|"
        r"from (the |his |their )?(grid|starting)",
        r"grid (to|and) (the )?finish|jump from",
    ]),
    # Qualifying, poles, Q1-Q3.
    ("qualifying_record", [
        r"\bpoles?\b|pole position|\bq[123]\b|qualif|front row|starting position|grid position|"
        r"started from",
    ]),
    # Championship table of a season.
    ("season_standings", [
        r"standings?|championship (table|leader|winner|points)|points leader|points table|most points|"
        r"world champion|runner[- ]up|top (10|ten)",
        r"who (led|won|lost|took) the (\d{4} )?(formula (1|one) )?(world |drivers\'? |constructors\'? )?(championship|title)",
        r"how many (different )?drivers",
    ]),
    # Winner / podium of one specific Grand Prix.
    ("race_result", [
        r"who (won|came first|finished (first|second|third)|took the win|was on the podium|"
        r"was victorious|claimed (the )?(win|victory))",
        r"winner of|podium (of|at|in) the (\d{4}|[a-z ]*(grand prix|gp))|result of the|results of the|"
        r"top three (at|of|in)|(second|third|first) (at|in) the",
        r"(grand prix|\bgp\b).*\b(winner|podium|result|won|victor\w*)|"
        r"(winner|podium|result|won|victor\w*).*(grand prix|\bgp\b)",
    ]),
    # Facts about a circuit.
    ("circuit_info", [
        r"where is\b|located|which (country|city)|in which (country|city)|what (country|city|year)",
        r"(races?|grands? prix|events?) (have been |has been |were |was )?(held|hosted|took place|taken place)",
        r"(first|last) (race|grand prix)\b|(first|last) (year|time)\b.*\b(held|hosted|host)|\bhost(s|ed)?\b|hold a grand prix|"
        r"since when|facts on|tell me about the .*(circuit|track)|information about the",
    ]),
    # One driver at one circuit ("at Silverstone").
    ("driver_circuit_record", [
        rf"\b(at|around|on|in)\b (the )?({_TRACKS})\b",
    ]),
    # Team results and reliability.
    ("constructor_stats", [
        rf"reliab|dependab|\b({_TEAMS})\b|\bteams?\b|dnf rate|retire[ds]? from|\bretire\b|retirements?",
    ]),
    # Broad fallback: any counting question about a driver.
    ("driver_season_stats", [
        r"\bwins?\b|\bwon\b|victor(y|ies)|podiums?|\bpoints?\b|career|season|average (finish|position)|"
        r"results? (in|for|of)|how many (races|titles|championships|fastest laps)|stats",
    ]),
]

_COMPILED = [(intent, [re.compile(p) for p in patterns]) for intent, patterns in RULES]


def classify_rule(text: str) -> str | None:
    """Intent name of the first rule that matches, or None if no pattern matches."""
    text = text.lower().strip()
    for intent, patterns in _COMPILED:
        if any(p.search(text) for p in patterns):
            return intent
    return None
