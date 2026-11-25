"""The 12 question types (intents) the assistant understands.

Each intent has a short description, the entity slots its Cypher template needs, and
8-12 example phrasings. The embedding classifier embeds these examples once and
matches new questions to the nearest intent; the rule classifier and the evaluation
use the same names. Which slots a template really needs (including "either of two") is
decided by `retrieval/templates.py:REQUIRED`.
"""
from dataclasses import dataclass


@dataclass
class Intent:
    name: str
    description: str
    examples: list[str]
    required: list[str]   # slots the template cannot run without
    optional: list[str]   # slots that refine the answer when present


_YEARS = ["year", "year_from", "year_to"]

_INTENT_LIST = [
    Intent(
        "driver_season_stats",
        "Wins, podiums, points or average finish of one driver in a season, a period or his career.",
        [
            "How many wins did Hamilton have in 2019?",
            "How many podiums did Leclerc get in 2022?",
            "How many points did Verstappen score in 2021?",
            "What are Fernando Alonso's career wins?",
            "What was Norris's average finishing position in 2023?",
            "How many races did Vettel win between 2010 and 2013?",
            "Give me Bottas's season stats for 2017",
            "How many podium finishes did Ricciardo have in his career?",
            "How many points has Sainz scored since 2019?",
            "What were Russell's results in 2022?",
        ],
        ["driver"], _YEARS,
    ),
    Intent(
        "driver_circuit_record",
        "A driver's record (wins, podiums, average finish) at one specific circuit.",
        [
            "How many wins did Hamilton have at Silverstone before 2020?",
            "What is Verstappen's record at Monza?",
            "How many podiums does Leclerc have at Monaco?",
            "What was Alonso's average finish at Spa?",
            "Has Vettel ever won at Suzuka?",
            "How did Schumacher perform at the Hungaroring?",
            "Hamilton's results at Interlagos since 2015",
            "How many times did Raikkonen finish on the podium at Austin?",
            "Show me Bottas's record at the Red Bull Ring",
            "How often has Sainz scored points at Albert Park?",
        ],
        ["driver", "circuit"], ["year_from", "year_to"],
    ),
    Intent(
        "constructor_stats",
        "Points, wins or reliability (retirements) of a constructor over a season or period.",
        [
            "How reliable was Williams between 2014 and 2021?",
            "How many points did Ferrari score in 2018?",
            "How many races did McLaren win in 2012?",
            "What was Red Bull's win count in 2023?",
            "How often did Mercedes retire from races in 2016?",
            "Give me Renault's stats for 2010",
            "How many podiums did Lotus get between 2012 and 2013?",
            "What is the DNF rate of Haas since 2016?",
            "How many points did Alpine score last decade?",
            "How did the Aston Martin team do in 2021?",
        ],
        ["constructor"], _YEARS,
    ),
    Intent(
        "head_to_head",
        "Which of two drivers finished ahead more often in races both started.",
        [
            "Who finished ahead more often, Norris or Ricciardo at McLaren?",
            "Hamilton vs Rosberg head to head",
            "Did Verstappen beat Perez more often in 2022?",
            "Compare Vettel and Webber in their shared races",
            "How many times did Alonso finish ahead of Massa?",
            "Leclerc versus Sainz in 2021",
            "Who was the faster teammate, Button or Hamilton?",
            "Who beat the other more often between 2014 and 2016, Hamilton or Bottas?",
            "Senna against Prost, who finished in front more often?",
            "Which one finished higher more often, Russell or Hamilton?",
        ],
        ["driver", "driver2"], ["constructor"] + _YEARS,
    ),
    Intent(
        "race_result",
        "Winner and podium of one specific Grand Prix in a given season.",
        [
            "Who won the 2019 German Grand Prix?",
            "Who was on the podium at the 2021 Dutch GP?",
            "Who won the Monaco Grand Prix in 2016?",
            "What was the result of the 2008 Brazilian Grand Prix?",
            "Which drivers finished in the top three at the 2022 Japanese GP?",
            "Who took the win at the 2012 Italian Grand Prix?",
            "Tell me the podium of the 2020 Turkish Grand Prix",
            "Who came first in the 2010 Australian GP?",
            "Winner of the 2017 Singapore Grand Prix",
            "Who finished second at the 2015 British Grand Prix?",
        ],
        ["race_name", "year"], [],
    ),
    Intent(
        "circuit_info",
        "Facts about a circuit: location, country, number of races held, first and last year.",
        [
            "Where is Silverstone located?",
            "In which country is the Monza circuit?",
            "How many Grands Prix have been held at Spa?",
            "When was the first race at Suzuka?",
            "What was the last year Imola hosted a race?",
            "Tell me about the Marina Bay circuit",
            "Which city is the Hungaroring in?",
            "How many races took place at Interlagos?",
            "Give me information about the Circuit of the Americas",
            "Since when does Baku host a Grand Prix?",
        ],
        ["circuit"], [],
    ),
    Intent(
        "qualifying_record",
        "A driver's qualifying record: pole positions, Q3 appearances, average qualifying position.",
        [
            "How many pole positions did Hamilton have?",
            "How many times did Leclerc qualify in Q3 in 2021?",
            "What was Verstappen's average qualifying position in 2022?",
            "How many poles did Senna take in his career?",
            "Did Vettel get pole position often between 2010 and 2013?",
            "How often did Alonso reach Q3 in 2012?",
            "What was Bottas's average grid position in 2019?",
            "How many front row starts did Norris have in 2023?",
            "Which year did Schumacher take most poles?",
            "Number of times Russell started from pole",
        ],
        ["driver"], _YEARS,
    ),
    Intent(
        "grid_gains",
        "Drivers who gained the most places from their grid slot to the finish in a race.",
        [
            "Which driver gained the most places from the grid at Monza in 2021?",
            "Who made up the most positions in the 2019 British Grand Prix?",
            "Biggest grid to finish gainers at the 2018 Italian GP",
            "Who climbed the most places in the 2022 Belgian Grand Prix?",
            "Who overtook the most cars at Interlagos in 2016?",
            "Top position gainers at the 2013 Hungarian Grand Prix",
            "Which driver improved the most from his starting slot at Spa in 2020?",
            "Who moved up the most from the grid at Suzuka in 2014?",
            "Who gained the most places in the 2011 Canadian Grand Prix?",
            "Largest jump from grid to finish at Silverstone in 2012",
        ],
        ["year", "race_name|circuit"], [],
    ),
    Intent(
        "pit_stops",
        "Fastest pit stops of a race and how many stops each driver made.",
        [
            "What was the fastest pit stop at the 2023 Qatar Grand Prix?",
            "How many pit stops did Hamilton make at the 2019 Hungarian Grand Prix?",
            "Who had the quickest pit stop in the 2021 Monaco GP?",
            "Average number of stops per driver in the 2022 Italian Grand Prix",
            "How many times did Verstappen pit at the 2018 Mexican Grand Prix?",
            "Show the fastest stops of the 2020 Bahrain Grand Prix",
            "Which team was quickest in the pits at the 2017 Spanish GP?",
            "How many stops did Alonso make in the 2012 Chinese Grand Prix?",
            "Which driver pitted the most in the 2015 Canadian GP?",
            "Pit stop times for the 2016 Singapore Grand Prix",
        ],
        ["race_name", "year"], ["driver"],
    ),
    Intent(
        "season_standings",
        "Championship standings of a season: points leader, top 10, number of drivers.",
        [
            "Who won the 2018 drivers championship?",
            "Who led the championship in 2021?",
            "Show me the top 10 of the 2017 standings",
            "How many drivers took part in the 2020 season?",
            "Who finished second in the 2016 world championship?",
            "Which driver had the most points in 2014?",
            "What did the 2011 championship table look like?",
            "Who was world champion in 2008?",
            "Final standings for the 2022 season",
            "How many drivers scored points in 2019?",
        ],
        ["year"], [],
    ),
    Intent(
        "podium_prediction",
        "Predict whether a driver will finish on the podium and explain the prediction.",
        [
            "Predict Norris at the 2024 Miami Grand Prix",
            "Will Hamilton finish on the podium at Silverstone?",
            "What are Leclerc's chances of a podium at Monza?",
            "Predict whether Verstappen gets a podium at the 2023 Dutch Grand Prix",
            "How likely is Sainz to get a podium in the next race?",
            "Run the podium model for Alonso at Spa in 2021",
            "Can you forecast if Russell will make the top three?",
            "Is Piastri going to be on the podium at Suzuka?",
            "Probability that Bottas finishes in the top three at the 2019 Bahrain GP",
            "Will Perez podium in Mexico?",
        ],
        ["driver"], ["race_name", "year"],
    ),
    Intent(
        "out_of_scope",
        "Anything the knowledge graph cannot answer: non-F1 topics, weather, salaries, tyres, telemetry, car specs, news, opinions.",
        [
            "What is the capital of France?",
            "What will the weather be like at Silverstone this weekend?",
            "How much does Verstappen earn per year?",
            "Which tyre compound did Hamilton use on lap 20?",
            "Show me the lap-by-lap telemetry of Leclerc at Monaco",
            "What is the top speed of the Red Bull car?",
            "Who is the best driver of all time in your opinion?",
            "Any news about the next season's regulations?",
            "How much horsepower does a Mercedes engine have?",
            "Tell me a joke",
            "Who won the football world cup in 2018?",
            "What is Hamilton's net worth?",
        ],
        [], [],
    ),
]

INTENTS: dict[str, Intent] = {intent.name: intent for intent in _INTENT_LIST}
