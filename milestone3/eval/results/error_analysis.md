# Error analysis

Failures grouped by the stage where they occur. Counts first, then examples (the first few of each list; full lists are in `pipeline_items.csv`, `intent_predictions.csv`, `llm_runs.jsonl`).

## Counts per stage

| stage | source | failures | out of |
| --- | --- | --- | --- |
| intent | intent test set (hybrid_rule_first) | 6 | 94 |
| intent | pipeline items | 0 | 57 |
| entity | pipeline items | 0 | 57 |
| query | pipeline items | 0 | 57 |
| model misread the rows | allam-2-7b (base) | 39 | 104 answers |
| model misread the rows | openai/gpt-oss-20b (base) | 33 | 104 answers |
| model misread the rows | qwen/qwen3.8-27b (base) | 41 | 103 answers |
| model misread the rows | allam-2-7b (strict) | 40 | 104 answers |
| model misread the rows | openai/gpt-oss-20b (strict) | 34 | 104 answers |
| model misread the rows | qwen/qwen3.8-27b (strict) | 35 | 101 answers |
| hallucination | allam-2-7b (base) | 51 | 104 answers |
| hallucination | openai/gpt-oss-20b (base) | 3 | 104 answers |
| hallucination | qwen/qwen3.8-27b (base) | 0 | 103 answers |
| hallucination | allam-2-7b (strict) | 42 | 104 answers |
| hallucination | openai/gpt-oss-20b (strict) | 2 | 104 answers |
| hallucination | qwen/qwen3.8-27b (strict) | 0 | 101 answers |

## Examples

### Intent (intent test set)

6 of 94 questions.

- `How many Grand Prix victories did Brawn get in 2009?` - predicted `race_result`, gold `constructor_stats`
- `How do I cook pasta?` - predicted `rephrase`, gold `out_of_scope`
- `What is Prost's average finishing position at Estoril?` - predicted `driver_season_stats`, gold `driver_circuit_record`

### Intent (pipeline items)

0 of 57 items.


### Entity

0 of 57 items.


### Query

0 of 57 items.


### Model misread the rows - allam-2-7b (base)

39 of 104 answers.

- `What are Fernando Alonso's career totals?` (zero-shot, q03, correctness 0.80) - rows `{'driver': 'Fernando Alonso', 'first_year': 2001, 'last_year': 2024, 'races': 402, 'points': 2329.0, 'wins': 32, 'podiums': 106, 'classified': 325}` - answer: "Fernando Alonso, with the driverId 4, has participated in 402 races since his debut in 2001. Across these races, he has secured 32 victories, 106 podium finishes, and has been classified in 325 races. The data provided covers his career from 2001 to 2024, focusing on World Championship race results ..."
- `What about Norris?` (zero-shot, q04, correctness 0.75) - rows `{'driver': 'Lando Norris', 'first_year': 2019, 'last_year': 2019, 'races': 20, 'points': 49.0, 'wins': 0, 'podiums': 0, 'classified': 15}` - answer: "In the 2019 season, Lando Norris participated in 20 races, scoring 49 points, without achieving any wins or podiums. Out of those 20 races, he was classified in 15 events. The data provided covers the World Championship race results from 1950 to the 2024 season, excluding sprint races, never-started..."
- `What about Norris?` (few-shot, q04, correctness 0.75) - rows `{'driver': 'Lando Norris', 'first_year': 2019, 'last_year': 2019, 'races': 20, 'points': 49.0, 'wins': 0, 'podiums': 0, 'classified': 15}` - answer: "In the 2019 season, Lando Norris participated in 20 races, scoring 49 points, but did not achieve any wins, podiums, or retirements (did not finish the race). He classified in 15 races. The data provided does not cover the 1950s, 1960s, or 1970s, nor does it include sprint races, nor does it cover t..."

### Model misread the rows - openai/gpt-oss-20b (base)

33 of 104 answers.

- `What are Fernando Alonso's career totals?` (few-shot, q03, correctness 0.80) - rows `{'driver': 'Fernando Alonso', 'first_year': 2001, 'last_year': 2024, 'races': 402, 'points': 2329.0, 'wins': 32, 'podiums': 106, 'classified': 325}` - answer: "Fernando Alonso has competed in 402 races, scored 2329 points, won 32 times, and stood on the podium 106 times (World Championship race results from 1950 to the 2024 season; sprint races are not in the data; unfinished or disqualified entries count as starts, never‑started entries are excluded)."
- `What about Norris?` (zero-shot, q04, correctness 0.75) - rows `{'driver': 'Lando Norris', 'first_year': 2019, 'last_year': 2019, 'races': 20, 'points': 49.0, 'wins': 0, 'podiums': 0, 'classified': 15}` - answer: "Lando Norris raced 20 times in 2019, scored 49 points, and finished 15 races classified. *Scope: World Championship race results from 1950 to 2024; sprint races and sprint points are excluded.*"
- `What about Norris?` (few-shot, q04, correctness 0.75) - rows `{'driver': 'Lando Norris', 'first_year': 2019, 'last_year': 2019, 'races': 20, 'points': 49.0, 'wins': 0, 'podiums': 0, 'classified': 15}` - answer: "Lando Norris raced 20 times in 2019, scored 49 points, and finished 15 races classified, with no wins or podiums (World Championship race results from 1950 to 2024, sprint races excluded)."

### Model misread the rows - qwen/qwen3.8-27b (base)

41 of 103 answers.

- `What are Fernando Alonso's career totals?` (few-shot, q03, correctness 0.80) - rows `{'driver': 'Fernando Alonso', 'first_year': 2001, 'last_year': 2024, 'races': 402, 'points': 2329.0, 'wins': 32, 'podiums': 106, 'classified': 325}` - answer: "Fernando Alonso's career totals are 402 races, 32 wins, 106 podiums, and 2329 points (World Championship race results from 1950 to the 2024 season, excluding sprint races)."
- `What about Norris?` (zero-shot, q04, correctness 0.75) - rows `{'driver': 'Lando Norris', 'first_year': 2019, 'last_year': 2019, 'races': 20, 'points': 49.0, 'wins': 0, 'podiums': 0, 'classified': 15}` - answer: "In the 2019 season, Lando Norris started 20 races and scored 49 points, with 0 wins and 0 podiums. These figures cover World Championship race results from 1950 to 2024, excluding sprint races and counting unfinished or disqualified entries as starts."
- `What about Norris?` (few-shot, q04, correctness 0.75) - rows `{'driver': 'Lando Norris', 'first_year': 2019, 'last_year': 2019, 'races': 20, 'points': 49.0, 'wins': 0, 'podiums': 0, 'classified': 15}` - answer: "Lando Norris started 20 races in 2019, finishing classified in 15 of them and scoring 49 points (World Championship race results from 1950 to the 2024 season; sprint races are not in the data; unfinished or disqualified entries count as starts, never-started entries are excluded. Points are race poi..."

### Model misread the rows - allam-2-7b (strict)

40 of 104 answers.

- `What are Fernando Alonso's career totals?` (zero-shot, q03, correctness 0.80) - rows `{'driver': 'Fernando Alonso', 'first_year': 2001, 'last_year': 2024, 'races': 402, 'points': 2329.0, 'wins': 32, 'podiums': 106, 'classified': 325}` - answer: "Fernando Alonso, driving ID 4, has participated in 402 races, earning 2329 points, achieving 32 victories, and securing 106 podiums. The data provided covers World Championship race results from 1950 to the 2024 season, focusing on races, points, wins, and podiums, while excluding sprint races, poin..."
- `What about Norris?` (zero-shot, q04, correctness 0.75) - rows `{'driver': 'Lando Norris', 'first_year': 2019, 'last_year': 2019, 'races': 20, 'points': 49.0, 'wins': 0, 'podiums': 0, 'classified': 15}` - answer: "In the 2019 season, Lando Norris participated in 20 races, scoring 49 points, with no wins, zero podiums, and being classified in 15 races. The data provided covers Formula 1 World Championship race results from 1950 to the 2024 season, excluding sprint races, and includes race points only."
- `What about Norris?` (few-shot, q04, correctness 0.50) - rows `{'driver': 'Lando Norris', 'first_year': 2019, 'last_year': 2019, 'races': 20, 'points': 49.0, 'wins': 0, 'podiums': 0, 'classified': 15}` - answer: "The data is insufficient to answer the question about Norris's performance in the 2019 season. Lando Norris participated in 20 races, scoring 49 points, with no wins, podiums, or classified finishes mentioned."

### Model misread the rows - openai/gpt-oss-20b (strict)

34 of 104 answers.

- `What about Norris?` (zero-shot, q04, correctness 0.75) - rows `{'driver': 'Lando Norris', 'first_year': 2019, 'last_year': 2019, 'races': 20, 'points': 49.0, 'wins': 0, 'podiums': 0, 'classified': 15}` - answer: "Lando Norris raced 20 times in 2019, scored 49 points, and finished 15 races classified. *Scope: World Championship race results from 1950 to 2024; sprint races are not included, and only race points are counted.*"
- `What about Norris?` (few-shot, q04, correctness 0.75) - rows `{'driver': 'Lando Norris', 'first_year': 2019, 'last_year': 2019, 'races': 20, 'points': 49.0, 'wins': 0, 'podiums': 0, 'classified': 15}` - answer: "Lando Norris raced 20 times in 2019, scored 49 points, and finished 15 classified races (World Championship race results from 1950 to 2024; sprint races are not in the data; unfinished or disqualified entries count as starts, never‑started entries are excluded)."
- `And in 2020?` (few-shot, q05, correctness 0.40) - rows `{'driver': 'Lewis Hamilton', 'first_year': 2020, 'last_year': 2020, 'races': 16, 'points': 347.0, 'wins': 11, 'podiums': 14, 'classified': 16}` - answer: "Lewis Hamilton won 11 races in 2020 (World Championship race results from 1950 to the 2024 season; sprint races are not in the data; unfinished or disqualified entries count as starts, never‑started entries are excluded)."

### Model misread the rows - qwen/qwen3.8-27b (strict)

35 of 101 answers.

- `What about Norris?` (few-shot, q04, correctness 0.50) - rows `{'driver': 'Lando Norris', 'first_year': 2019, 'last_year': 2019, 'races': 20, 'points': 49.0, 'wins': 0, 'podiums': 0, 'classified': 15}` - answer: "Lando Norris started 20 races in 2019 and scored 49 points (World Championship race results from 1950 to the 2024 season; sprint races are not in the data; unfinished or disqualified entries count as starts, never-started entries are excluded. Points are race points only, so sprint points are missin..."
- `And in 2020?` (few-shot, q05, correctness 0.60) - rows `{'driver': 'Lewis Hamilton', 'first_year': 2020, 'last_year': 2020, 'races': 16, 'points': 347.0, 'wins': 11, 'podiums': 14, 'classified': 16}` - answer: "Lewis Hamilton recorded 11 wins and 347 points in 2020 (World Championship race results from 1950 to the 2024 season, excluding sprint races)."
- `How many times has Lewis Hamilton won at Silverstone before 2020?` (zero-shot, q07, correctness 0.20) - rows `{'driver': 'Lewis Hamilton', 'circuit': 'Silverstone Circuit', 'first_year': 2007, 'last_year': 2019, 'starts': 13, 'wins': 6, 'podiums': 9, 'best_fin...` - answer: "Based on the World Championship race results from 1950 to the 2024 season, Lewis Hamilton has 6 wins at Silverstone Circuit before 2020."

### Hallucination (check the unsupported tokens: scope-sentence words and computed totals are false positives) - allam-2-7b (base)

51 of 104 answers.

- `How many wins did Lewis Hamilton have in 2019?` (zero-shot, q01) - unsupported numbers [], names ['Formula'] - answer: "In the context provided, Lewis Hamilton had 11 wins during the 2019 Formula 1 season. The data covers World Championship race results from 1950 to the 2024 season, excluding sprint races, never-started entries, and sprint points."
- `How many wins did Lewis Hamilton have in 2019?` (few-shot, q01) - unsupported numbers [], names ['Formula'] - answer: "In the 2019 Formula 1 season, Lewis Hamilton had 11 race wins from 21 starts, during that year. The data provided covers the number of wins, points, podiums, and classified finishes, but not the number of starts."
- `How many podiums did Lando Norris have in 2022?` (zero-shot, q02) - unsupported numbers [], names ['Formula'] - answer: "Based on the provided data, Lando Norris had 1 podium finish in the 2022 season. The scope of the figures covers Formula 1 World Championship race results from 1950 to the 2024 season, excluding sprint races, and includes race points only."

### Hallucination (check the unsupported tokens: scope-sentence words and computed totals are false positives) - openai/gpt-oss-20b (base)

3 of 104 answers.

- `How did Charles Leclerc do at Monaco between 2018 and 2022?` (zero-shot, q09) - unsupported numbers [], names ['Grands', 'Prix'] - answer: "Between 2018 and 2022, Charles Leclerc started four Monaco Grands Prix, did not win any, did not finish on the podium, his best finish was 4th, and his average finish was 15.5. *Scope: World Championship race results from 1950 to 2024; sprint races are not included; unfinished or disqualified entrie..."
- `Show me the top finishers of the 2021 Italian Grand Prix.` (few-shot, q19) - unsupported numbers [], names ['These'] - answer: "The top finishers of the 2021 Italian Grand Prix were: 1. Daniel Ricciardo – McLaren – 26 points 2. Lando Norris – McLaren – 18 points 3. Valtteri Bottas – Mercedes – 15 points 4. Charles Leclerc – Ferrari – 12 points 5. Sergio Pérez – Red Bull – 10 points (These are the World Championship race resu..."
- `What were the quickest stops at the 2019 Italian Grand Prix?` (few-shot, q33) - unsupported numbers ['3', '4', '5'], names ['These'] - answer: "The quickest pit stops at the 2019 Italian Grand Prix were: 1. Lance Stroll – 18.838 seconds (stop 1, lap 15) 2. Valtteri Bottas – 23.7 seconds (stop 1, lap 27) 3. Lewis Hamilton – 23.731 seconds (stop 2, lap 49) 4. Charles Leclerc – 23.833 seconds (stop 1, lap 20) 5. Robert Kubica – 23.86 seconds (..."

### Hallucination (check the unsupported tokens: scope-sentence words and computed totals are false positives) - qwen/qwen3.8-27b (base)

0 of 103 answers.


### Hallucination (check the unsupported tokens: scope-sentence words and computed totals are false positives) - allam-2-7b (strict)

42 of 104 answers.

- `How many wins did Lewis Hamilton have in 2019?` (zero-shot, q01) - unsupported numbers [], names ['Formula'] - answer: "Based on the provided data, Lewis Hamilton had 11 wins in the 2019 Formula 1 season. Please note that the scope of the figures covers World Championship race results from 1950 to the 2024 season, excluding sprint races, and points are race points only."
- `How many podiums did Lando Norris have in 2022?` (zero-shot, q02) - unsupported numbers [], names ['Formula'] - answer: "Based on the provided data, Lando Norris had 1 podium in the 2022 Formula 1 season. The scope of the figures covers World Championship race results from 1950 to the 2024 season, excluding sprint races, and points are race points only."
- `What are Fernando Alonso's career totals?` (zero-shot, q03) - unsupported numbers [], names ['ID'] - answer: "Fernando Alonso, driving ID 4, has participated in 402 races, earning 2329 points, achieving 32 victories, and securing 106 podiums. The data provided covers World Championship race results from 1950 to the 2024 season, focusing on races, points, wins, and podiums, while excluding sprint races, poin..."

### Hallucination (check the unsupported tokens: scope-sentence words and computed totals are false positives) - openai/gpt-oss-20b (strict)

2 of 104 answers.

- `Show me the top finishers of the 2021 Italian Grand Prix.` (few-shot, q19) - unsupported numbers [], names ['These'] - answer: "The top finishers of the 2021 Italian Grand Prix are: 1. Daniel Ricciardo – 26 points 2. Lando Norris – 18 points 3. Valtteri Bottas – 15 points 4. Charles Leclerc – 12 points 5. Sergio Pérez – 10 points 6. Carlos Sainz – 8 points 7. Lance Stroll – 6 points 8. Fernando Alonso – 4 points 9. George Ru..."
- `Biggest grid climbers at the 2019 Brazilian Grand Prix?` (few-shot, q29) - unsupported numbers [], names ['These'] - answer: "The biggest grid climbers at the 2019 Brazilian Grand Prix were: - Carlos Sainz – 17 places gained (grid 20, finish 3) - Antonio Giovinazzi – 7 places gained (grid 12, finish 5) (These figures cover World Championship race results from 1950 to 2024, excluding sprint races and only include classified..."

### Hallucination (check the unsupported tokens: scope-sentence words and computed totals are false positives) - qwen/qwen3.8-27b (strict)

0 of 101 answers.

