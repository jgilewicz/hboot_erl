# paper-v2 pilot metryki: champion populacji vs aktor RL

- **Status:** zgoda na 18 runów (2026-11-14). Dodane `pop/champion_fitness`, test zgodności PASS ponownie.
- **Account:** `hpc-jakgil6519-1780267238`
- **Prerejestracja:** 2026-11-14, przed pierwszym runem pilota
- **Spec:** `experiment.toml`
- Powód: kontrola implementacji (`../paper-v2-stage2/record.md`). ERL-Re2 i EvoRainbow raportują max(aktor RL, champion), nasz `perf/eval_rl` tylko aktora TD3.
- Krok B etapu 2 i zmiany konfiguracji (pop 5, ea_tau, γ, tanh) wstrzymane do wyniku pilota.

## Zmiana kodu (tylko logowanie)

- `perf/eval_champion`: `cem.elite` po `tell` (argmax fitnessu bieżącej generacji, ten sam osobnik, do którego ciągnie soft update).
  - 10 epizodów deterministycznych, osobne środowisko ewaluacji, seed jak `eval_rl` (seed + 10000), więc pierwsze 5 epizodów ma te same stany startowe co `eval_rl`.
  - Embedding: ten, z którym populacja była oceniana w tej generacji (przed treningiem TD3 w tej generacji).
  - Kroki ewaluacji nie liczą się do budżetu.
- `perf/eval_best = max(eval_rl, eval_champion)`, w tych samych punktach co `eval_rl` (co 5000 kroków).
- `pop/champion_fitness`: `max(fitness)` po `cem.tell` w każdej generacji (= fitness `cem.elite`), w skali fitnessu warunku: erl pełny epizod, prefix-h200 zwrot z 200 kroków, sc_erl-random skalibrowana mieszanka zwrotu i oceny krytyka. Ten sam wiersz wandb co `eval_champion`.
- `eval_rl` bez zmian (5 epizodów). td3 (`rl_only`) loguje tylko `eval_rl`.
- Bez zmian algorytmu i hiperparametrów. Ewaluacja nie zużywa klucza RNG treningu.

## Test zgodności

- HalfCheetah-v5, prefix-h200, s0, 100k kroków, lokalnie (CPU), 2× stary commit `c626aaf` + 1× nowy kod.
- Kryterium: `eval_rl` nowego w granicach rozrzutu dwóch przebiegów starego.
- Wynik (2026-11-14): PASS. 20/20 punktów `eval_rl` identyczne co do bitu w 3 przebiegach (rozrzut starego = 0), np. 100k: 4848.1. Czas: stary ~21 min, nowy ~22.5 min.
- Po dodaniu `pop/champion_fitness`: powtórka nowego kodu, 31/31 linii `eval_rl` identyczne ze starym commitem, 100k: 4848.1.
- Smoke Swimmer erl s0, 40k kroków: `eval_champion` logowany (11k: rl 27.6, champion 25.7). `eval_rl` przy 11k = 27.6, tak jak w runie klastrowym `erl_Swimmer-v5_s0`.

## Runy

- Swimmer-v5, Ant-v5 × erl, prefix-h200, sc_erl-random × seedy 0-2 = 18 runów. Tag `paper-v2-metric`.
- `SUITE=core`, `--array=0-2,10-12,20-22` per env.
- Koszt z czasów etapu 2 / `paper-core-v1` (bez narzutu championa): Swimmer 3.3 h, Ant 3.8 h na seed, razem ~21.5 GPU-h, ~86 CPU-h.

## Rozstrzygnięcie (ustalone przed runami)

- Per (env, warunek), średnie z ostatnich 10% budżetu z tego samego runu, seedy 0-2.
- "Champion wyraźnie > aktor RL": średnia(champion − rl) ≥ 10% |średnia(rl)| i champion > rl w 3/3 seedach. "Champion ≈ aktor RL": |różnica| < 10%.
- Swimmer, champion wyraźnie > RL: dotychczasowe wyniki Swimmera zaniżają ERL, krok B z nową metryką.
- Ant, champion ≈ RL: wyniki grupy łatwej ważne w obu metrykach. Champion wyraźnie > RL: decyzja o powtórce części macierzy CORE.
- Raport: eval_rl, eval_champion, eval_best w czasie i na końcu, udział ewaluacji z champion > RL, porównanie z td3 w obu metrykach (td3: `eval_rl` z istniejących runów).
- Wynik główny: `eval_rl` i `eval_champion` osobno. `eval_best` tylko jako odpowiednik metryki z literatury, z adnotacją: maksimum z dwóch zaszumionych ocen (5 epizodów aktora, 10 championa), więc zawyżone względem td3 z jedną oceną.
- Trafność wyboru championa: obok `eval_champion` fitness treningowy championa z tej samej generacji (`pop/champion_fitness`), per warunek.
