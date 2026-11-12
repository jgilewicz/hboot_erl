# paper-v2: kontrole i 6 środowisk

- **Status:** planned
- **Account:** `hpc-jakgil6519-1780267238` (SC-ERL)
- **Started:** 2026-11-12
- **Code:** git@github.com:jgilewicz/usc_erl_jax.git, branch `paper-v2`
- **Spec:** `experiment.toml` (siatka, metryka, porównania)

## Pytanie

- Czy populacja pomaga ponad sam TD3 (H1)?
- Czy zysk prefiksu bierze się z podziału budżetu, a nie z jakości oceny (H2)?
- Czy wynik prefiks > ERL przenosi się na Swimmer i Humanoid (H3)?

## Zasady

- Kod i logowanie jak `paper-core-v1` (commit `67c7b58`), bez nowych metryk.
- Jedyna zmiana kodu: tryb `eval_mode=rl_only` (warunek `td3`), commit `0ce868b`.
  - `train_and_merge` rozdzielone na `train_td3` + merge, kolejność kluczy RNG bez zmian.
  - `maybe_evaluate`: okno ewaluacji dla `rl_only` = sam epizod RL, dla pozostałych trybów bez zmian.
- CORE na HalfCheetah, Hopper, Walker2d, Ant nie jest powtarzany: komórki z tagu `paper-core-v1` (`join` w toml).
- Nowe runy: tag `paper-v2`, nazwy `<warunek>_<env>_s<seed>`, `WANDB_MODE=online`.

## Hipotezy (prerejestracja)

| | etap | porównanie | kryterium | FAIL znaczy |
|---|---|---|---|---|
| H1 | 1 | prefix-h200 vs td3 | 95% CI różnicy IQM (wynik końcowy, znormalizowany) > 0 | populacja nie pomaga |
| H2 | 1 | prefix-h200 vs sc_erl-always | \|ΔIQM\| < 0.1 | różnica jakości oceny, nie podziału budżetu |
| H3 | 2 | prefix-h200 vs erl | wyższa średnia i ≥ 7/10 wygranych seedów w Swimmer i w Humanoid | prefiks nie przenosi się |

- Prefiks vs SC-ERL w etapie 2: opisowo, bez testu.
- H1 i H2 liczone na 4 envach etapu 1, po etapie 2 powtórzone na 6.

## Statystyka

- Wynik końcowy: średnia `perf/eval_rl` z ostatnich 10% budżetu (1M `env_steps`).
- AUC: średnia `perf/eval_rl` po wszystkich ewaluacjach (pole pod krzywą / budżet).
- Normalizacja per env: 0 = najgorsza średnia warunku, 1 = najlepsza, po wszystkich 7 warunkach (liczby nie są wprost porównywalne z raportem `paper-core-v1`, który normalizował po 5).
- IQM z 95% CI ze stratyfikowanego bootstrapu po (env, seed), dla wyniku końcowego i AUC.
- P(A>B) po parach seedów (ten sam env, ten sam seed).
- Po etapie 2: Wilcoxon po 6 envach + Holm po H1, H2, H3.

## Etapy

| etap | envy | warunki | runy | stan |
|---|---|---|---|---|
| 0 | HalfCheetah | prefix-h200, seed 0, 100k kroków (regresja) | 1 lokalnie | PASS |
| 1 | HalfCheetah, Hopper, Walker2d, Ant | td3, sc_erl-always × seedy 0-9 | 80 | czeka |
| 2a | Humanoid | 7 warunków × seed 0 (pilot: czas, czy TD3 się uczy w 1M) | 7 | czeka |
| 2b | Swimmer | 7 warunków × seedy 0-9 | 70 | czeka |
| 2c | Humanoid | 7 warunków × seedy 1-9 | 63 | po pilocie |

- Swimmer `6256649` (stary CORE): już `CANCELLED`, nigdy nie wystartował, nic do wycofania.

## Budżet

Saldo SC-ERL 2026-11-12 (`discover.sh`): **6110.9 CPU-h, 1747.2 GPU-h**.
Czas na run z `paper-core-v1` (200 runów, lem-gpu, `-c 4`, 1 H100):

| warunek | średnio | max |
|---|---|---|
| erl | 0.52 h | 0.66 h |
| sc_erl-random | 1.69 h | 2.42 h |
| prefix-h200 | 1.07 h | 1.32 h |
| prefix-td | 0.68 h | 0.84 h |
| hboot-td | 0.68 h | 0.86 h |

| etap | runy | założenie | GPU-h | CPU-h (×4) |
|---|---|---|---|---|
| 1 | 80 | td3 ~1.0 h (brak pomiaru), sc_erl-always ~1.7 h | ~110 (max ~160) | ~430 (max ~640) |
| 2a | 7 | Humanoid ~3 h (brak pomiaru) | ~20 | ~85 |
| 2b | 70 | średnia warunków ~1.05 h | ~75 | ~300 |
| 2c | 63 | z pilota | ~190 przy 3 h | ~760 przy 3 h |
| razem | 220 | | ~395 | ~1575 |

- Zapas: ok. 4× w CPU-h, 4× w GPU-h. Przed 2c przeliczyć z czasów pilota.

## Runy

| Job ID | Data | Etap | Env | Array | Commit |
|---|---|---|---|---|---|

## Wyniki

### Etap 0: regresja (PASS)

- Commit kodu `0ce868b`, lokalnie (Mac M, CPU, JAX), `prefix-h200` HalfCheetah-v5 seed 0, 100k `env_steps`, 31 generacji.
- Porównanie z 10 seedami `prefix-h200` HalfCheetah z `paper-core-v1` (H100), na tych samych krokach ewaluacji.
- Kroki ewaluacji identyczne jak w CORE (11k, 17k, 20k, ..., 101k), więc licznik kroków i wyzwalacz ewaluacji bez zmian.
- 16/19 punktów w kopercie min–max CORE. 3 punkty nad max o ≤ 50 (47k: 3403 vs 3354, 62k: 4163 vs 4153, 77k: 4718 vs 4700). Żaden poniżej min.
- Przy 101k: 4848 (CORE 3356–5170, seed 0 w CORE 4445). Bitowej zgodności z CORE s0 nie ma (CPU vs GPU).

| krok | CORE min | CORE max | lokalnie |
|---|---|---|---|
| 20k | -809 | 595 | 307 |
| 41k | 1417 | 3305 | 3161 |
| 62k | 1633 | 4153 | 4163 |
| 80k | 2128 | 4963 | 4466 |
| 101k | 3356 | 5170 | 4848 |
