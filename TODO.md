# TODO — paper plan

Tytuł: *Do We Need the Critic? Truncated Fitness Evaluation in Budget-Constrained Evolutionary Reinforcement Learning*
Cel: GECCO 2027 (deadline 25.01.2027). Teza i oś: [README.md](README.md#research-question).

## Eksperymenty

| # | Co | Suite / envy | Runy | ~CPU-h | Status |
|---|---|---|---|---|---|
| 1 | TD3 bez populacji + sc_erl-always | `controls` × HC, Hopper, Walker2d, Ant × 10 seedów | 80 | ~250 | kod gotowy (`rl_only`), nie wysłane |
| 2 | Swimmer CORE + Humanoid CORE | `core` × Swimmer-v5, Humanoid-v5 | 100 | ~450 | Swimmer 6256649 PENDING (QOSGrpCPUMinutesLimit); Humanoid dodany do rejestru |
| 3 | Horyzont h=100/400/800 (h=200 z CORE) | `ablation-horizon` × 4 envy × 10 seedów | 120 | ~450 | nie wysłane |
| 4 | Zgodność rankingu: prefix vs SC-ERL vs pełny rollout | HC + Walker2d × 5 seedów, checkpointy + offline eval | 20 | ~100 | brak kodu |
| | **Razem** | | 320 | ~1250 | |

- [ ] **Budżet**: grant ma 136 CPU-h (< 1250). Decyzja: Solvro (wszystkie punkty, także #1) vs doładowanie grantu.
- [ ] Swimmer `6256649_[0-49]`: zostawić w kolejce czy anulować i wysłać na Solvro.
- [ ] Humanoid-v5: lokalny smoke OK (obs 348, ~3.7 GB RSS, bufor 1M). Zmierzyć czas/run na lem-gpu (1 task) przed wysłaniem 50.
- [ ] Wysłać #1: `./scripts/submit_suite.sh controls HalfCheetah-v5 Hopper-v5 Walker2d-v5 Ant-v5`
- [ ] Wysłać #2: `TAG=paper-core-v1 ./scripts/submit_suite.sh core Swimmer-v5 Humanoid-v5`
- [ ] Wysłać #3: `./scripts/submit_suite.sh ablation-horizon HalfCheetah-v5 Hopper-v5 Walker2d-v5 Ant-v5`
- [ ] #1 i #3 dla Swimmer/Humanoid, jeśli budżet pozwoli (6 envów → Wilcoxon + Holm).

## Kod

- [x] `eval_mode=rl_only` (TD3 kontrola: ta sama sieć, budżet i ewaluacja, bez populacji/gene flow).
- [ ] #4: zapis checkpointów populacji (+ krytyka) co N generacji; skrypt offline: ranking wg prefixu h, surogatu SC-ERL i pełnego rolloutu → Spearman, overlap elity.
- [ ] Logować `select/*` / `gate/*` w runach CORE (teraz puste).
- [ ] `eval_reward` po treningu do wandb (teraz tylko JSON).
- [ ] `_step_generation` używa `cfg.eval_mode` zamiast `_resolve_eval_mode(cfg)` — dotyczy tylko `eval_mode=full` + `prefix_steps`; ujednolicić albo usunąć backward-compat.

## Analiza i tekst

- [ ] Oś 2: rozkład zysku ERL → prefix-h200 na (a) TD3 sam, (b) więcej generacji, (c) surogat (sc_erl-always).
- [ ] Oś 3: wykres h → liczba generacji → final score (h=100/200/400/800 + full).
- [ ] Oś 4: hboot-td vs prefix-td + diagnoza saturacji kontrolera TD (δ_ema rośnie ze skalą zwrotów).
- [ ] Statystyka: IQM + stratified bootstrap po (env, seed) (rliable); P(A>B); Wilcoxon + Holm przy ≥ 6 envach.

## Mail do promotora

- [ ] Wysłać mail (TL;DR, wyniki, teza, plan, GECCO 2027, zasoby). Punkt 4 (zgoda na Solvro) tylko jeśli liczenie na Solvro wymaga zgody promotora.
