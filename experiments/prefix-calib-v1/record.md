# prefix-calib-v1 record

## Plan & Pre-registration (2026-11-04)

- **Cel:** Sprawdzić, czy stałe $h$ dobrane osobno dla każdego środowiska bije stałe $h=200$, oraz czy jednorazowa kalibracja na początku treningu znajduje to $h$ tanio (po odrzuceniu adaptacji online w `prefix-adapt-v1` przez narzut 8–24% i blokadę $h=50$ na `Hopper-v5`).
- **Gałąź i budżet:** `feat/sc-erl`, tag W&B `prefix-calib-v1`, budżet `1_000_000` `env_steps` na run (wszystkie kroki, w tym kalibracyjne, wliczane do budżetu).

### Etap A — Sufit (`HalfCheetah-v5`, `Hopper-v5`)
- **Warunki:** `prefix-h100`, `prefix-h200`, `prefix-h400`, `erl-log` (pełne $H=1000$ z logowaniem zwrotów prefiksu $h \in \{50, 100, 200, 400, 1000\}$ oraz długości epizodu każdego osobnika).
- **Seedy:** `0, 1, 2, 3, 4` (5 seedów na warunek i środowisko = 40 runów łącznie).
- **Re-użycie istniejących runów:** Seedy `0–2` dla `prefix-h100` i `prefix-h200` na `HalfCheetah-v5` (z `prefix-v1`) i `Hopper-v5` (z `prefix-adapt-v1`) mają identyczną konfigurację i ścieżkę kodu (`_prefix_generation` bez zmian) — zostają otagowane `prefix-calib-v1` (12 runów). Nowe przebiegi w Etapie A: 28 runów (`prefix-h100` i `prefix-h200` seedy `3–4`, `prefix-h400` seedy `0–4`, `erl-log` seedy `0–4` na obu środowiskach).
- **Metryki Etapu A:**
  1. Końcowe `MA(5)` z `perf/eval_rl` (średnia z ostatnich 5 ewaluacji).
  2. AUC krzywej `perf/eval_rl` po `env_steps`.
  3. Kroki `env_steps` do pierwszego osiągnięcia końcowego `MA(5)` `sc_erl-random` (z istniejących runów).
- **Reguła A (zapisana przed startem):**
  - $h_{\text{best}}(\text{env}) = \arg\max_{h \in \{100, 200, 400, 1000\}} \text{mean}(\text{końcowe MA(5)})$.
  - **GO do Etapu B**, jeśli w co najmniej jednym środowisku (`HalfCheetah-v5` lub `Hopper-v5`):
    1. $h_{\text{best}}(\text{env}) \ne 200$, ORAZ
    2. $h_{\text{best}}$ bije `prefix-h200` w $\ge 4/5$ seedach (sparowane po seedzie), ORAZ
    3. średnia różnica końcowego `MA(5)` wynosi $\ge +5\%$.
  - **W przeciwnym razie STOP:** $h=200$ zostaje domyślne, a wynik raportujemy jako niewrażliwość na $h$.
- **Analiza offline na `erl-log` (bez dodatkowych runów):**
  - Dla okna kalibracji zaczynającego się przy $T_0 \in \{50\text{k}, 100\text{k}\}$ kroków, $K=3$ kolejne generacje:
    $$\rho(h) = \frac{1}{K}\sum_{g=1}^{K} \text{Spearman}(\text{return}_h^{(g)}, \text{return}_{1000}^{(g)})$$
  - Reguła wyboru: $h^* = \min \{h \in \{100, 200, 400\} : \rho(h) \ge \tau\}$; jeśli żadne, to $400$.
  - Na `HalfCheetah-v5` i `Hopper-v5` dobrać $\tau \in \{0.8, 0.85, 0.9, 0.95\}$ i $T_0 \in \{50\text{k}, 100\text{k}\}$ tak, aby $h^* = h_{\text{best}}$ w możliwie wielu seedach. Zamrozić $(\tau, T_0)$ w `record.md` przed Etapem B; zaraportować $h^*$ per seed i jego stabilność (to samo $h^*$ w $\ge 4/5$ seedach).

### Etap B — Kalibracja na nowych środowiskach (warunkowo, TYLKO po GO z Etapu A)
- **Środowiska held-out:** `Walker2d-v5`, `Ant-v5` ($\tau$ i $T_0$ zamrożone z Etapu A).
- **Warunki:** `prefix-calib`, `prefix-h100`, `prefix-h200`, `prefix-h400`, `sc_erl-random`, `erl`, seedy `0–4`.
- **Mechanizm `prefix-calib`:**
  1. Do $T_0$ trening z $h=200$.
  2. Następnie $K=3$ generacje z pełnym $H=1000$; z tych rolloutów $\rho(h)$ i wybór $h^* \in \{100, 200, 400\}$.
  3. Reszta treningu ze stałym $h^*$. Logowanie $h^*$, $\rho(h)$ i narzutu kroków.
- **Reguła B:**
  - **GO**, jeśli `prefix-calib >= prefix-h200` (średnie końcowe `MA(5)`) w obu środowiskach ORAZ bije `prefix-h200` o $\ge 3\%$ w co najmniej jednym ORAZ narzut kalibracji $\le 5\%$ budżetu.
  - Raportowany żal względem najlepszego stałego $h$: $\text{regret}(\text{calib})$ vs $\text{regret}(h_{200})$, oraz zgodność $h^* = h_{\text{best}}$ w $\ge 3/5$ seedach.

### Szacunek kosztu obliczeniowego Etapu A (przed startem)
- **28 nowych zadań SLURM** na `lem-gpu-short` (`-c 8`, `1x H100` na zadanie):
  - `4x prefix-h100` (`s3, s4` na HC i Hopper): `~30.4 CPU-h / ~3.8 GPU-h`
  - `4x prefix-h200` (`s3, s4` na HC i Hopper): `~25.0 CPU-h / ~3.1 GPU-h`
  - `10x prefix-h400` (`s0..4` na HC i Hopper): `~38.4 CPU-h / ~4.8 GPU-h`
  - `10x erl-log` (`s0..4` na HC i Hopper): `~24.0 CPU-h / ~3.0 GPU-h`
  - **Łączny szacunek Etapu A:** **~118 CPU-h / ~14.7 GPU-h** (czas zegarowy ~45–60 min).

## Submissions
