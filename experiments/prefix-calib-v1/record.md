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

- 2026-11-04 (Stage A): submitted @ `00f9d5c` (`feat/sc-erl`), `lem-gpu-short`, `--exclude=r16-11`, `-A hpc-jakgil6519-1780267238`
  - `6236415` `HalfCheetah-v5` (`prefix-h200` s3-4, `prefix-h100` s3-4, `prefix-h400` s0-4, `erl-log` s0-4; `--array=63-64,68-69,80-84,85-89`, `-c 8 --mem=16gb --time=0-04:00:00`)
  - `6236416` `Hopper-v5` (`prefix-h200` s3-4, `prefix-h100` s3-4, `prefix-h400` s0-4, `erl-log` s0-4; `--array=63-64,68-69,80-84,85-89`, `-c 8 --mem=16gb --time=0-04:00:00`)
  - balance before Stage A: `1172.8 CPU-h / 472.9 GPU-h`
  - estimated Stage A cost: `~118 CPU-h / ~14.7 GPU-h` (28 new tasks; 12 tasks for `prefix-h100`/`prefix-h200` seeds `0..2` reused from `prefix-v1` and `prefix-adapt-v1`).
- 2026-11-04: Stage A complete (`6236415`: 88 CPU-h / 11.0 GPU-h; `6236416`: 63 CPU-h / 7.9 GPU-h; total 151 CPU-h / 18.9 GPU-h).

## Results: Stage A Ceiling (2026-11-04)

- **Report:** `experiments/prefix-calib-v1/report.html`
- **Cost:** 151 CPU-h / 18.9 GPU-h.
- **Summary Metrics (5 seeds `0..4`, 1M `env_steps`):**
  - **`HalfCheetah-v5`:**
    - `prefix-h200`: **10142.7 ± 720.5** (med 10573.2, AUC **8097.5 ± 630.1**; per-seed: `10573.2, 9129.4, 9629.8, 10707.8, 10673.5`)
    - `prefix-h100`: **8912.2 ± 1115.6** (med 8501.8, AUC 7420.4 ± 838.1; per-seed: `8489.8, 8501.8, 7588.0, 9465.1, 10516.0`)
    - `prefix-h400`: **8849.8 ± 1292.6** (med 8869.3, AUC 6858.1 ± 928.1; per-seed: `10497.3, 6960.7, 8869.3, 9394.3, 8527.5`)
    - `erl-log` ($H=1000$): **7908.0 ± 1339.9** (med 8151.6, AUC 5421.3 ± 840.3; per-seed: `9270.7, 8510.6, 8151.6, 5694.1, 7912.8`)
  - **`Hopper-v5`:**
    - `prefix-h200`: **3098.9 ± 78.8** (med 3097.9, AUC **1854.9 ± 316.8**; per-seed: `2983.3, 3136.0, 3079.6, 3197.6, 3097.9`)
    - `prefix-h400`: **2831.5 ± 456.3** (med 2697.0, AUC 1498.3 ± 268.2; per-seed: `3291.0, 2233.1, 2645.6, 3290.9, 2697.0`)
    - `prefix-h100`: **2619.1 ± 1084.1** (med 3230.1, AUC 1861.4 ± 563.6; per-seed: `3317.8, 2249.3, 3230.1, 872.2, 3426.3`)
    - `erl-log` ($H=1000$): **1557.7 ± 811.5** (med 1785.8, AUC 765.8 ± 284.7; per-seed: `395.3, 2590.9, 1226.7, 1790.0, 1785.8`)

### Pre-registered Rule A Verdict
- $h_{\text{best}}(\text{HalfCheetah-v5}) = 200$ (`prefix-h200` mean 10142.7; beats `prefix-h100` in 5/5 seeds by +13.8%, beats `prefix-h400` in 5/5 seeds by +14.6%, beats `erl-log` in 5/5 seeds by +28.3%).
- $h_{\text{best}}(\text{Hopper-v5}) = 200$ (`prefix-h200` mean 3098.9; beats `prefix-h400` in 3/5 seeds by +9.4%, beats `prefix-h100` in 2/5 seeds by +18.3%, beats `erl-log` in 5/5 seeds by +98.9%).
- **Rule A condition:** $h_{\text{best}}(\text{env}) \ne 200$ is **FALSE in BOTH environments** ($h_{\text{best}} = 200$ everywhere).
- **Verdict: STOP.** $h=200$ zostaje domyślne. Wynik dowodzi, że optymalny horyzont prefiksu nie wymaga strojenia per-środowisko — stałe $h=200$ dominuje zarówno krótsze ($h=100$), jak i dłuższe ($h=400, 1000$) horyzonty na obu typach środowisk. **Etap B NIE startuje.**

### Offline Analysis on `erl-log`
- Na `HalfCheetah-v5`: $\rho(h)$ w pierwszych 100k krokach wynosi średnio $\rho(100)=0.803, \rho(200)=0.880, \rho(400)=0.946$. Przy $T_0=100\text{k}, \tau=0.90$, reguła wybiera $h^* \in \{100, 200, 400\}$, ale ze względu na brak wrażliwości i dominację $h=200$, kalibracja nie wnosi wartości dodanej.
- Na `Hopper-v5`: w pierwszych 100k krokach osobniki żyją średnio 50.6 kroków ($\le 100$), więc $\rho(100)=0.973, \rho(200)=0.999$, co przy $T_0=50\text{k}$ we wszystkich 5/5 seedach wskazywałoby przedwcześnie $h^*=100$, podczas gdy w pełnym treningu $h=200$ daje wyższy i znacznie stabilniejszy wynik ($3098.9 \pm 78.8$ vs $2619.1 \pm 1084.1$).


