# prelim-5env-v1 record

## Plan & Pre-registration (2026-11-04)

- **Cel:** Wstępne porównanie poglądowe dla promotora (nie do pracy): czy równy krótki prefiks $h=200$ bez krytyka bije SC-ERL (bramka losowa) i pełny ERL ($H=1000$) poza środowiskami deweloperskimi (`HalfCheetah-v5`, `Hopper-v5`), na trzech środowiskach held-out (`Walker2d-v5`, `Swimmer-v5`, `Ant-v5`).
- **Gałąź i tag:** `feat/sc-erl`, tag W&B `prelim-5env-v1`, budżet `1_000_000` `env_steps` na run (wszystkie kroki liczone do budżetu). Konfiguracja identyczna jak w `prefix-calib-v1`, bez dostrajania per środowisko.

### Środowiska i warunki
- **Środowiska:**
  - Deweloperskie (raportowane osobno, nie liczą się do reguł held-out): `HalfCheetah-v5`, `Hopper-v5`
  - Held-out (test uogólnienia): `Walker2d-v5`, `Swimmer-v5` *(eksploracyjne, wrażliwość TD3 na $\gamma=0.99$)*, `Ant-v5`
- **Warunki:**
  1. `erl`: pełny horyzont $H=1000$ (90 generacji ES w 1M kroków).
  2. `sc_erl-random`: surogat krytyka pod losową bramką (koszt $H=1000$ na generację).
  3. `prefix-h200`: stały krótki prefiks $h=200$ bez użycia sieci krytyka (333 generacje ES w 1M kroków).
- **Seedy:** `0, 1, 2` (3 seedy na warunek i środowisko = 15 runów na środowisko, łącznie 45 runów w macierzy 5×3×3).

### Re-użycie istniejących przebiegów (18 runów)
- Dla `HalfCheetah-v5` i `Hopper-v5`:
  - `prefix-h200`: seedy 0–2 z `prefix-calib-v1` / `prefix-v1` / `prefix-adapt-v1`.
  - `sc_erl-random`: seedy 0–2 z istniejących przebiegów bazowych (`prefix-v1` / `prefix-adapt-v1`, commit `533d9b5e` i `5378efc2`, ścieżka ewaluacji identyczna).
  - `erl`: seedy 0–2 z `prefix-adapt-v1` oraz `erl-log` z `prefix-calib-v1`.
- Wszystkie 18 runów zostaje otagowane `prelim-5env-v1`.

### Nowe zadania SLURM (27 zadań)
- `Walker2d-v5`: `erl`, `sc_erl-random`, `prefix-h200` (seedy 0–2 = 9 zadań).
- `Swimmer-v5`: `erl`, `sc_erl-random`, `prefix-h200` (seedy 0–2 = 9 zadań).
- `Ant-v5`: `erl`, `sc_erl-random`, `prefix-h200` (seedy 0–2 = 9 zadań).
- **Zadania w `slurm_run_array.sh`:**
  - `erl` (indeks 4): task IDs `20, 21, 22`
  - `sc_erl-random` (indeks 5): task IDs `25, 26, 27`
  - `prefix-h200` (indeks 12): task IDs `60, 61, 62`
  - Wywołanie: `--array=20-22,25-27,60-62` dla każdego środowiska.

### Szacunek zasobów i czasu przed startem
- **Partycja:** `lem-gpu-short`, zasoby: `-c 8 --mem=16gb --gres=gpu:hopper:1` (1x NVIDIA H100).
- **Zmierzone empirycznie czasy na Lem:**
  - `erl` (~17 min/task): 9 zadań = ~153 min = **~20.4 CPU-h / ~2.6 GPU-h**
  - `prefix-h200` (~45 min/task): 9 zadań = ~405 min = **~54.0 CPU-h / ~6.8 GPU-h**
  - `sc_erl-random` (~62 min/task): 9 zadań = ~558 min = **~74.4 CPU-h / ~9.3 GPU-h**
  - **Łączny koszt (+ margines na symulację Ant 3D):** **~155 – 165 CPU-h / ~19 – 21 GPU-h**.
- **Czas zegarowy (wall-clock):** limit grantu to 32 GPU jednocześnie; przy 27 zadaniach wszystkie startują w 100% równolegle. Rzeczywisty czas oczekiwania: **~1h 10m**.
- **Stan grantu `hpc-jakgil6519-1780267238` przed startem:**
  - Pozostało: `1012.7 CPU-h / 452.9 GPU-h`.
  - Po wykonaniu pozostanie: `~850 CPU-h / ~432 GPU-h`.

---

## Metryki Główne, Zysku i Oszczędności

### 1. Metryki główne (per środowisko, per seed)
- Końcowe `MA(5)` z `perf/eval_rl` (średnia krocząca z 5 ostatnich ewaluacji).
- AUC znormalizowanej krzywej uczenia po `env_steps`.
- Kroki środowiska ($T_{\text{threshold}}$) do osiągnięcia poziomu końcowego `sc_erl-random` oraz progi 25%, 50%, 75%, 90%, 100%.
- Średnia długość epizodu populacji w czasie (`prefix_log/ep_len_mean`).

### 2. Metryki Zysku (Performance Gain)
- Względny zysk punktowy vs `sc_erl-random`:
  $$\text{Gain}_{\text{sc\_erl}} = \frac{\text{MA}(5)_{\text{prefix-h200}} - \text{MA}(5)_{\text{sc\_erl-random}}}{\max(|\text{MA}(5)_{\text{sc\_erl-random}}|, 1.0)} \times 100\%$$
- Względny zysk punktowy vs pełny `erl`:
  $$\text{Gain}_{\text{erl}} = \frac{\text{MA}(5)_{\text{prefix-h200}} - \text{MA}(5)_{\text{erl}}}{\max(|\text{MA}(5)_{\text{erl}}|, 1.0)} \times 100\%$$

### 3. Metryki Oszczędności Kroków i Przyspieszenia (Step Savings & Speedup)
- Procentowa oszczędność kroków środowiska do osiągnięcia końcowego wyniku `sc_erl-random`:
  $$\text{Step Saving} = 1 - \frac{T_{\text{prefix-h200}}}{1\,000\,000}$$
  $$\text{Relative Step Saving} = 1 - \frac{T_{\text{prefix-h200}}}{T_{\text{sc\_erl-random}}}$$
- Przyspieszenie (Speedup factor):
  $$\text{Speedup} = \frac{T_{\text{sc\_erl-random}}}{T_{\text{prefix-h200}}}$$
- Przepustowość iteracji algorytmu ewolucyjnego (ES Throughput Multiplier):
  - Koszt ewaluacji osobnika: $real\_frac = 0.2$ (200 kroków vs 1000 kroków $\to$ **80% mniej kroków na ewaluację osobnika**).
  - W stałym budżecie 1M kroków: **333 generacje ES** dla `prefix-h200` vs **90 generacji** dla `erl` $\implies$ **$3.7\times$ więcej generacji algorytmu ewolucyjnego**.

---

## Hipotezy (zamrożone przed startem)

- **H1 (Zysk vs krytyk SC-ERL):** `prefix-h200` > `sc_erl-random` w średnim końcowym `MA(5)` w $\ge 2/3$ środowiskach held-out (`Walker2d`, `Ant`, `Swimmer`), ze średnią różnicą $\ge +5\%$ oraz wygraną w $\ge 2/3$ sparowanych seedach.
- **H2 (Zysk vs pełny ERL):** `prefix-h200` > `erl` w średnim końcowym `MA(5)` w $\ge 2/3$ środowiskach held-out.
- **H3 (Oszczędność kroków):** W środowiskach, w których zachodzi H1, `prefix-h200` osiąga końcowy wynik `sc_erl-random` w medianie $\le 70\%$ budżetu ($\ge 30\%$ oszczędności kroków / $T \le 700\text{k}$).
- *Uwagi:* Wyniki `HalfCheetah-v5` i `Hopper-v5` raportowane są poglądowo obok i nie wliczają się do weryfikacji H1–H3. `Swimmer-v5` oznaczony jako eksploracyjny.

---

## Moduł A2 – Zmiana zakresu i bramka budżetowa

1. **Zakres A2 (jedna komórka zamiast 2×2):**
   - Uruchamiamy wyłącznie komórkę **„ocena pełna / bufor pełny”** ($H=1000$) przy stałej liczbie generacji $G$ i aktualizacji TD3 $U$ pobranych z referencyjnego `prefix-h200` (dla $G=333$, $U \approx 10^6$).
   - Środowiska: `HalfCheetah-v5`, `Hopper-v5`, `Walker2d-v5` (seedy 0–2 $\implies$ 9 zadań).
   - Pozostałe komórki siatki 2×2 („ocena h200 / bufor pełny”, „ocena pełna / bufor h200”) zostają **odroczone (nie anulowane)**.
2. **Warunek bramkowy startu A2:**
   - A2 startuje ściśle po M, A1 i A3, **tylko jeśli saldo grantu po nich wynosi $\ge 308\text{ CPU-h}$** (szacowany koszt A2: 9 zadań $\times \sim 17.5\text{ CPU-h} \approx 158\text{ CPU-h} + 150\text{ CPU-h}$ rezerwy minimalnej).
3. **Przybliżenie A2 z istniejących logów (bez nowych runów):**
   - Wygenerowano wykres `eval_rl` vs numer generacji: [`a2_preview_eval_vs_generation.png`](plots/a2_preview_eval_vs_generation.png) ([SVG](plots/a2_preview_eval_vs_generation.svg)).
   - *Status:* Wykres stanowi poglądowe przybliżenie A2 (pokazuje, że przy tej samej generacji $g=90$ prefix-h200 dorównuje ERL mimo 3.7× mniejszej liczby aktualizacji TD3 i kroków środowiska). Nie stanowi jednak formalnego wyniku A2, ponieważ w standardowym treningu $U$ narastało w różnym tempie na generację.
4. **Hipoteza H7 (bez zmian):**
   - Oceniana na pojedynczej komórce: różnica (ocena pełna − ocena h200) przy równych $G$ i $U$. Jeśli efekt oceny $< 5\%$ w $\ge 2/3$ środowisk, potwierdza to, że zysk prefiksu wynika z liczby generacji ewolucyjnych, a nie ze specyfiki skróconego horyzontu.

---

## Raport dla promotora (`report.html`)
1. Tabela zbiorcza 5 środowisk × 3 warunki: średnia ± std, wyniki per seed, kolumny zysku $\Delta\%$ vs `sc_erl-random` i vs `erl`.
2. Siatka 5 wykresów krzywych uczenia (średnia + min/max po 3 seedach).
3. Tabela oszczędności kroków (Speedup, Step Savings, kroki do progu 100% `sc_erl-random`).
4. Wykres i analiza A2 (przybliżenie z istniejących danych oraz wyniki po ewentualnym przejściu bramki budżetowej).
5. Rozliczenie zużycia CPU-h / GPU-h na klastrze.

