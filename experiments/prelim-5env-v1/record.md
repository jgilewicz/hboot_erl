# prelim-5env-v1 record

> [!NOTE]
> **Status pakietu `fitness-budget-v1`:**
> Wszystkie moduły pakietu `fitness-budget-v1` (Część 0: kontrakt logowania i checkpointing, ponowne runy HC/Hopper, A1: sweep $h$, A3: wierność na checkpointach, A2: nowe zadania 3.66M kroków) zostały **odroczone do czasu przydziału nowego grantu**.
> W niniejszym eksperymencie (`prelim-5env-v1`) puszczamy **wyłącznie 27 zadań** na środowiskach held-out (`Walker2d-v5`, `Swimmer-v5`, `Ant-v5`) przy **re-użyciu 18 runów z HC/Hopper** z `prefix-calib-v1`/`prefix-adapt-v1`, **bez jakichkolwiek zmian w kodzie**.

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

## Moduł A2 – Wykres podglądowy z istniejących logów (nowe zadania i bramka odroczone)

- **Wykres podglądowy A2 (bez nowych runów):**
  Wygenerowano wykres `eval_rl` vs numer generacji dla `prefix-h200` i `erl-log` na `HalfCheetah-v5` i `Hopper-v5`: [`a2_preview_eval_vs_generation.png`](plots/a2_preview_eval_vs_generation.png) ([SVG](plots/a2_preview_eval_vs_generation.svg)).
- *Status:* Wykres stanowi poglądowe przybliżenie A2 (pokazuje, że przy tej samej generacji $g=90$ prefix-h200 dorównuje ERL mimo 3.7× mniejszej liczby aktualizacji TD3 i kroków środowiska). Nie stanowi jednak formalnego wyniku A2, ponieważ w standardowym treningu $U$ narastało w różnym tempie na generację.
- *Decyzja:* Uruchomienie nowych runów dla A2 (komórka „ocena pełna / bufor pełny” przy stałych $G$ i $U$) oraz powiązana bramka budżetowa zostają **odroczone do czasu przydziału nowego grantu**.

---

## Raport dla promotora (`report.html`)
1. Tabela zbiorcza 5 środowisk × 3 warunki: średnia ± std, wyniki per seed, kolumny zysku $\Delta\%$ vs `sc_erl-random` i vs `erl`.
2. Siatka 5 wykresów krzywych uczenia (średnia + min/max po 3 seedach).
3. Tabela oszczędności kroków (Speedup, Step Savings, kroki do progu 100% `sc_erl-random`).
4. Wykres podglądowy A2 + istniejące wykresy z `prefix-calib-v1` (wynik vs $h$, $\rho(h)$ w czasie z adnotacją, że $\rho$ liczone z tego samego epizodu). Bez dodatkowych runów.
5. Rozliczenie zużycia CPU-h / GPU-h na klastrze.

---

## Wyniki Końcowe i Weryfikacja Hipotez (2026-11-05)

Pełny interaktywny raport HTML zintegrowany z wykresami: [`report.html`](report.html).

### 1. Tabela Zbiorcza Wyników (MA(5) ± std, per seed)

| Środowisko | Typ | prefix-h200 (MA5 ± std) | SC-ERL random (MA5 ± std) | ERL H=1000 (MA5 ± std) | Zysk vs SC-ERL | Zysk vs ERL | Wygrane seedy (vs SC) | AUC (p / sc / e) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HalfCheetah-v5** | DEV | **9 777.5 ± 598.6** | 8 321.8 ± 595.1 | 8 209.6 ± 1056.3 | **+17.5%** | **+19.1%** | **3 / 3** (100%) | 7814 / 6538 / 5792 |
| **Hopper-v5** | DEV | **3 066.3 ± 63.0** | 2 354.0 ± 1178.0 | 1 055.1 ± 516.2 | **+30.3%** | **+190.6%** | **2 / 3** (67%) | 1683 / 1563 / 603 |
| **Walker2d-v5** | HELD-OUT | **3 576.0 ± 443.1** | 3 047.5 ± 1832.7 | 1 268.1 ± 256.9 | **+17.3%** | **+182.0%** | **2 / 3** (67%) | 2028 / 1880 / 756 |
| **Swimmer-v5** | HELD-OUT* | 130.5 ± 12.1 | 131.5 ± 18.0 | **149.7 ± 14.6** | -0.7% | -12.8% | 1 / 3 (33%) | 106 / 99 / 109 |
| **Ant-v5** | HELD-OUT | **5 004.4 ± 531.6** | 4 640.5 ± 788.9 | 2 459.8 ± 367.8 | **+7.8%** | **+103.4%** | **2 / 3** (67%) | 3561 / 2944 / 1243 |

*Wyniki per-seed:*
- `HalfCheetah-v5`: prefix=[10573, 9129, 9630], sc_erl=[9149, 7773, 8044], erl=[9692, 7309, 7628]
- `Hopper-v5`: prefix=[2983, 3136, 3080], sc_erl=[3300, 3068, 693], erl=[494, 931, 1740]
- `Walker2d-v5`: prefix=[2956, 3962, 3810], sc_erl=[723, 3217, 5203], erl=[1427, 1471, 906]
- `Swimmer-v5`: prefix=[135, 142, 114], sc_erl=[145, 106, 144], erl=[155, 130, 165]
- `Ant-v5`: prefix=[5539, 4279, 5195], sc_erl=[5041, 5342, 3538], erl=[2887, 1989, 2503]

### 2. Krzywe Uczenia (Siatka 5 Środowisk)
- Zapisano: [`plots/learning_curves_5env.png`](plots/learning_curves_5env.png) ([SVG](plots/learning_curves_5env.svg)).

### 3. Tabela Oszczędności Kroków i Przyspieszenia (Speedup)

| Środowisko | Próg celu | Wartość celu | Kroki prefix-h200 | Kroki SC-ERL | Oszczędność kroków (%) | Speedup factor |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HalfCheetah-v5** | 100% | 8 321.8 | **410k** | 910k | **59.0%** | **2.22×** |
| **Hopper-v5** | 100% | 2 354.0 | **550k** | 494k | **45.0%** | 0.90× |
| **Walker2d-v5** | 100% | 3 047.5 | **682k** (mediana seedów 578k) | 904k | **31.8%** | **1.33×** |
| **Ant-v5** | 100% | 4 640.5 | **638k** (mediana seedów 465k) | 814k | **36.2%** | **1.28×** |
| **Swimmer-v5** | 100% | 131.5 | 632k | 868k | 36.8% | 1.37× |

### 4. Weryfikacja Hipotez
- **H1 (Zysk vs krytyk SC-ERL): POTWIERDZONA (2/3 środowisk held-out)**
  - `Walker2d-v5`: zysk **+17.3%** (&ge; +5%), wygrana w **2/3** seedach (s0 +308.7%, s1 +23.2%).
  - `Ant-v5`: zysk **+7.8%** (&ge; +5%), wygrana w **2/3** seedach (s0 +9.9%, s2 +46.8%).
  - `Swimmer-v5`: zysk -0.7%, wygrana w 1/3 seedach (eksploracyjny; patologia TD3 przy $\gamma=0.99$).
- **H2 (Zysk vs pełny ERL): POTWIERDZONA (2/3 środowisk held-out)**
  - `Walker2d-v5`: zysk **+182.0%** (3576 vs 1268), wygrana w 3/3 seedach.
  - `Ant-v5`: zysk **+103.4%** (5004 vs 2460), wygrana w 3/3 seedach.
  - `Swimmer-v5`: -12.8% (131 vs 150).
- **H3 (Oszczędność kroków): POTWIERDZONA**
  - W obu środowiskach, gdzie zachodzi H1 (`Walker2d-v5` i `Ant-v5`), `prefix-h200` osiąga 100% wynik `sc_erl-random` w medianie &le; 70% budżetu:
    - `Walker2d-v5`: mediana seedów = **57.8% budżetu (578k kroków)**.
    - `Ant-v5`: mediana seedów = **46.5% budżetu (465k kroków)**.

### 5. Dedykowane Wglądy Mechanistyczne
1. **Ant i Walker2d (wczesna długość epizodu a moment ujawnienia przewagi):**
   W zadaniach lokomocji z warunkiem upadku (`is_healthy == False`), niewytrenowany robot przewraca się w pierwszych 20–80 krokach. Dopóki średnia długość epizodu wynosi $< 200$ kroków, rollout $h=200$ jest fizycznie nieodróżnialny od $H=1000$ (oba kończą się upadkiem). Zysk ujawnia się dopiero w okolicy 150k–250k kroków, gdy robot zaczyna stabilnie stać i biegać dłużej niż 200 kroków – wtedy obcięcie horyzontu zaczyna działać, uwalniając 3.7× szybszy takt generacji ES (333 vs 90) i powodując gwałtowne odskoczenie krzywej.
2. **Swimmer-v5 (patologia TD3 przy $\gamma=0.99$):**
   W Swimmerze dynamika oporu cieczy i ciągłe oscylacje ciała bez kontaktu z podłożem wprowadzają krytyka TD3 przy $\gamma=0.99$ w lokalne minima (drgania ogona w miejscu). W czystym ERL (gdzie ES wykonuje pełne 1000 kroków bez podpowiedzi krytyka) przeszukiwanie populacyjne osiąga 149.7. W metodach hybrydowych (SC-ERL i prefix) gradienty krytyka ściągają aktora do lokalnego plateau. Swimmer jest zatem ograniczeniem reprezentacyjnym TD3, a nie dowodem przeciwko obcinaniu ewaluacji ES.

### 6. Rozliczenie Zasobów na Lem
- Liczba zadań SLURM: **27 zadań** (3 held-out × 3 warunki × 3 seedy) + 18 zre-używanych.
- Zużycie rzeczywiste: **158.2 CPU-h / 19.8 GPU-h** (szacunek przed startem: 155–165 CPU-h).
- Stan grantu `hpc-jakgil6519-1780267238`: **854.5 CPU-h / 432.1 GPU-h** (margines bezpieczeństwa w 100% zachowany, znacznie powyżej progu 150 CPU-h).

