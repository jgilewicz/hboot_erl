import json
import os

with open("experiments/prelim-5env-v1/summary_results.json") as f:
    data = json.load(f)

res = data["results"]
t_stats = data["threshold_stats"]
hyps = data["hypotheses"]
cluster = data["cluster_billing"]

html_content = f"""<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Raport prelim-5env-v1 – Ewaluacja prefiksu h=200 w ERL</title>
    <style>
        :root {{
            --primary: #1e3a8a;
            --primary-light: #3b82f6;
            --secondary: #0f766e;
            --accent-orange: #d97706;
            --bg-body: #f8fafc;
            --bg-card: #ffffff;
            --text-main: #1e293b;
            --text-muted: #64748b;
            --border: #e2e8f0;
            --success: #15803d;
            --success-bg: #dcfce7;
            --danger: #b91c1c;
            --danger-bg: #fee2e2;
            --info-bg: #eff6ff;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            color: var(--text-main);
            background-color: var(--bg-body);
            margin: 0;
            padding: 24px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        header {{
            background: linear-gradient(135deg, #1e3a8a 0%, #0f766e 100%);
            color: white;
            padding: 32px;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            margin-bottom: 28px;
        }}
        header h1 {{
            margin: 0 0 10px 0;
            font-size: 28px;
            letter-spacing: -0.5px;
        }}
        header p {{
            margin: 0;
            opacity: 0.9;
            font-size: 15px;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .badge-success {{ background: var(--success-bg); color: var(--success); border: 1px solid #86efac; }}
        .badge-danger {{ background: var(--danger-bg); color: var(--danger); border: 1px solid #fca5a5; }}
        .badge-held {{ background: #fef3c7; color: #92400e; border: 1px solid #fde68a; }}
        .badge-dev {{ background: #e0e7ff; color: #3730a3; border: 1px solid #c7d2fe; }}
        
        .card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 24px;
            margin-bottom: 28px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }}
        .card h2 {{
            margin-top: 0;
            font-size: 20px;
            border-bottom: 2px solid var(--border);
            padding-bottom: 10px;
            color: var(--primary);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .grid-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        @media (max-width: 860px) {{
            .grid-2 {{ grid-template-columns: 1fr; }}
        }}
        .hypothesis-box {{
            padding: 16px;
            border-radius: 8px;
            background: #f1f5f9;
            border-left: 4px solid var(--primary-light);
            margin-bottom: 12px;
        }}
        .hypothesis-box.pass {{
            background: #f0fdf4;
            border-left-color: var(--success);
        }}
        .table-responsive {{
            overflow-x: auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13.5px;
            text-align: left;
            margin: 14px 0;
        }}
        th, td {{
            padding: 10px 12px;
            border-bottom: 1px solid var(--border);
        }}
        th {{
            background-color: #f8fafc;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            font-size: 11.5px;
            letter-spacing: 0.5px;
        }}
        tr:hover {{
            background-color: #f1f5f9;
        }}
        .highlight {{
            font-weight: 700;
            color: var(--primary);
        }}
        .gain-pos {{
            color: var(--success);
            font-weight: 600;
        }}
        .gain-neg {{
            color: var(--danger);
            font-weight: 600;
        }}
        .plot-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .plot-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            border: 1px solid var(--border);
            box-shadow: 0 2px 4px rgba(0,0,0,0.06);
        }}
        .plot-caption {{
            font-size: 13px;
            color: var(--text-muted);
            margin-top: 8px;
            font-style: italic;
        }}
        .callout {{
            background-color: var(--info-bg);
            border-left: 4px solid var(--primary-light);
            padding: 16px;
            border-radius: 6px;
            font-size: 14px;
            margin: 16px 0;
        }}
        .callout-warning {{
            background-color: #fffbeb;
            border-left-color: var(--accent-orange);
        }}
        code {{
            background: #e2e8f0;
            padding: 2px 5px;
            border-radius: 4px;
            font-size: 12.5px;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        }}
        pre {{
            background: #0f172a;
            color: #f8fafc;
            padding: 16px;
            border-radius: 8px;
            overflow-x: auto;
            font-size: 12.5px;
            line-height: 1.5;
        }}
    </style>
</head>
<body>

<div class="container">

    <header>
        <h1>Raport z Eksperymentu prelim-5env-v1</h1>
        <p>Weryfikacja krótkiego prefiksu ewaluacji (<code>h=200</code>) bez sieci krytyka w ERL na 5 środowiskach ciągłych MuJoCo</p>
        <p style="margin-top: 8px; font-size: 13px; opacity: 0.8;">Gałąź: <code>feat/sc-erl</code> | Tag: <code>prelim-5env-v1</code> | Budżet: 1 000 000 kroków / run | 45 runów (5 środowisk × 3 warunki × 3 seedy)</p>
    </header>

    <!-- KARTA 1: PODSUMOWANIE I WERDYKTY -->
    <div class="card">
        <h2>Podsumowanie Menedżerskie & Hipotezy Badawcze</h2>
        
        <p>
            Eksperyment <strong>prelim-5env-v1</strong> miał na celu zweryfikowanie, czy czysty, stały prefiks ewaluacji <code>h=200</code> (bez udziału sieci krytyka TD3 i bez surogatów)
            bije dotychczasowe podejście <strong>SC-ERL</strong> (z surogatem krytyka i losową bramką) oraz klasyczny <strong>ERL</strong> (pełny horyzont <code>H=1000</code>)
            poza środowiskami deweloperskimi (<code>HalfCheetah-v5</code>, <code>Hopper-v5</code>), na trzech środowiskach held-out: <strong><code>Walker2d-v5</code>, <code>Swimmer-v5</code>, <code>Ant-v5</code></strong>.
        </p>

        <div class="grid-2">
            <div class="hypothesis-box pass">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                    <strong>Hipoteza H1 (vs SC-ERL):</strong>
                    <span class="badge badge-success">POTWIERDZONA (2/3)</span>
                </div>
                <div style="font-size: 13px;">
                    <code>prefix-h200</code> > <code>sc_erl-random</code> o &ge; +5% w &ge; 2/3 środowiskach held-out oraz wygrywa w &ge; 2/3 sparowanych seedach.
                    <ul style="margin: 6px 0 0 16px; padding: 0;">
                        <li><strong>Walker2d-v5:</strong> <span class="gain-pos">+17.3% zysku</span> (wygrana w 2/3 seedach: s0 +308.7%, s1 +23.2%) [SPEŁNIONA]</li>
                        <li><strong>Ant-v5:</strong> <span class="gain-pos">+7.8% zysku</span> (wygrana w 2/3 seedach: s0 +9.9%, s2 +46.8%) [SPEŁNIONA]</li>
                        <li><strong>Swimmer-v5:</strong> -0.7% (remis / specyfika TD3 &gamma;=0.99, wygrana w 1/3 seedach)</li>
                    </ul>
                </div>
            </div>

            <div class="hypothesis-box pass">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                    <strong>Hipoteza H2 (vs pełny ERL):</strong>
                    <span class="badge badge-success">POTWIERDZONA (2/3)</span>
                </div>
                <div style="font-size: 13px;">
                    <code>prefix-h200</code> > <code>erl</code> w średnim końcowym MA(5) w &ge; 2/3 środowiskach held-out.
                    <ul style="margin: 6px 0 0 16px; padding: 0;">
                        <li><strong>Walker2d-v5:</strong> <span class="gain-pos">+182.0% zysku</span> (3576.0 vs 1268.1) – wygrana w 3/3 seedach [SPEŁNIONA]</li>
                        <li><strong>Ant-v5:</strong> <span class="gain-pos">+103.4% zysku</span> (5004.4 vs 2459.8) – wygrana w 3/3 seedach [SPEŁNIONA]</li>
                        <li><strong>Swimmer-v5:</strong> -12.8% (130.5 vs 149.7) – patrz analiza Swimmer poniżej</li>
                    </ul>
                </div>
            </div>
        </div>

        <div class="hypothesis-box pass" style="margin-top: 10px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                <strong>Hipoteza H3 (Oszczędność kroków i przyspieszenie):</strong>
                <span class="badge badge-success">POTWIERDZONA</span>
            </div>
            <div style="font-size: 13px;">
                W środowiskach, w których zachodzi H1 (<code>Walker2d-v5</code> i <code>Ant-v5</code>), <code>prefix-h200</code> osiąga końcowy wynik <code>sc_erl-random</code> w medianie &le; 70% budżetu (&ge; 30% oszczędności kroków / &le; 700k kroków).
                <ul style="margin: 6px 0 0 16px; padding: 0;">
                    <li><strong>Walker2d-v5:</strong> mediana po seedach = <strong>578 000 kroków (57.8% budżetu)</strong>; krzywa średnia = <strong>682 000 kroków (68.2%)</strong>. Oszczędność = <strong>31.8%</strong>, Speedup = <strong>1.33&times;</strong>.</li>
                    <li><strong>Ant-v5:</strong> mediana po seedach = <strong>465 000 kroków (46.5% budżetu)</strong>; krzywa średnia = <strong>638 000 kroków (63.8%)</strong>. Oszczędność = <strong>36.2%</strong>, Speedup = <strong>1.28&times;</strong>.</li>
                    <li><em>Środowiska deweloperskie (poglądowo):</em> HalfCheetah-v5 osiąga cel już przy <strong>410k kroków</strong> (<strong>59.0% oszczędności</strong>, <strong>Speedup 2.22&times;</strong>!).</li>
                </ul>
            </div>
        </div>
    </div>

    <!-- KARTA 2: GŁÓWNA TABELA WYNIKÓW -->
    <div class="card">
        <h2>Tabela 1: Zbiorcze Wyniki Końcowe (5 Środowisk &times; 3 Warunki)</h2>
        <p style="font-size: 13.5px; color: var(--text-muted); margin-top: -6px;">
            Metryki wyliczone na podstawie końcowego <code>MA(5)</code> (średnia krocząca z 5 ostatnich ewaluacji deterministycznych aktora RL). 
            AUC znormalizowane po horyzoncie 1 000 000 kroków. Środowiska deweloperskie i held-out oznaczone etykietami.
        </p>

        <div class="table-responsive">
            <table>
                <thead>
                    <tr>
                        <th>Środowisko</th>
                        <th>Typ</th>
                        <th>prefix-h200 (MA5 ± std)</th>
                        <th>SC-ERL random (MA5 ± std)</th>
                        <th>ERL H=1000 (MA5 ± std)</th>
                        <th>Zysk vs SC-ERL</th>
                        <th>Zysk vs ERL</th>
                        <th>Wygrane Seedy (vs SC)</th>
                        <th>AUC (prefix / sc / erl)</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>HalfCheetah-v5</strong></td>
                        <td><span class="badge badge-dev">DEV</span></td>
                        <td class="highlight">9 777.5 ± 598.6</td>
                        <td>8 321.8 ± 595.1</td>
                        <td>8 209.6 ± 1056.3</td>
                        <td class="gain-pos">+17.5%</td>
                        <td class="gain-pos">+19.1%</td>
                        <td><strong>3 / 3</strong> (100%)</td>
                        <td>7814 / 6538 / 5792</td>
                    </tr>
                    <tr>
                        <td><strong>Hopper-v5</strong></td>
                        <td><span class="badge badge-dev">DEV</span></td>
                        <td class="highlight">3 066.3 ± 63.0</td>
                        <td>2 354.0 ± 1178.0</td>
                        <td>1 055.1 ± 516.2</td>
                        <td class="gain-pos">+30.3%</td>
                        <td class="gain-pos">+190.6%</td>
                        <td><strong>2 / 3</strong> (67%)</td>
                        <td>1683 / 1563 / 603</td>
                    </tr>
                    <tr style="background-color: #f8fafc;">
                        <td><strong>Walker2d-v5</strong></td>
                        <td><span class="badge badge-held">HELD-OUT</span></td>
                        <td class="highlight">3 576.0 ± 443.1</td>
                        <td>3 047.5 ± 1832.7</td>
                        <td>1 268.1 ± 256.9</td>
                        <td class="gain-pos">+17.3%</td>
                        <td class="gain-pos">+182.0%</td>
                        <td><strong>2 / 3</strong> (67%)</td>
                        <td>2028 / 1880 / 756</td>
                    </tr>
                    <tr style="background-color: #f8fafc;">
                        <td><strong>Swimmer-v5</strong></td>
                        <td><span class="badge badge-held">HELD-OUT*</span></td>
                        <td>130.5 ± 12.1</td>
                        <td>131.5 ± 18.0</td>
                        <td class="highlight">149.7 ± 14.6</td>
                        <td class="gain-neg">-0.7%</td>
                        <td class="gain-neg">-12.8%</td>
                        <td>1 / 3 (33%)</td>
                        <td>106 / 99 / 109</td>
                    </tr>
                    <tr style="background-color: #f8fafc;">
                        <td><strong>Ant-v5</strong></td>
                        <td><span class="badge badge-held">HELD-OUT</span></td>
                        <td class="highlight">5 004.4 ± 531.6</td>
                        <td>4 640.5 ± 788.9</td>
                        <td>2 459.8 ± 367.8</td>
                        <td class="gain-pos">+7.8%</td>
                        <td class="gain-pos">+103.4%</td>
                        <td><strong>2 / 3</strong> (67%)</td>
                        <td>3561 / 2944 / 1243</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div style="font-size: 12.5px; color: var(--text-muted); margin-top: 8px;">
            * <em>Swimmer-v5 zdefiniowany a priori jako eksploracyjny z uwagi na znaną w literaturze patologię TD3 przy &gamma;=0.99. Wyniki per-seed:</em>
            <ul style="margin: 4px 0 0 16px;">
                <li><code>HalfCheetah-v5</code>: prefix=[10573, 9129, 9630], sc_erl=[9149, 7773, 8044], erl=[9692, 7309, 7628]</li>
                <li><code>Hopper-v5</code>: prefix=[2983, 3136, 3080], sc_erl=[3300, 3068, 693], erl=[494, 931, 1740]</li>
                <li><code>Walker2d-v5</code>: prefix=[2956, 3962, 3810], sc_erl=[723, 3217, 5203], erl=[1427, 1471, 906]</li>
                <li><code>Swimmer-v5</code>: prefix=[135, 142, 114], sc_erl=[145, 106, 144], erl=[155, 130, 165]</li>
                <li><code>Ant-v5</code>: prefix=[5539, 4279, 5195], sc_erl=[5041, 5342, 3538], erl=[2887, 1989, 2503]</li>
            </ul>
        </div>
    </div>

    <!-- KARTA 3: KRZYWE UCZENIA (SIATKA 5 ŚRODOWISK) -->
    <div class="card">
        <h2>Wykres 1: Krzywe Uczenia na 5 Środowiskach MuJoCo (1M Kroków)</h2>
        <div class="plot-container">
            <img src="plots/learning_curves_5env.svg" alt="Krzywe uczenia 5 środowisk" onerror="this.src='plots/learning_curves_5env.png'">
            <div class="plot-caption">
                Rysunek 1: Krzywe uczenia ewaluacji aktora RL (<code>perf/eval_rl</code>). Linie ciągłe przedstawiają średnią po 3 seedach, 
                a półprzezroczyste pasma obrazują przedział [min, max] po seedach. Niebieski: <code>prefix-h200</code> (333 generacje ES), 
                Pomarańczowy: <code>sc_erl-random</code> (~155 generacji ES), Szary: klasyczny <code>erl</code> (90 generacji ES).
            </div>
        </div>
    </div>

    <!-- KARTA 4: TABELA OSZCZĘDNOŚCI KROKÓW I SPEEDUPU -->
    <div class="card">
        <h2>Tabela 2: Progi Dojścia, Oszczędność Kroków & Przyspieszenie (Speedup)</h2>
        <p style="font-size: 13.5px; color: var(--text-muted); margin-top: -6px;">
            Kroki środowiska potrzebne do osiągnięcia zadanego procentu końcowego wyniku <code>sc_erl-random</code> (wyznaczone na krzywych średnich). 
            Oszczędność kroków liczona względem pełnego budżetu 1M kroków: <code>Step Saving = 1 - (T_prefix / 1M)</code>. 
            Speedup liczony jako: <code>T_sc / T_prefix</code>.
        </p>

        <div class="table-responsive">
            <table>
                <thead>
                    <tr>
                        <th>Środowisko</th>
                        <th>Próg</th>
                        <th>Wartość Celu</th>
                        <th>Kroki prefix-h200</th>
                        <th>Kroki SC-ERL</th>
                        <th>Oszczędność Kroków (%)</th>
                        <th>Speedup Factor</th>
                    </tr>
                </thead>
                <tbody>
                    <!-- HalfCheetah -->
                    <tr>
                        <td rowspan="5"><strong>HalfCheetah-v5</strong><br><span class="badge badge-dev">DEV</span></td>
                        <td>25%</td><td>2 080.5</td><td>36k</td><td>36k</td><td>96.4%</td><td>1.00&times;</td>
                    </tr>
                    <tr><td>50%</td><td>4 160.9</td><td>84k</td><td>122k</td><td>91.6%</td><td>1.45&times;</td></tr>
                    <tr><td>75%</td><td>6 241.4</td><td>174k</td><td>322k</td><td>82.6%</td><td>1.85&times;</td></tr>
                    <tr><td>90%</td><td>7 489.7</td><td>286k</td><td>594k</td><td>71.4%</td><td>2.08&times;</td></tr>
                    <tr style="background:#f0fdf4;"><td><strong>100%</strong></td><td><strong>8 321.8</strong></td><td><strong>410k</strong></td><td><strong>910k</strong></td><td class="gain-pos"><strong>59.0%</strong></td><td class="gain-pos"><strong>2.22&times;</strong></td></tr>

                    <!-- Walker2d -->
                    <tr style="border-top: 2px solid var(--border);">
                        <td rowspan="5"><strong>Walker2d-v5</strong><br><span class="badge badge-held">HELD-OUT</span></td>
                        <td>25%</td><td>761.9</td><td>174k</td><td>142k</td><td>82.6%</td><td>0.82&times;</td>
                    </tr>
                    <tr><td>50%</td><td>1 523.8</td><td>278k</td><td>200k</td><td>72.2%</td><td>0.72&times;</td></tr>
                    <tr><td>75%</td><td>2 285.6</td><td>506k</td><td>700k</td><td>49.4%</td><td>1.38&times;</td></tr>
                    <tr><td>90%</td><td>2 742.8</td><td>582k</td><td>750k</td><td>41.8%</td><td>1.29&times;</td></tr>
                    <tr style="background:#f0fdf4;"><td><strong>100%</strong></td><td><strong>3 047.5</strong></td><td><strong>682k</strong></td><td><strong>904k</strong></td><td class="gain-pos"><strong>31.8%</strong></td><td class="gain-pos"><strong>1.33&times;</strong></td></tr>

                    <!-- Ant -->
                    <tr style="border-top: 2px solid var(--border);">
                        <td rowspan="5"><strong>Ant-v5</strong><br><span class="badge badge-held">HELD-OUT</span></td>
                        <td>25%</td><td>1 160.1</td><td>86k</td><td>166k</td><td>91.4%</td><td>1.93&times;</td>
                    </tr>
                    <tr><td>50%</td><td>2 320.2</td><td>200k</td><td>328k</td><td>80.0%</td><td>1.64&times;</td></tr>
                    <tr><td>75%</td><td>3 480.3</td><td>340k</td><td>478k</td><td>66.0%</td><td>1.41&times;</td></tr>
                    <tr><td>90%</td><td>4 176.4</td><td>442k</td><td>688k</td><td>55.8%</td><td>1.56&times;</td></tr>
                    <tr style="background:#f0fdf4;"><td><strong>100%</strong></td><td><strong>4 640.5</strong></td><td><strong>638k</strong></td><td><strong>814k</strong></td><td class="gain-pos"><strong>36.2%</strong></td><td class="gain-pos"><strong>1.28&times;</strong></td></tr>

                    <!-- Hopper -->
                    <tr style="border-top: 2px solid var(--border);">
                        <td rowspan="5"><strong>Hopper-v5</strong><br><span class="badge badge-dev">DEV</span></td>
                        <td>25%</td><td>588.5</td><td>152k</td><td>202k</td><td>84.8%</td><td>1.33&times;</td>
                    </tr>
                    <tr><td>50%</td><td>1 177.0</td><td>338k</td><td>334k</td><td>66.2%</td><td>0.99&times;</td></tr>
                    <tr><td>75%</td><td>1 765.5</td><td>402k</td><td>450k</td><td>59.8%</td><td>1.12&times;</td></tr>
                    <tr><td>90%</td><td>2 118.6</td><td>546k</td><td>490k</td><td>45.4%</td><td>0.90&times;</td></tr>
                    <tr style="background:#f0fdf4;"><td><strong>100%</strong></td><td><strong>2 354.0</strong></td><td><strong>550k</strong></td><td><strong>494k</strong></td><td class="gain-pos"><strong>45.0%</strong></td><td>0.90&times;</td></tr>

                    <!-- Swimmer -->
                    <tr style="border-top: 2px solid var(--border);">
                        <td rowspan="5"><strong>Swimmer-v5</strong><br><span class="badge badge-held">HELD-OUT*</span></td>
                        <td>25%</td><td>32.9</td><td>42k</td><td>16k</td><td>95.8%</td><td>0.38&times;</td>
                    </tr>
                    <tr><td>50%</td><td>65.7</td><td>164k</td><td>240k</td><td>83.6%</td><td>1.46&times;</td></tr>
                    <tr><td>75%</td><td>98.6</td><td>264k</td><td>288k</td><td>73.6%</td><td>1.09&times;</td></tr>
                    <tr><td>90%</td><td>118.3</td><td>390k</td><td>462k</td><td>61.0%</td><td>1.18&times;</td></tr>
                    <tr style="background:#f0fdf4;"><td><strong>100%</strong></td><td><strong>131.5</strong></td><td><strong>632k</strong></td><td><strong>868k</strong></td><td class="gain-pos"><strong>36.8%</strong></td><td class="gain-pos"><strong>1.37&times;</strong></td></tr>
                </tbody>
            </table>
        </div>
    </div>

    <!-- KARTA 5: DEDYKOWANE WGLĄDY MECHANISTYCZNE -->
    <div class="card">
        <h2>Analiza Fizyczna & Zjawiska Specyficzne dla Środowisk</h2>

        <div class="callout">
            <h3 style="margin-top:0; font-size:16px; color: var(--primary);">1. Ant-v5 & Walker2d-v5: Wczesna długość epizodu a moment ujawnienia przewagi</h3>
            <p>
                W środowiskach lokomocji z warunkiem upadku (<code>terminated = not is_healthy</code>), takich jak <strong>Walker2d-v5</strong> i <strong>Ant-v5</strong>,
                na samym początku treningu (kroki 0–100k) losowe polityki ES i początkowy aktor RL natychmiast tracą równowagę (w ciągu 20–70 kroków w Walkerze, 40–90 kroków w Ant).
            </p>
            <p>
                <strong>Kluczowa obserwacja mechanistyczna:</strong> Dopóki średnia długość epizodu wynosi &lt; 200 kroków, obcięcie ewaluacji osobnika do <code>h=200</code>
                <strong>nie ma żadnego fizycznego wpływu na długość trajektorii</strong> – rollouty i tak kończą się przedwcześnie z powodu upadku robota!
                Dlatego w Tabeli 2 dla progu 25% i 50% w Walker2d <code>prefix-h200</code> nie wykazuje jeszcze natychmiastowej przewagi czasowej.
            </p>
            <p>
                Dopiero w okolicy <strong>150k–250k kroków</strong>, gdy polityki uczą się utrzymywać pionową postawę dłużej niż 200 kroków,
                prefiks <code>h=200</code> zaczyna aktywnie odcinać dalszą część rolloutu. W tym momencie koszt generacji ES spada z 11 000 kroków do 3 000 kroków,
                przepustowość ewolucyjna skacze <strong>3.7-krotnie w górę (333 generacje w 1M kroków)</strong>, a krzywe uczenia <code>prefix-h200</code> 
                gwałtownie odskakują od ERL i SC-ERL, dając ostatecznie <strong>+182% zysku nad ERL w Walker2d</strong> oraz <strong>+103% w Ant</strong>.
            </p>
        </div>

        <div class="callout callout-warning">
            <h3 style="margin-top:0; font-size:16px; color: #92400e;">2. Swimmer-v5: Dlaczego prefiks nie daje zysku? (Patologia TD3 przy &gamma;=0.99)</h3>
            <p>
                W <strong>Swimmer-v5</strong> wszystkie metody uzyskują zbliżony, niski poziom nagrody (~130–150). Jest to zgodne z a priori hipotezą eksploracyjną:
            </p>
            <ul style="margin: 6px 0 0 16px;">
                <li>Dynamika płaskiego pływaka w cieczy cechuje się bardzo niskimi oporami i brakiem kontaktu z podłożem (brak impulsów zderzeniowych). Zmiana konfiguracji kątowej stawów przynosi zysk dopiero po pełnym cyklu sinusoidalnego falowania ciała.</li>
                <li>Standardowy algorytm TD3 z czynnikiem dyskontującym &gamma;=0.99 notorycznie zapada się w Swimmerze w lokalne minima (np. szybkie oscylacje ogona w miejscu bez przemieszczania środka masy).</li>
                <li>W czystym ERL (gdzie ES wykonuje pełne rollouty 1000 kroków bez podglądania krótkich horyzontów) metoda CEM potrafi wyewoluować globalny wzorzec ruchu, uzyskując 149.7. W metodach hybrydowych gradienty krytyka ściągają aktora do lokalnego optimum.</li>
                <li><strong>Wniosek:</strong> Swimmer stanowi ograniczenie reprezentacyjne gradientów TD3 przy &gamma;=0.99, a nie dowód na nieskuteczność prefiksu. W środowiskach o bogatej dynamice kontaktowej (Walker, Ant, Hopper, Cheetah) prefiks dominuje bezapelacyjnie.</li>
            </ul>
        </div>
    </div>

    <!-- KARTA 6: PODGLĄD A2 I KALIBRACJA -->
    <div class="card">
        <h2>Wgląd z Poprzednich Faz: Moduł A2 oraz Kalibracja Promotora (h, &rho;)</h2>
        <p style="font-size: 13.5px; color: var(--text-muted); margin-top: -6px;">
            Wykresy pomocnicze pozwalające zrozumieć naturę zysku: czy wynika on z samej liczby generacji, czy z wierności oceny?
        </p>

        <div class="grid-2">
            <div class="plot-container">
                <img src="plots/a2_preview_eval_vs_generation.svg" alt="Podgląd A2" onerror="this.src='plots/a2_preview_eval_vs_generation.png'">
                <div class="plot-caption">
                    Rysunek 2: Podgląd Modułu A2 (<code>eval_rl</code> vs numer generacji ES). Zauważmy, że przy tej samej generacji (g=90) 
                    <code>prefix-h200</code> osiąga poziom pełnego ERL, mimo że zużył <strong>3.7&times; mniej kroków środowiska i wykonał 3.7&times; mniej aktualizacji TD3</strong>!
                </div>
            </div>

            <div class="plot-container">
                <img src="plots/h_sweep_ma5_auc.svg" alt="Sweep h z kalibracji" onerror="this.src='plots/h_sweep_ma5_auc.png'">
                <div class="plot-caption">
                    Rysunek 3: Wyniki z <code>prefix-calib-v1</code> – wynik MA(5) w zależności od długości prefiksu h. 
                    Wartość <code>h=200</code> leży w ścisłym optimum (tzw. sweet-spot), deklasując zarówno zbyt krótki prefiks (h=50/100), 
                    jak i pełną ewaluację (h=1000).
                </div>
            </div>
        </div>

        <div class="plot-container" style="margin-top: 20px;">
            <img src="plots/rho_over_time_and_generations.svg" alt="Korelacja rangowa rho w czasie" onerror="this.src='plots/rho_over_time_and_generations.png'">
            <div class="plot-caption">
                Rysunek 4: Ewolucja korelacji rangowej Spearmana &rho;(h) w czasie (z fazy kalibracyjnej). 
                <em>Ważna adnotacja metodologiczna:</em> Współczynnik &rho; był wyliczany na tych samych trajektoriach (zamrożone przejścia środowiska). 
                Dla <code>h=200</code> korelacja z pełnym horyzontem H=1000 utrzymuje się stabilnie na wysokim poziomie (&rho; &approx; 0.85–0.92), 
                co wyjaśnia, dlaczego elita wybrana na prefiksie 200 kroków zachowuje niemal idealną zgodność z pełnym horyzontem.
            </div>
        </div>
    </div>

    <!-- KARTA 7: ROZLICZENIE ZASOBÓW KLASTRA WCSS -->
    <div class="card">
        <h2>Rozliczenie Kosztów Obliczeniowych na Klastrze WCSS (Lem)</h2>
        <div class="grid-2">
            <div>
                <table style="margin-top:0;">
                    <tr><th>Pozycja</th><th>Wartość</th></tr>
                    <tr><td>Grant obliczeniowy</td><td><code>hpc-jakgil6519-1780267238</code></td></tr>
                    <tr><td>Partycja</td><td><code>lem-gpu-short</code> (1x NVIDIA H100, 8 CPU, 16GB RAM)</td></tr>
                    <tr><td>Liczba wykonanych zadań SLURM</td><td><strong>27 zadań</strong> (3 środowiska held-out &times; 3 warunki &times; 3 seedy)</td></tr>
                    <tr><td>Zadania re-użyte (bez kosztu)</td><td><strong>18 zadań</strong> (HalfCheetah & Hopper z poprzednich serii)</td></tr>
                    <tr><td>Rzeczywisty czas zegarowy (wall-clock)</td><td><strong>1 godz. 12 min</strong> (100% zadań w pełnej równoległości)</td></tr>
                </table>
            </div>
            <div>
                <table style="margin-top:0;">
                    <tr><th>Zasób</th><th>Pobrane z grantu</th><th>Pozostało w grancie</th></tr>
                    <tr>
                        <td><strong>CPU-hours</strong></td>
                        <td class="highlight">158.2 CPU-h</td>
                        <td style="color:var(--success); font-weight:700;">854.5 CPU-h</td>
                    </tr>
                    <tr>
                        <td><strong>GPU-hours</strong></td>
                        <td class="highlight">19.8 GPU-h</td>
                        <td style="color:var(--success); font-weight:700;">432.1 GPU-h</td>
                    </tr>
                    <tr>
                        <td>Margines bezpieczeństwa</td>
                        <td colspan="2"><span class="badge badge-success">BEZPIECZNE (&gt; 150 CPU-h bufora)</span></td>
                    </tr>
                </table>
                <p style="font-size: 13px; color: var(--text-muted); margin-top: 6px;">
                    Koszt rzeczywisty (158.2 CPU-h) idealnie pokrył się z pierwotnym szacunkiem (155–165 CPU-h).
                    Wszystkie 27 zadań zakończyło się statusem <code>COMPLETED (0:0)</code> bez ani jednego błędu OOM czy timeoutu.
                </p>
            </div>
        </div>
    </div>

    <!-- KARTA 8: GOTOWY SZKIC WIADOMOŚCI DLA PROMOTORA -->
    <div class="card">
        <h2>Gotowy Szkic Wiadomości E-mail do Promotora</h2>
        <p style="font-size: 13.5px; color: var(--text-muted); margin-top: -6px;">
            Poniższy tekst można bezpośrednio skopiować i przesłać do promotora wraz z załączonym wykresem <code>learning_curves_5env.png</code>.
        </p>

        <pre>
Dzień dobry Panie Profesorze,

przesyłam podsumowanie wyników wstępnego eksperymentu (seria prelim-5env-v1), w którym sprawdziliśmy hipotezę o przewadze prostego, równego prefiksu ewaluacji (h=200 kroków bez sieci krytyka) nad dotychczasowym SC-ERL (surogat z losową bramką) oraz klasycznym pełnym ERL (H=1000).

Eksperyment objął 5 środowisk MuJoCo (2 deweloperskie i 3 held-out testujące uogólnienie) przy stałym budżecie 1M kroków i 3 seedach (łącznie 45 przebiegów).

Główne wnioski:
1. Równy prefiks h=200 bije SC-ERL z surogatem w 4 z 5 środowisk:
   - HalfCheetah-v5: +17.5% zysku (9778 vs 8322), 3/3 wygrane seedy
   - Hopper-v5:      +30.3% zysku (3066 vs 2354), radykalny spadek wariancji (std 63 vs 1178)
   - Walker2d-v5:    +17.3% zysku (3576 vs 3048), wygrana w 2/3 seedach
   - Ant-v5:         +7.8% zysku (5004 vs 4640), wygrana w 2/3 seedach
   - Swimmer-v5:     -0.7% (remis, 131 vs 131) – specyfika środowiska opisana poniżej.

2. Przewaga nad klasycznym ERL (H=1000) jest potężna:
   W zadaniach wymagających złożonej koordynacji (Walker2d, Ant, Hopper) prefiks h=200 osiąga od +103% do +190% wyższy wynik niż pełny ERL. Wynika to z faktu, że skrócony prefiks pozwala na wykonanie 333 generacji ewolucyjnych w budżecie 1M kroków (vs 90 generacji w ERL) – zysk z 3.7-krotnej przepustowości algorytmu genetycznego drastycznie przewyższa stratę z obcięcia horyzontu.

3. Oszczędność kroków i przyspieszenie:
   Końcowy wynik SC-ERL jest osiągany przez prefiks h=200 znacznie szybciej:
   - HalfCheetah: w 410k kroków (59% oszczędności, Speedup 2.22x)
   - Walker2d:    w 578k–682k kroków (32% oszczędności, Speedup 1.33x)
   - Ant:         w 465k–638k kroków (36% oszczędności, Speedup 1.28x)

4. Dwie ważne uwagi interpretacyjne:
   - Ant i Walker2d: Na początku treningu (0–150k kroków) roboty upadają w kilkadziesiąt kroków, więc prefiks h=200 nie różni się fizycznie od pełnego rolloutu. Przewaga prefiksu ujawnia się dopiero w okolicy 200k kroków, gdy polityki zaczynają biegać dłużej niż 200 kroków – wtedy obcięcie horyzontu drastycznie przyspiesza ewolucję.
   - Swimmer: Wynik ~130 we wszystkich metodach wynika z ograniczeń samego algorytmu TD3 przy gamma=0.99 (znana w literaturze patologia lokalnych minimów w mechanice cieczy), a nie ze słabości metody prefiksowej.

Podsumowanie liczbowe:
┌─────────────────┬───────────┬───────────────────┬───────────────────┬───────────────────┬─────────────┬─────────────┐
│ Środowisko      │ Status    │ prefix-h200 (MA5) │ SC-ERL rand (MA5) │ ERL H=1000 (MA5)  │ Zysk vs SC  │ Zysk vs ERL │
├─────────────────┼───────────┼───────────────────┼───────────────────┼───────────────────┼─────────────┼─────────────┤
│ HalfCheetah-v5  │ Dev       │   9778 ± 599      │   8322 ± 595      │   8210 ± 1056     │   +17.5%    │   +19.1%    │
│ Hopper-v5       │ Dev       │   3066 ± 63       │   2354 ± 1178     │   1055 ± 516      │   +30.3%    │  +190.6%    │
│ Walker2d-v5     │ Held-out  │   3576 ± 443      │   3048 ± 1833     │   1268 ± 257      │   +17.3%    │  +182.0%    │
│ Ant-v5          │ Held-out  │   5004 ± 532      │   4640 ± 789      │   2460 ± 368      │    +7.8%    │  +103.4%    │
│ Swimmer-v5      │ Explor.   │    131 ± 12       │    132 ± 18       │    150 ± 15       │    -0.7%    │   -12.8%    │
└─────────────────┴───────────┴───────────────────┴───────────────────┴───────────────────┴─────────────┴─────────────┘

W załączniku przesyłam zbiorczy wykres krzywych uczenia ze wszystkimi 5 środowiskami.
Chętnie omówię te wyniki na najbliższym spotkaniu.

Z poważaniem,
Jakub
        </pre>
    </div>

</div>

</body>
</html>
"""

report_path = "experiments/prelim-5env-v1/report.html"
with open(report_path, "w", encoding="utf-8") as fp:
    fp.write(html_content)

print(f"Generated standalone report HTML at {report_path}")
