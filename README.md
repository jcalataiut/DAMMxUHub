# LineWise — Damm x Engineering HUB Hackathon

**Line sequencing & OEE optimization for canning lines 14, 17, 19 at El Prat factory.**

Optimizes SKU sequences on canning lines to minimize total hours (production + changeovers + startup) and respect capacity constraints, using historical data to learn transition patterns and detect inefficiencies.

---

## Repository Structure

```
.
├── app.py                  # Streamlit dashboard (2 pages)
├── build_clean_data.py     # One-time pipeline: raw Excel → clean CSVs
├── ga_optimizer.py         # Genetic Algorithm optimizer + core scheduling logic
├── simulated_annealing.py  # Simulated Annealing optimizer
├── optuna_optimizer.py     # Optuna (Bayesian) optimizer (deprecated — slow)
├── data_loaders.py         # Raw Excel loading utilities
├── post_mortem.py          # Historical transition matrix builder
├── clean_data/             # Pre-computed CSVs (output of build_clean_data.py)
│   ├── demand.csv                          # Weekly demand (28 SKUs)
│   ├── frames_2025.csv                     # Cumulative transitions × 53 weeks
│   ├── nodes_2025.csv                      # Node degree per week
│   ├── black_spots_2025.csv                # Inefficient transitions (z-score > 1.5)
│   ├── changeovers.csv                     # Changeover time per (prev, next, line)
│   ├── throughput_rates.csv                # HL/hour per (SKU, line)
│   ├── sku_info.csv                        # SKU → physical format
│   ├── sku_eligibility.csv                 # SKU → eligible lines
│   ├── historical_pairs.csv                # Known (SKU, line) pairs
│   ├── historical_weeks.csv                # OEE + volume by week
│   └── params.json                         # Constants (hours/week, startup, etc.)
├── raw_data/                # Original Excel files (confidential, not committed)
├── notebooks/               # Jupyter notebooks (EDA, postmortem, exploration)
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# Create environment (Python 3.11+)
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Data Pipeline (one-time)

```bash
python build_clean_data.py
```

Reads raw Excel files from `raw_data/`, computes throughput rates, changeover matrices, SKU eligibility, weekly frames for animation, and black spot detection. Outputs all CSVs to `clean_data/`.

---

## Run the App

```bash
streamlit run app.py
```

Opens a dashboard with two visors:

### Page 1 — Aprendizaje 2025
- Animated transition graph (53 weeks)
- Nodes colored by business category: **black spots** (red), **critical hubs** (orange), **normal** (blue)
- Edge width maps to changeover time (thicker = slower change)
- Force-directed spherical layout (k=3.0) with fixed positions across all weeks
- Play button to animate the evolution week by week

### Page 2 — Optimización 2026
- Two optimizers: **Genetic Algorithm** (GA) and **Simulated Annealing** (SA)
- Side-by-side Gantt charts: baseline planner plan vs optimized plan
- Stacked hours comparison by line (production, changeover, startup, slack)
- Data table with hours saved per line
- Editable urgent orders table — add priority SKUs with extra volume
- Full historical network as faded background; optimized path in green on top

---

## Optimizers

### Genetic Algorithm (`ga_optimizer.py`)
- Population: 20–200 individuals (default 60)
- Generations: 30–400 (default 150)
- Ordered crossover + swap/migrate mutation
- Tournament selection (k=3)
- Elitism (top 4)
- Fitness: total hours + penalties for format incompatibility, capacity overflow, late priority orders

### Simulated Annealing (`simulated_annealing.py`)
- Iterations: 2K–50K (default 15K)
- Moves: 60% swap, 40% migrate
- Adaptive temperature schedule based on initial fitness
- Automatic restart if < 2% moves accepted in a window
- ~0.18s for 5K iterations (much faster than Optuna)

---

## Core Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  raw_data/      │ ──▶ │ build_clean_data │ ──▶ │  clean_data/    │
│  (Excel files)  │     │  (one-time)      │     │  (CSVs + JSON)  │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      app.py (Streamlit)                        │
│                                                                 │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────┐  │
│  │ learn_2025()  │    │  optimize_2026() │    │  visualizers  │  │
│  │ · week slider │    │ · GA / SA runner  │   │ · Bokeh graphs│  │
│  │ · play btn    │    │ · urgent orders   │   │ · Gantt       │  │
│  │ · Bokeh graph │    │ · compare tables  │   │ · stacked bars│  │
│  └──────┬───────┘    └────────┬─────────┘   └───────┬──────┘  │
│         │                    │                       │        │
│         └────────────────────┼───────────────────────┘        │
│                              │                                │
│                     ┌────────▼────────┐                       │
│                     │  ga_optimizer   │                       │
│                     │  simulated_annealing                   │
│                     └─────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

- **Clean CSV layer**: One-time pipeline decouples raw Excel from the app. The app never touches `raw_data/`.
- **Fixed graph layout**: `nx.spring_layout(k=3.0)` computed once on all weeks combined, preventing node position jitter during animation.
- **3 business categories**: Black spots (red, border), critical hubs (orange, top 30% degree), normal (blue) — replaces Louvain community detection.
- **Separate path layer**: Optimized transitions drawn as green overlay even if not in historical data.
- **Streamlit animation pattern**: State advance at bottom of render function, after all widgets are created.

---

## Urgent Orders

The editor (DataFrame with `num_rows="dynamic"`) allows adding priority SKUs:

| Column | Description |
|---|---|
| `Activa` | Enable the urgent order |
| `SKU` | Select from dropdown of known SKUs |
| `Línea` | Specific line or `Auto` (all eligible lines) |
| `HL extra` | Additional volume to schedule |
| `Posición` | Not enforced (the optimizer decides) |

Active urgent orders:
1. Add penalty if SKU appears after the first 25% of its line's sequence
2. Extra volume is added to the SKU's demand before optimization
3. A feedback table shows each order's resulting position and status (✓/✗)

---

## Visualization (Bokeh)

- **Nodes**: 3 categories — black spot (red, dark border), critical (orange, top 30% degree), normal (blue)
- **Edges**: Color + width = changeover time; green = optimized path
- **Layout**: `nx.spring_layout(k=3.0)` — spherical force-directed, 100 iterations
- **When highlighting** (2026 page): non-optimized nodes gray (alpha 0.25, size 6), non-path edges gray (alpha 0.12)

---

## Requirements

```
pandas>=2.0
bokeh>=3.0
numpy==1.26.4
openpyxl>=3.1
networkx>=3.0
plotly>=5.9
streamlit>=1.30
```

---

## Notes

- Running `build_clean_data.py` requires the raw Excel files in `raw_data/` (confidential, not in repo).
- Optuna optimizer exists but is **deprecated** — it was slower than SA and GA due to the sequential Bayesian fitting overhead for this domain.
- The `data_loaders.py` and `post_mortem.py` modules handle raw Excel ingestion and transition matrix computation and are only needed for `build_clean_data.py`.
- All app functionality reads exclusively from `clean_data/` CSVs.
