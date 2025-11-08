# PowerBI_Client

Here’s a polished, “what-it-is / what-it-uses” README you can drop into the repo. I focused on describing the project, its structure, the artifacts inside, dependencies, and technologies—without setup steps, as requested.

---

# PowerBI_Client

A hybrid analytics workspace combining **Power BI** artifacts (PBIX report and data files) with **Python** and **notebooks** to explore, prepare, and enrich business data. The repository includes an end-to-end slice: raw inputs (CSV/XLSX), a curated Power BI report, and supporting Python utilities—including a local **GGUF LLM** helper for auto-generating insights and suggestions.

> At a glance: PBIX + CSV/XLSX data + Python scripts (`sales_python_scripting.py`, `llm_server_gguf.py`) and a `notebook/` folder. Languages reported by GitHub: ~81% Jupyter Notebook, ~19% Python. ([GitHub][1])

---

## What this project is

* **Self-contained BI sample**: Demonstrates how customer feedback and sales-related data can be shaped and visualized in **Power BI Desktop** (`AQCCA_overall_experience.pbix`) with companion raw/derived files (`AQCCA_Customer_Feedback.csv`, `Top4_suggestions.csv`, `Growth LLM Generated.xlsx`). ([GitHub][1])
* **Data prep + automation**: Python scripts are included for data wrangling/exports and for spinning up a lightweight **local LLM (GGUF)** helper that can generate growth ideas or narrative insights which are then saved to spreadsheets/CSVs for BI consumption. (See the script names: `sales_python_scripting.py` and `llm_server_gguf.py`.) ([GitHub][1])
* **Notebook workspace**: A `notebook/` directory is present for exploration/EDA, modeling notes, or ad-hoc transformations before handing data off to Power BI. ([GitHub][1])

---

## Repository structure

```
PowerBI_Client/
├─ AQCCA_overall_experience.pbix         # Power BI report (overall customer experience)
├─ AQCCA_Customer_Feedback.csv           # Source/input data (customer feedback)
├─ Top4_suggestions.csv                  # Derived output (e.g., LLM- or analysis-generated suggestions)
├─ Growth LLM Generated.xlsx             # Derived insights/ideas saved to Excel
├─ sales_python_scripting.py             # Data prep / export logic for BI-ready tables
├─ llm_server_gguf.py                    # Local GGUF LLM helper to generate insights
└─ notebook/                             # Jupyter notebooks for EDA and prototyping
```

> File names and presence confirmed via the GitHub file listing. ([GitHub][1])

---

## Key artifacts

* **Power BI report**: `AQCCA_overall_experience.pbix` — a compiled report ready to open in **Power BI Desktop** for interactive analysis and dashboarding. ([GitHub][1])
* **Raw / curated data**:

  * `AQCCA_Customer_Feedback.csv` – likely the primary input table.
  * `Top4_suggestions.csv` – concise recommendation output (e.g., top actions).
  * `Growth LLM Generated.xlsx` – insight sheet produced via the LLM helper. ([GitHub][1])
* **Python utilities**:

  * `sales_python_scripting.py` – scripting for cleaning/aggregating/exporting sales or feedback data for BI.
  * `llm_server_gguf.py` – a small local service/script to run a **GGUF-format** model for suggestion generation; outputs land in CSV/XLSX for consumption in Power BI. ([GitHub][1])
* **Notebooks**: `notebook/` – for reproducible EDA, quick experiments, and documentation of transformations. ([GitHub][1])

---

## Dependencies

> The repo doesn’t include a lockfile/requirements list. Below are the **practical** dependencies implied by the files and typical usage patterns—organized by layer.

### Business Intelligence

* **Power BI Desktop** — to open and explore `AQCCA_overall_experience.pbix`. ([GitHub][1])

### Python & Data (implied by scripts and data files)

* **Python 3.x**
* **Data wrangling & I/O**

  * `pandas` (CSV/Excel read–write, transforms)
  * `openpyxl` (Excel I/O)
  * `numpy` (common numeric helpers)
* **LLM (GGUF) helper**

  * `llama-cpp-python` (run local GGUF models referenced by `llm_server_gguf.py`)

> Notes: These libraries are standard for the workflows suggested by the script names and file types; exact versions aren’t specified in the repo. If you want me to pin versions, share your current environment or let me read the scripts’ imports directly. ([GitHub][1])

### Notebooks

* **Jupyter** (to run anything under `notebook/`). GitHub language stats indicate a heavy presence of notebooks. ([GitHub][1])

---

## Technologies used

* **Power BI Desktop** — reporting, modeling, measures, interactive visuals, and dashboards.
* **Python (pandas/numpy)** — data preparation and exports to CSV/XLSX for downstream BI.
* **Local LLM (GGUF via llama-cpp)** — generates narrative insights or growth suggestions that feed into BI artifacts.
* **Jupyter Notebooks** — exploration, EDA, and documentation of transformations.

> Tech stack summarized from the repository contents and language breakdown. ([GitHub][1])

---

## Project scope & typical flow

1. **Ingest & Explore**: Start with CSV feedback (`AQCCA_Customer_Feedback.csv`) or other inputs in notebooks for quick EDA. ([GitHub][1])
2. **Prepare Data (Python)**: Use `sales_python_scripting.py` (and notebooks) to create clean, BI-ready tables; export to CSV/XLSX. ([GitHub][1])
3. **Augment with Insights**: Run `llm_server_gguf.py` to produce suggestions/insight sheets (e.g., `Top4_suggestions.csv`, `Growth LLM Generated.xlsx`). ([GitHub][1])
4. **Visualize in Power BI**: Load curated tables into `AQCCA_overall_experience.pbix` to analyze overall customer experience and related KPIs. ([GitHub][1])

---

## Status, license, and housekeeping

* **Commits / Activity**: The repository currently shows a small number of commits and early-stage structure. ([GitHub][1])
* **License**: No license file is present at the root as of **Nov 7, 2025**. If you plan to open-source, consider adding a LICENSE. ([GitHub][1])

---

## Credits

* Repository: **immortal-datascientist / PowerBI_Client** (public). ([GitHub][1])

---

If you want, I can also:

* scan the Python scripts to extract **exact** import lists (so the dependency section is fully pinned),
* inventory the `notebook/` folder and summarize each notebook’s purpose, and
* add badges/sections (e.g., data dictionary, KPIs covered, measures glossary) tailored to the PBIX.

[1]: https://github.com/immortal-datascientist/PowerBI_Client "GitHub - immortal-datascientist/PowerBI_Client"
