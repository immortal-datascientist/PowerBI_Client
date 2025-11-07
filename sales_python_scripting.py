# Power BI — Add derived financial columns and produce LLM-generated insights (10 rows)
# Robust: tries batch endpoint then falls back to single generate endpoint if /batch_generate not available.
import pandas as pd, numpy as np, requests, json, re, os, time
from math import ceil

# ---------- CONFIG ----------
SERVER_BATCH = "http://127.0.0.1:7860/batch_generate"   # preferred (batch)
SERVER_SINGLE = "http://127.0.0.1:7860/generate"       # fallback single-call
HEALTH = "http://127.0.0.1:7860/health"

OUTPUT_CACHE = r"E:\AJAY\powerbi_codes\llm_suggestions_cache.json"
INSIGHTS_AUDIT_CSV = r"E:\AJAY\powerbi_codes\llm_insights_audit.csv"  # <-- audit file saved here

TIMEOUT = 600
RETRIES = 2
RETRY_WAIT = 2

MAX_CONTEXT_CHARS = 2500
MAX_TOKENS = 400
TEMPERATURE = 0.2
# Profit category thresholds (adjust if needed)
OPT_MARGIN_LOW = 0.20
OPT_MARGIN_HIGH = 0.40
# ----------------------------


# ---------- Helper Functions ----------
def safe_to_numeric(s):
    return pd.to_numeric(s, errors='coerce')

def profit_category_from_pct(pct):
    try:
        if pd.isna(pct): return "Unknown"
        if pct < 0: return "Loss"
        if pct < 10: return "Low"
        if pct < 25: return "Moderate"
        return "Healthy"
    except Exception:
        return "Unknown"

def safe_price_range(cogs):
    try:
        if pd.isna(cogs) or cogs <= 0: return (np.nan, np.nan)
        low = cogs / (1.0 - OPT_MARGIN_LOW)
        high = cogs / (1.0 - OPT_MARGIN_HIGH)
        lo, hi = sorted([low, high])
        return (lo, hi)
    except Exception:
        return (np.nan, np.nan)

def extract_json_from_text(s):
    """Try to extract a JSON structure from free text."""
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    # direct parse
    try:
        return json.loads(s)
    except Exception:
        pass
    # try find first JSON array
    m = re.search(r'(\[[\s\S]*\])', s)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # try find first JSON object
    m2 = re.search(r'(\{[\s\S]*\})', s)
    if m2:
        try:
            return json.loads(m2.group(1))
        except Exception:
            # try replace single quotes
            try:
                return json.loads(m2.group(1).replace("'", '"'))
            except Exception:
                pass
    return None

def normalize_to_rows(obj, max_rows=10):
    """Normalize parsed JSON into a list of dicts with suggestion, insight, drawback."""
    rows = []
    if obj is None:
        return rows
    if isinstance(obj, dict):
        # common wrappers
        if 'results' in obj and isinstance(obj['results'], list):
            cand = obj['results']
        elif 'items' in obj and isinstance(obj['items'], list):
            cand = obj['items']
        else:
            cand = [obj]
    elif isinstance(obj, list):
        cand = obj
    else:
        cand = [obj]

    for item in cand:
        if len(rows) >= max_rows:
            break
        if not isinstance(item, dict):
            # try parse simple k:v text -> skip non-dict here
            continue
        def get_field(d, names):
            for n in names:
                if n in d:
                    return d[n]
            # case-insensitive fallback
            for k,v in d.items():
                if isinstance(k, str) and k.lower() in [x.lower() for x in names]:
                    return v
            return ""
        s = get_field(item, ['suggestion','Suggestion','suggest'])
        i = get_field(item, ['insight','Insight','ins'])
        d = get_field(item, ['drawback','Drawback','issue','problem'])
        rows.append({
            'suggestion':("" if s is None else str(s).strip()),
            'insight':("" if i is None else str(i).strip()),
            'drawback':("" if d is None else str(d).strip())
        })
    return rows

def strip_labels_and_clean(v):
    if pd.isna(v): return ''
    v = str(v)
    v = re.sub(r'^\s*(suggestion|insight|drawback)\s*[:\-\–\—]\s*', '', v, flags=re.IGNORECASE)
    v = v.strip(' \'"{},:')
    v = re.sub(r'\s+', ' ', v).strip()
    return v

# ---------- Load Dataset ----------
df = dataset.copy()  # Power BI provides `dataset`

# Expected columns — change these strings to match your exact Excel headers if different
COL_REVENUE = "Revenue"
COL_COGS = "Actual Cost of Goods Sold"
COL_AVG_COGS_90 = "Average Historical COGS (90-Day)"
COL_AVG_REV_90 = "Average Historical Revenue (90-Day)"
COL_INSURANCE = "Insurance Name"
COL_SERVICE_DATE = "Service Date"

# Ensure numeric columns exist and are numeric
for col in [COL_REVENUE, COL_COGS, COL_AVG_COGS_90, COL_AVG_REV_90]:
    if col not in df.columns:
        df[col] = np.nan
    df[col] = safe_to_numeric(df[col])

# Service date -> Year, Month, Month_Name
if COL_SERVICE_DATE in df.columns:
    df[COL_SERVICE_DATE] = pd.to_datetime(df[COL_SERVICE_DATE], errors='coerce')
    df['Year'] = df[COL_SERVICE_DATE].dt.year.astype('Int64')
    df['Month'] = df[COL_SERVICE_DATE].dt.month.astype('Int64')
    df['Month_Name'] = df[COL_SERVICE_DATE].dt.strftime('%B').fillna('')
else:
    df['Year'], df['Month'], df['Month_Name'] = pd.NA, pd.NA, ''

# ---------- Derived Financial Columns ----------
df['Profit_Margin_pct'] = np.where(
    (df[COL_REVENUE].notna()) & (df[COL_REVENUE] != 0),
    ((df[COL_REVENUE] - df[COL_COGS]) / df[COL_REVENUE]) * 100,
    np.nan
)
df['COGS_pct_of_Revenue'] = np.where(
    (df[COL_REVENUE].notna()) & (df[COL_REVENUE] != 0),
    (df[COL_COGS] / df[COL_REVENUE]) * 100,
    np.nan
)
df['Profit_Category'] = df['Profit_Margin_pct'].apply(profit_category_from_pct)

# Outlier detection (z-score > 3)
pm = df['Profit_Margin_pct'].dropna()
if pm.empty or pm.std(ddof=0) == 0:
    df['Profit_Margin_Outlier'] = False
else:
    mean_pm = pm.mean()
    std_pm = pm.std(ddof=0)
    df['Profit_Margin_Outlier'] = df['Profit_Margin_pct'].apply(
        lambda x: False if pd.isna(x) else abs((x - mean_pm) / std_pm) > 3
    )

# Variation columns
df['Revenue_Gap_pct'] = np.where(
    (df[COL_AVG_REV_90].notna()) & (df[COL_AVG_REV_90] != 0),
    ((df[COL_REVENUE] - df[COL_AVG_REV_90]) / df[COL_AVG_REV_90]) * 100,
    np.nan
)
df['COGS_Gap_pct'] = np.where(
    (df[COL_AVG_COGS_90].notna()) & (df[COL_AVG_COGS_90] != 0),
    ((df[COL_COGS] - df[COL_AVG_COGS_90]) / df[COL_AVG_COGS_90]) * 100,
    np.nan
)

df['Revenue_Anomaly_Flag'] = df['Revenue_Gap_pct'].apply(lambda x: False if pd.isna(x) else abs(x) > 10.0)
df['High_COGS_Flag'] = df['COGS_pct_of_Revenue'].apply(lambda x: False if pd.isna(x) else x > 80.0)

max_rev = df[COL_REVENUE].max(skipna=True)
df['Profitability_Score'] = np.where(
    (pd.notna(df['Profit_Margin_pct'])) & (pd.notna(max_rev)) & (max_rev != 0),
    df['Profit_Margin_pct'] * (df[COL_REVENUE] / max_rev),
    np.nan
)

# Payer ROI (per-payer average margin * payer revenue share)
if COL_INSURANCE in df.columns:
    payer_avg_margin = df.groupby(COL_INSURANCE)['Profit_Margin_pct'].mean()
    total_revenue = df[COL_REVENUE].sum(skipna=True)
    df['Payer_Revenue_Share'] = np.where(total_revenue and not pd.isna(total_revenue),
                                         df[COL_REVENUE] / total_revenue, 0.0)
    df['_Payer_Avg_Margin'] = df[COL_INSURANCE].map(payer_avg_margin).astype(float)
    df['Payer_ROI'] = df['_Payer_Avg_Margin'] * df['Payer_Revenue_Share']
    df = df.drop(columns=['_Payer_Avg_Margin'])
else:
    df['Payer_Revenue_Share'], df['Payer_ROI'] = 0.0, np.nan

# Optimal Price Range (approx)
opt_ranges = df[COL_COGS].apply(safe_price_range)
df['Optimal_Price_Min'] = opt_ranges.apply(lambda x: x[0])
df['Optimal_Price_Max'] = opt_ranges.apply(lambda x: x[1])

# ---------- LLM Insights (10 rows) ----------
def build_context_for_llm(dframe, max_chars=MAX_CONTEXT_CHARS):
    parts = []
    anomalies = dframe[(dframe['Revenue_Anomaly_Flag']==True) | (dframe['High_COGS_Flag']==True)]
    for _, r in anomalies.head(8).iterrows():
        parts.append(f"Anomaly: Rev={r.get(COL_REVENUE,'')}, COGS={r.get(COL_COGS,'')}, Payer={r.get(COL_INSURANCE,'')}")
    top = dframe.sort_values('Profitability_Score', ascending=False).head(6)
    for _, r in top.iterrows():
        parts.append(f"Top: Rev={r.get(COL_REVENUE,'')}, Profit%={r.get('Profit_Margin_pct','')}, Payer={r.get(COL_INSURANCE,'')}")
    joined = " ||| ".join([str(x) for x in parts])
    return joined[:max_chars] + ("..." if len(joined) > max_chars else "")

context_snip = build_context_for_llm(df)
empty_rows = [{'suggestion':'','insight':'','drawback':''} for _ in range(10)]
insights_table = pd.DataFrame(empty_rows, columns=['suggestion','insight','drawback'])

# check server health
server_up = False
try:
    hh = requests.get(HEALTH, timeout=5)
    server_up = (hh.status_code == 200)
except Exception:
    server_up = False

if server_up:
    PROMPT = (
        "You are an expert financial analyst. Using the dataset context below, generate EXACTLY 10 distinct items. "
        "Return ONLY a JSON array of 10 objects. Each object must include exactly these keys: "
        "\"suggestion\",\"insight\",\"drawback\" (lowercase). Each value: plain sentence <= 18 words. Use numbers (%, counts, currency) when relevant. "
        "Do not include any text outside the JSON array.\n\n"
        f"Dataset context: {context_snip}\n\nReturn JSON array now."
    )

    # Try batch endpoint first (many servers support /batch_generate)
    body = None
    last_err = None

    # Build payload for batch endpoint (items list)
    batch_payload = {"items": [{"prompt": PROMPT, "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE}]}

    try_batch = True
    try:
        r = requests.post(SERVER_BATCH, json=batch_payload, timeout=TIMEOUT)
        # If 404 or not OK, we'll fallback
        if r.status_code == 200:
            try:
                body = r.json()
            except Exception:
                body = r.text
        else:
            # Not 200 -> fall back to single
            try_batch = False
            last_err = f"batch status {r.status_code}"
    except requests.exceptions.RequestException as e:
        try_batch = False
        last_err = str(e)

    # If batch failed or not available, try single /generate endpoint
    if not try_batch:
        for attempt in range(1, RETRIES+1):
            try:
                r2 = requests.post(SERVER_SINGLE, json={"prompt": PROMPT, "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE}, timeout=TIMEOUT)
                if r2.status_code == 200:
                    try:
                        body = r2.json()
                    except Exception:
                        body = r2.text
                    break
                else:
                    last_err = f"single status {r2.status_code}"
            except requests.exceptions.RequestException as e:
                last_err = str(e)
                time.sleep(RETRY_WAIT)

    # If we have a response body, attempt to parse
    parsed_rows = []
    if body is not None:
        # body might be dict with 'results' or 'text' or similar;
        # unify to textual content then try to extract JSON
        candidate_texts = []

        if isinstance(body, dict):
            # collect likely text fields
            for k in ('results','text','output','raw','choices'):
                if k in body and body[k]:
                    candidate_texts.append(body[k])
            # if results is present and is a list with text per item, collect them
            if 'results' in body and isinstance(body['results'], list):
                # try to extract first result text
                for it in body['results']:
                    if isinstance(it, dict) and 'text' in it and it['text']:
                        candidate_texts.append(it['text'])
                    elif isinstance(it, str):
                        candidate_texts.append(it)
            # fallback to entire serialized body
            if not candidate_texts:
                candidate_texts.append(json.dumps(body, ensure_ascii=False))
        elif isinstance(body, list):
            # server returned list - convert to text candidates
            candidate_texts.append(json.dumps(body, ensure_ascii=False))
        else:
            candidate_texts.append(str(body))

        extracted = None
        for ct in candidate_texts:
            # if choices list -> handle specially
            if isinstance(ct, list) and len(ct) > 0:
                # try first element
                first = ct[0]
                if isinstance(first, dict) and 'text' in first:
                    ct = first['text']
                else:
                    ct = json.dumps(ct, ensure_ascii=False)
            # if it's a dict->stringify
            if isinstance(ct, dict):
                ct = json.dumps(ct, ensure_ascii=False)
            # attempt to get JSON from ct
            extracted = extract_json_from_text(str(ct))
            if extracted is not None:
                break

        # normalize into rows
        if extracted is not None:
            parsed_rows = normalize_to_rows(extracted, max_rows=10)
        else:
            # final fallback: heuristics on full textual body (split lines into triplets)
            full_text = " ".join([str(x) for x in candidate_texts])[:10000]
            lines = [ln.strip() for ln in re.split(r'[\r\n]+', full_text) if ln.strip()]
            heur = []
            i = 0
            while i < len(lines) and len(heur) < 10:
                s = lines[i] if i < len(lines) else ""
                i += 1
                ins = lines[i] if i < len(lines) else ""
                i += 1
                d = lines[i] if i < len(lines) else ""
                i += 1
                heur.append({'suggestion': strip_labels_and_clean(s), 'insight': strip_labels_and_clean(ins), 'drawback': strip_labels_and_clean(d)})
            parsed_rows = heur

    # If parsed_rows valid, ensure length 10 and convert to DataFrame
    if parsed_rows and isinstance(parsed_rows, list):
        # convert non-dict entries to dicts if needed
        final = []
        for it in parsed_rows[:10]:
            if not isinstance(it, dict):
                continue
            s = strip_labels_and_clean(it.get('suggestion',''))
            ins = strip_labels_and_clean(it.get('insight',''))
            d = strip_labels_and_clean(it.get('drawback',''))
            final.append({'suggestion': s, 'insight': ins, 'drawback': d})
        # pad to 10
        while len(final) < 10:
            final.append({'suggestion':'','insight':'','drawback':''})
        insights_table = pd.DataFrame(final[:10], columns=['suggestion','insight','drawback'])
    else:
        # keep empty default or cached body for debugging
        try:
            with open(OUTPUT_CACHE, 'w', encoding='utf-8') as fh:
                fh.write(json.dumps(body, ensure_ascii=False, indent=2))
        except Exception:
            pass
        # insights_table remains empty rows

# ---------- Persist insights_table for audit ----------
try:
    insights_table.to_csv(INSIGHTS_AUDIT_CSV, index=False, encoding='utf-8')
except Exception as e:
    # avoid failure to Power BI UI; just print
    print("⚠️ Could not save audit CSV:", e)

# ---------- Final Outputs ----------
dataset = df
insights_table = insights_table.reset_index(drop=True)
