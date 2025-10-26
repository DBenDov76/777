
from flask import Flask, request, jsonify, send_from_directory
import pandas as pd, numpy as np, math, itertools, datetime, random, os, io, threading, webbrowser

app = Flask(__name__)

ALL = list(range(1,71))
PRIMES = set([2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67])
MULT10 = set([10,20,30,40,50,60,70])

def row_to_numbers(row):
    vals = [int(x) for x in row.dropna().tolist() if isinstance(x,(int,float)) and not math.isnan(x)]
    vals = [n for n in vals if 1 <= n <= 70]
    return sorted(set(vals))

def parse_history_from_bytes(content: bytes):
    df = None
    for sep in [',',';','\\s+','\\t',None]:
        try:
            d = pd.read_csv(io.BytesIO(content), header=None, sep=sep, engine='python')
            d = d.apply(pd.to_numeric, errors='coerce')
        except Exception:
            d = None
        if d is not None and d.notna().sum().sum() >= 50:
            df = d; break
    if df is None:
        raise ValueError('Failed to parse CSV (need numeric grid of 1..70).')
    df = df.dropna(axis=1, how='all')
    rows = [row_to_numbers(df.iloc[i,:]) for i in range(len(df))]
    rows = [r for r in rows if len(r) >= 12]
    if len(rows) > 2000: rows = rows[:2000]
    return rows

def chronological(rows_file_first_is_latest=True, rows=None):
    if rows_file_first_is_latest:
        return list(reversed(rows))
    return rows

def zone_id(n): return (n-1)//10

def count_in_window(window):
    c = {n:0 for n in ALL}
    for r in window:
        for n in r: c[n]+=1
    return c

# ===== scoring logic (auto-blend as before) =====
def build_scores(hist, t, K=60, lam=0.05):
    K = min(K, len(hist[:t]))
    recent = hist[:t][-K:]
    cK = count_in_window(recent)
    max_cK = max(cK.values()) if cK else 1
    rec = {n:0.0 for n in ALL}
    for i,row in enumerate(recent):
        age = (K-1 - i)
        w = math.exp(-lam*age)
        for n in row: rec[n] += w
    return {n: rec[n] + 0.3*(cK[n]/max(1,max_cK)) for n in ALL}

def recent_diversity(hist, t, window=20):
    start = max(0, t-window)
    seen = set()
    for r in hist[start:t]:
        seen.update(r)
    return len(seen)

def scores_two_stage_ewma(hist, t):
    last20 = hist[max(0,t-20):t]
    prev80 = hist[max(0,t-100):max(0,t-20)]
    rec = {n:0.0 for n in ALL}
    for i,row in enumerate(reversed(last20)):
        w = math.exp(-0.22*i)
        for n in row: rec[n]+=w
    for i,row in enumerate(reversed(prev80)):
        w = math.exp(-0.05*i)*0.6
        for n in row: rec[n]+=w
    last50 = hist[max(0,t-50):t]
    c50 = count_in_window(last50)
    max50 = max(c50.values()) if c50 else 1
    return {n: rec[n] + 0.25*(c50[n]/max(1,max50)) for n in ALL}

def scores_dynamic_K(hist, t):
    div = recent_diversity(hist, t, 20)
    if div < 100:
        K, lam = 45, 0.08
    elif div > 120:
        K, lam = 90, 0.04
    else:
        K, lam = 60, 0.05
    return build_scores(hist, t, K=K, lam=lam)

def scores_auto_blend(hist, t):
    base = build_scores(hist, t, K=60, lam=0.05)
    ewma = scores_two_stage_ewma(hist, t)
    dynK = scores_dynamic_K(hist, t)
    div = recent_diversity(hist, t, 20)
    if div < 100:
        w_ewma, w_dyn, w_base = 0.6, 0.25, 0.15
    elif div > 120:
        w_ewma, w_dyn, w_base = 0.25, 0.6, 0.15
    else:
        w_ewma, w_dyn, w_base = 0.45, 0.45, 0.10
    scores = {n: w_ewma*ewma[n] + w_dyn*dynK[n] + w_base*base[n] for n in ALL}
    return scores

def apply_strict_recent_exclusions(scores, hist, t):
    # Strict rules requested:
    # - If a number appeared >=2 in last 3 games, exclude.
    # - If a number appeared >=3 in last 5 games, exclude.
    last3 = hist[max(0, t-3):t]
    last5 = hist[max(0, t-5):t]
    c3 = count_in_window(last3)
    c5 = count_in_window(last5)
    for n in ALL:
        if c3[n] >= 2 or c5[n] >= 3:
            scores[n] = -999
    return scores

def sample17_excluding_rules(hist, t, M=50, seed=42):
    scores = scores_auto_blend(hist, t)
    scores = apply_strict_recent_exclusions(scores, hist, t)

    order_all = sorted(ALL, key=lambda n: scores[n], reverse=True)

    # Also keep the old rule: remove numbers repeated in BOTH last two games
    excl_last2 = set()
    if t>=2:
        excl_last2 = set(hist[t-1]) & set(hist[t-2])

    # Pool from Top-M minus last2 excl
    pool = [n for n in order_all[:M] if n not in excl_last2 and scores[n] > -999]
    # Backfill from the rest if needed
    if len(pool) < 17:
        for n in order_all[M:]:
            if n not in excl_last2 and scores[n] > -999 and n not in pool:
                pool.append(n)
            if len(pool) >= 17:
                break
    # Final fallback to ensure 17 (avoid dead-ends)
    if len(pool) < 17:
        for n in ALL:
            if n not in excl_last2 and scores[n] > -999 and n not in pool:
                pool.append(n)
            if len(pool) >= 17:
                break

    rnd = random.Random(seed + t*137)
    s17 = sorted(rnd.sample(sorted(set(pool)), 17))
    return s17, sorted(excl_last2)

def has_any_consecutive(c7):
    s = sorted(c7)
    for i in range(1, len(s)):
        if s[i] == s[i-1] + 1:
            return True
    return False

def combo_filter_and_score(s17, hist, t, K=60, lam=0.05):
    recent = hist[max(0,t-50):t]
    c10 = count_in_window(recent[-10:]) if len(recent) >= 10 else count_in_window(recent)

    def combo_ok(c7):
        # Keep your original strict filters
        if sum(1 for x in c7 if x in MULT10) > 1:
            return False
        if has_any_consecutive(c7):
            return False
        ev = sum(1 for x in c7 if x%2==0); od = 7-ev
        if ev > 4 or od > 4:
            return False
        cold = sum(1 for x in c7 if c10.get(x,0)==0)
        if not (0 <= cold <= 3):
            return False
        return True

    # Use the same score map as selection
    scores = scores_auto_blend(hist, t)
    scores = apply_strict_recent_exclusions(scores, hist, t)

    all7 = list(itertools.combinations(s17, 7))
    filtered = [c for c in all7 if combo_ok(c)]

    def score7(c7):
        base = sum(scores.get(n,0.0) for n in c7)
        zc = [0]*7
        for n in c7: zc[zone_id(n)] += 1
        pen = sum((zc[i] - 1.0)**2 for i in range(7)) * 0.03
        return base - pen

    ranked = sorted(filtered, key=lambda c: score7(c), reverse=True)
    scores7 = [score7(c) for c in ranked]
    return ranked, scores7

def diversify(ranked, raw_scores, want=10, max_overlap=3):
    chosen, scs = [], []
    for c,sc in zip(ranked, raw_scores):
        if all(len(set(c)&set(x)) <= max_overlap for x in chosen):
            chosen.append(c); scs.append(sc)
        if len(chosen)>=want: break
    if scs:
        mn, mx = min(scs), max(scs)
        sc_norm = [50.0]*len(scs) if mx-mn < 1e-9 else [(s-mn)*100.0/(mx-mn) for s in scs]
    else:
        sc_norm = []
    return chosen, sc_norm

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    f = request.files.get('file')
    if not f: return jsonify({'error':'No file uploaded'}), 400
    content = f.read()
    hist_rows = parse_history_from_bytes(content)
    chron = chronological(True, hist_rows)

    # Read UI params (exactly as your original HTML expects)
    want = int(request.form.get('want','10'))
    overlap = int(request.form.get('overlap','3'))
    seed_base = int(request.form.get('seed','42'))
    n_sets = int(request.form.get('n_sets','5'))
    K = int(request.form.get('K','60'))
    lam = float(request.form.get('lam','0.05'))
    M = int(request.form.get('M','50'))

    t = len(chron)
    if t<6: return jsonify({'error':'Not enough history'}), 400

    sets17 = []
    excl_debug = None
    prime_warnings = []
    combos_all = []

    for i in range(n_sets):
        s17, excl_last2 = sample17_excluding_rules(chron, t, M=M, seed=seed_base + i)
        sets17.append(s17)
        if excl_debug is None:
            excl_debug = excl_last2

        # Prime count check (warn-only)
        prime_count = sum(1 for n in s17 if n in PRIMES)
        if prime_count < 3 or prime_count > 6:
            prime_warnings.append(f"⚠️ בסט #{i+1} יש {prime_count} ראשוניים — נדרש בין 3 ל־6.")

        ranked7, raw7 = combo_filter_and_score(s17, chron, t, K=K, lam=lam)
        chosen7, norm7 = diversify(ranked7, raw7, want=want, max_overlap=overlap)
        combos_all.append([{'numbers': list(c), 'score': round(sc,2)} for c,sc in zip(chosen7, norm7)])

    # Files
    ts = datetime.datetime.now() + datetime.timedelta(hours=3)  # keep as before
    ts_str = ts.strftime('%Y%m%d_%H%M')
    out_dir = 'outputs'; os.makedirs(out_dir, exist_ok=True)

    # Save sets17 CSV
    sets_rows = []
    for i, s17 in enumerate(sets17, start=1):
        row = {'set_id': i}
        for j, n in enumerate(s17, start=1):
            row[f'K{j}'] = n
        sets_rows.append(row)
    sets_path = os.path.join(out_dir, f'sets17_{ts_str}.csv')
    pd.DataFrame(sets_rows).to_csv(sets_path, index=False)

    # Save combos CSV (flattened)
    combo_rows = []
    for i, lst in enumerate(combos_all, start=1):
        for rank, item in enumerate(lst, start=1):
            row = {'set_id': i, 'rank': rank}
            for k,n in enumerate(item['numbers'], start=1):
                row[f'N{k}'] = n
            row['score_0to100'] = item['score']
            combo_rows.append(row)
    combos_path = os.path.join(out_dir, f'combos7_{ts_str}.csv')
    if combo_rows:
        pd.DataFrame(combo_rows).to_csv(combos_path, index=False)
    else:
        combos_path = None

    return jsonify({
        'strategy': 'auto_blend + strict_recent_exclusion(2-in-3, 3-in-5)',
        'excluded_repeats_last2': excl_debug,
        'sets17': sets17,
        'prime_warnings': prime_warnings,
        'combos_by_set': combos_all,
        'files': {
            'sets17_csv': '/download/' + os.path.basename(sets_path),
            'combos_csv': '/download/' + os.path.basename(combos_path) if combos_path else None
        }
    })

@app.route('/download/<path:fname>')
def download_file(fname):
    return send_from_directory('outputs', fname, as_attachment=True)

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
# Render-compatible entry
#
    threading.Timer(1.0, open_browser).start()
    app.run(host='127.0.0.1', port=5000, debug=False)
