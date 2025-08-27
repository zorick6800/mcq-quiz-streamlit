import streamlit as st, pandas as pd, numpy as np, re, json

# Audio + ASR
try:
    import sounddevice as sd
    from vosk import Model, KaldiRecognizer
except Exception as e:
    st.error("Missing packages. Run:  py -m pip install streamlit vosk sounddevice numpy pandas")
    st.stop()

st.set_page_config(page_title="MCQ — Tap to Talk (v3.1 fixed + persistent explanation)", page_icon="🎤", layout="centered")
st.title("MCQ Checker — Tap to Talk")

# ---------- Grammar & helpers ----------
GRAMMAR = ["select","choose","pick","option","check","submit","verify",
           "next","previous","prev","back","clear","reset","none","repeat","question","quit","exit",
           "a","b","c","d",
           "alpha","bravo","charlie","delta",
           "see","sea","bee","be","dee","de","ay",
           "[unk]"]

CANON = {
    "alpha":"A","bravo":"B","charlie":"C","delta":"D",
    "see":"C","sea":"C","bee":"B","be":"B","dee":"D","de":"D","ay":"A",
    "a":"A","b":"B","c":"C","d":"D"
}

def tokenize(text:str):
    return re.findall(r"[a-z]+", (text or "").lower())

def letters_from_tokens(tokens):
    letters = [CANON[t] for t in tokens if t in CANON]
    return sorted(set([L for L in letters if L in ["A","B","C","D"]]))

def parse_command(text: str):
    toks = tokenize(text)
    if not toks: return None, {"tokens":[], "letters":[], "cmd":None}
    letters = letters_from_tokens(toks)
    cmd = None
    if any(w in toks for w in ["quit","exit"]): cmd = ("quit", None)
    elif any(w in toks for w in ["repeat","question"]): cmd = ("repeat", None)
    elif any(w in toks for w in ["clear","reset","none"]): cmd = ("clear", None)
    elif any(w in toks for w in ["next","forward"]): cmd = ("next", None)
    elif any(w in toks for w in ["previous","prev","back"]): cmd = ("prev", None)
    elif any(w in toks for w in ["check","submit","verify"]): cmd = ("check", None)
    elif letters and (any(w in toks for w in ["select","choose","pick","option"]) or True):
        cmd = ("select", letters)
    return cmd, {"tokens":toks, "letters":letters, "cmd":(cmd[0] if cmd else None)}

def resample_to_16k(x: np.ndarray, sr_in: int) -> np.ndarray:
    if sr_in == 16000: return x.astype(np.float32)
    n_in = x.shape[0]
    n_out = int(n_in * 16000 / sr_in)
    if n_out <= 0: return np.zeros(0, dtype=np.float32)
    xp = np.linspace(0, n_in-1, n_in, dtype=np.float32)
    xq = np.linspace(0, n_in-1, n_out, dtype=np.float32)
    return np.interp(xq, xp, x).astype(np.float32)

# ---------- State ----------
def init_state():
    d = {
        "df": None, "idx": 0,
        "selection": set(),          # source of truth
        "selection_dirty": False,    # True when changed programmatically
        "last_text": "", "last_action": "",
        "last_explanation": "",      # <— persist explanation here
        "last_cmd_debug": {"tokens":[],"letters":[],"cmd":None},
        "model_dir": "models/vosk-model-small-en-us-0.15",
        "widget_rev": 0,
        "csv_loaded": False,
        "show_loaded_banner": False,
    }
    for k,v in d.items():
        if k not in st.session_state: st.session_state[k] = v
init_state()

# Show the green banner only once after load
if st.session_state.show_loaded_banner and st.session_state.df is not None:
    st.success(f"Loaded {len(st.session_state.df)} questions.")
    st.session_state.show_loaded_banner = False

def rerun():
    try: st.rerun()
    except Exception: st.experimental_rerun()

# ---------- Sidebar ----------
with st.sidebar:
    f = st.file_uploader("Upload CSV (Question,A,B,C,D,Answer,Explanation)", type=["csv"])
    st.session_state.model_dir = st.text_input("Vosk model folder", value=st.session_state.model_dir)
    sr = st.number_input("Input sample rate", value=44100, step=1000)
    seconds = st.slider("Listen seconds", 1.0, 4.0, 2.0, 0.5)
    energy_gate = st.slider("Energy gate (RMS)", 0.0001, 0.50, 0.03, 0.0005)

    device_idx = None
    try:
        devs = sd.query_devices()
        ins = [(i,d) for i,d in enumerate(devs) if int(d.get("max_input_channels",0))>0]
        if ins:
            labels = [f"{i}: {d['name']} — in={int(d.get('max_input_channels',0))} — defaultSR={int(d.get('default_samplerate',16000))}" for i,d in ins]
            sel = st.selectbox("Input device", options=list(range(len(ins))), format_func=lambda k: labels[k])
            device_idx = ins[sel][0]
        else:
            st.warning("No input (microphone) devices found.")
    except Exception as e:
        st.warning(f"Could not list devices: {e}")

# ---------- CSV (load once) ----------
def load_csv(file):
    df = pd.read_csv(file)
    need = ["Question","A","B","C","D","Answer"]
    for c in need:
        if c not in df.columns: raise ValueError(f"Missing column: {c}")
    if "Explanation" not in df.columns: df["Explanation"] = ""
    for c in ["Question","A","B","C","D","Answer","Explanation"]:
        df[c] = df[c].astype(str).fillna("")
    return df

if not st.session_state.csv_loaded:
    if f is not None:
        try:
            st.session_state.df = load_csv(f)
            st.session_state.idx = 0
            st.session_state.selection = set()
            st.session_state.selection_dirty = True
            st.session_state.last_action = ""
            st.session_state.last_explanation = ""
            st.session_state.widget_rev += 1
            st.session_state.csv_loaded = True
            st.session_state.show_loaded_banner = True
        except Exception as e:
            st.error(f"Failed to load CSV: {e}")
else:
    if st.sidebar.button("🔁 Reload CSV"):
        st.session_state.csv_loaded = False
        st.session_state.df = None
        st.session_state.widget_rev += 1
        rerun()

# ---------- Model cache ----------
def _get_model(path):
    return Model(path)
try:
    get_model = st.cache_resource(show_spinner=False)(_get_model)  # type: ignore
except Exception:
    get_model = _get_model

# ---------- Listen & transcribe ----------
def listen_once(seconds: float, sr: int, device):
    if device is None:
        raise RuntimeError("No microphone selected. Choose an input device in the sidebar.")
    stream = sd.InputStream(samplerate=int(sr), channels=1, dtype="float32", device=device)
    stream.start()
    frames = int(sr * seconds)
    chunks, remaining = [], frames
    while remaining > 0:
        take = min(2048, remaining)
        x,_ = stream.read(take)
        if x is not None and x.size:
            chunks.append(x.flatten().astype(np.float32))
            remaining -= x.shape[0]
    stream.stop(); stream.close()
    return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)

def transcribe_once(audio_f32: np.ndarray, sr: int, model_dir: str):
    model = get_model(model_dir)
    rec = KaldiRecognizer(model, 16000, json.dumps(GRAMMAR))
    y = resample_to_16k(audio_f32, sr)
    rms = float(np.sqrt(np.mean(np.square(y))) if y.size else 0.0)
    if rms < 1e-6: return "", 0.0
    pcm16 = (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    step = 3200  # ~0.2s at 16k
    for i in range(0, len(pcm16), step*2):  # 2 bytes/sample
        rec.AcceptWaveform(pcm16[i:i+step*2])
    res = json.loads(rec.FinalResult())
    return (res.get("text") or "").strip(), rms

# ---------- UI ----------
df = st.session_state.df
if df is not None:
    i = st.session_state.idx
    row = df.iloc[i]

    st.subheader(f"Question {i+1}/{len(df)}")
    st.write(row["Question"])

    # Render checkboxes with revisioned keys; never mutate widget keys after creation
    rev = st.session_state.widget_rev
    voice_sel = set(st.session_state.selection)

    for k in ["A","B","C","D"]:
        val = row[k].strip()
        if val:
            st.checkbox(
                f"{k}) {val}",
                key=f"chk_{i}_{k}_rev{rev}",
                value=(k in voice_sel)
            )

    # Only adopt manual changes if we did NOT just programmatically update selection
    if not st.session_state.selection_dirty:
        manual_sel = {k for k in ["A","B","C","D"]
                      if st.session_state.get(f"chk_{i}_{k}_rev{rev}", False)}
        if manual_sel != st.session_state.selection:
            st.session_state.selection = manual_sel
    st.session_state.selection_dirty = False  # clear flag

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        do_listen = st.button("🎤 Listen")
    with col2:
        if st.button("⏭️ Next"):
            st.session_state.idx = min(len(df)-1, i+1)
            st.session_state.selection = set()
            st.session_state.selection_dirty = True
            st.session_state.last_action = ""
            st.session_state.last_explanation = ""
            st.session_state.widget_rev += 1
            rerun()
    with col3:
        if st.button("⏮️ Previous"):
            st.session_state.idx = max(0, i-1)
            st.session_state.selection = set()
            st.session_state.selection_dirty = True
            st.session_state.last_action = ""
            st.session_state.last_explanation = ""
            st.session_state.widget_rev += 1
            rerun()
    with col4:
        if st.button("🧹 Clear"):
            st.session_state.selection = set()
            st.session_state.selection_dirty = True
            st.session_state.last_action = "Cleared."
            st.session_state.last_explanation = ""
            st.session_state.widget_rev += 1
            rerun()
    with col5:
        if st.button("✅ Check"):
            ans = set(list(str(row["Answer"]).upper().replace(" ", "")))
            st.session_state.last_explanation = row["Explanation"].strip() or ""
            if st.session_state.selection == ans:
                st.session_state.last_action = "✅ Correct!"
            else:
                st.session_state.last_action = f"❌ Wrong. Yours: {''.join(sorted(st.session_state.selection)) or '—'} | Answer: {''.join(sorted(ans))}"
            rerun()

    if do_listen:
        try:
            audio = listen_once(seconds, int(sr), device_idx)
            rms = float(np.sqrt(np.mean(np.square(audio))) if audio.size else 0.0)
            if rms < float(energy_gate):
                st.warning(f"Too quiet (RMS {rms:.6f} < gate {energy_gate:.6f}). Try louder or lower the gate.")
            else:
                text, _ = transcribe_once(audio, int(sr), st.session_state.model_dir)
                st.session_state.last_text = text
                cmd, dbg = parse_command(text)
                st.session_state.last_cmd_debug = dbg
                if cmd:
                    kind, data = cmd
                    if kind == "select":
                        st.session_state.selection = set(data or [])
                        st.session_state.selection_dirty = True
                        st.session_state.last_action = f"Selected: {' '.join(sorted(st.session_state.selection)) or '—'}"
                        st.session_state.last_explanation = ""
                        st.session_state.widget_rev += 1
                        rerun()
                    elif kind == "check":
                        ans = set(list(str(row["Answer"]).upper().replace(" ", "")))
                        st.session_state.last_explanation = row["Explanation"].strip() or ""
                        if st.session_state.selection == ans:
                            st.session_state.last_action = "✅ Correct!"
                        else:
                            st.session_state.last_action = f"❌ Wrong. Yours: {''.join(sorted(st.session_state.selection)) or '—'} | Answer: {''.join(sorted(ans))}"
                        rerun()
                    elif kind == "next":
                        st.session_state.idx = min(len(df)-1, i+1)
                        st.session_state.selection = set()
                        st.session_state.selection_dirty = True
                        st.session_state.last_action = ""
                        st.session_state.last_explanation = ""
                        st.session_state.widget_rev += 1
                        rerun()
                    elif kind == "prev":
                        st.session_state.idx = max(0, i-1)
                        st.session_state.selection = set()
                        st.session_state.selection_dirty = True
                        st.session_state.last_action = ""
                        st.session_state.last_explanation = ""
                        st.session_state.widget_rev += 1
                        rerun()
                    elif kind == "clear":
                        st.session_state.selection = set()
                        st.session_state.selection_dirty = True
                        st.session_state.last_action = "Cleared."
                        st.session_state.last_explanation = ""
                        st.session_state.widget_rev += 1
                        rerun()
                    elif kind == "repeat":
                        pass
                    elif kind == "quit":
                        st.stop()
                else:
                    st.session_state.last_action = "(no actionable command)"
                    st.session_state.last_explanation = ""
                    rerun()
        except Exception as e:
            st.error(f"Listening failed: {e}")

    st.markdown("---")
    st.subheader("Transcript")
    st.info(st.session_state.last_text or "(press Listen and speak)")

    st.subheader("Result")
    sel = "".join(sorted(st.session_state.selection)) or "—"
    st.write(f"Selected: {sel}")
    msg = st.session_state.last_action
    if msg:
        if msg.startswith("✅"): st.success(msg)
        elif msg.startswith("❌"): st.error(msg)
        else: st.info(msg)
    # Persistent explanation display
    if st.session_state.last_explanation:
        st.info("Explanation: " + st.session_state.last_explanation)

    st.subheader("Debug")
    dbg = st.session_state.last_cmd_debug or {}
    st.json({"tokens": dbg.get("tokens", []),
             "letters": dbg.get("letters", []),
             "cmd": dbg.get("cmd", None)})

else:
    st.info("Upload your CSV to begin. Required columns: Question, A, B, C, D, Answer, (optional) Explanation.")
