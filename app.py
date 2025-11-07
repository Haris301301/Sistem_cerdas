# ================================================================
# Chatbot Harga Mobil — Bebas Ngetik, tetapi 100% patuh model.pkl
# ================================================================
# Karakteristik:
# - Chat natural & bebas (multi-turn), streaming seperti Gemini
# - Semua angka HANYA dari model.pkl (bukan dari Gemini)
# - Gemini dipakai untuk NLU (intent + slot/fitur) & merangkai kalimat saja
# - Jika info kurang: tanya balik (clarification) tanpa mengulang-ulang
# - Small talk/pertanyaan umum: dijawab ramah tapi tidak keluar batas model
# - Tidak menampilkan daftar inventori/tipe; fokus prediksi per deskripsi user
# ================================================================

import os, json, pickle, re, textwrap, time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import streamlit as st

# --------------------- Konfigurasi UI ---------------------
st.set_page_config(page_title="Chatbot Harga Mobil (Model Lokal)", page_icon="🚗", layout="wide")
st.title("🤖 Chatbot Harga Mobil — Bebas Ngetik, Hasil Patuh Model Lokal")

WORKDIR = Path.cwd()

# --------------------- Util umum --------------------------
def rupiah(x: float) -> str:
    try:
        return "Rp {:,.0f}".format(float(x)).replace(",", ".")
    except:
        return f"Rp {x}"

def is_null(v) -> bool:
    return v in (None, "", "unknown", "null") or (isinstance(v, float) and pd.isna(v))

def coerce_number(x, default=0):
    try:
        return float(str(x).replace(".", "").replace(",", "."))
    except:
        return default

def find_year_key(cols: List[str]) -> str:
    for c in cols:
        if c.lower() == "year": return c
    for c in cols:
        if "year" in c.lower(): return c
    return "year"

def normalize_trans(v: Any):
    t = str(v).lower()
    if "man" in t: return "Manual"
    if "auto" in t or "matic" in t or "otomatis" in t: return "Automatic"
    return v

def normalize_fuel(v: Any):
    t = str(v).lower()
    if "diesel" in t: return "Diesel"
    if "bensin" in t or "petrol" in t or "gasoline" in t: return "Petrol"
    return v

# ---------------- Gemini (untuk NLU & phrasing, bukan angka) -------------
try:
    import google.generativeai as genai
except Exception:
    st.error("❌ Library `google-generativeai` belum terpasang. Jalankan: `pip install -U google-generativeai`")
    st.stop()

API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if not API_KEY:
    st.error("❌ GEMINI_API_KEY belum diset.\nPowerShell →  $env:GEMINI_API_KEY=\"PASTE_KEY\"")
    st.stop()

genai.configure(api_key=API_KEY)
GMODEL = genai.GenerativeModel("models/gemini-2.0-flash")

def stream_markdown(text_generator):
    """Tampilkan hasil generate_content(stream=True) secara bertahap."""
    ph = st.empty()
    acc = ""
    for ev in text_generator:
        chunk = getattr(ev, "text", None)
        if not chunk: continue
        acc += chunk
        ph.markdown(acc)
    return acc.strip() if acc.strip() else ""

# ---------------- Model Lokal -----------------------------
st.sidebar.header("⚙️ Model Lokal (Wajib)")
model_path = st.sidebar.text_input("Path model.pkl", str(WORKDIR / "model.pkl"))
cols_path  = st.sidebar.text_input("Path columns.json", str(WORKDIR / "columns.json"))

if not (os.path.exists(model_path) and os.path.exists(cols_path)):
    st.warning("Pastikan file `model.pkl` dan `columns.json` tersedia.")
    st.stop()

@st.cache_resource(show_spinner=False)
def load_model_and_schema(mp, cp):
    with open(mp, "rb") as f:
        mdl = pickle.load(f)
    with open(cp, "r", encoding="utf-8") as f:
        cols = json.load(f)
        # terima bentuk list atau dict {"numerical":[...], "categorical":[...]}
        if isinstance(cols, dict):
            features = list(cols.get("numerical", [])) + list(cols.get("categorical", []))
        else:
            features = list(cols)
    return mdl, features

model, feature_cols = load_model_and_schema(model_path, cols_path)
YEAR_KEY = find_year_key(feature_cols)

st.sidebar.success("✅ Model & schema termuat")

# ---------------- State Chat ------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": (
            "Halo! Aku bisa memperkirakan **harga mobil bekas** berdasarkan model lokal milikmu.\n\n"
            "Ketik bebas, misal: `harga yaris 2017 matic 15rb km bensin`, `cek civic 2019 manual`, atau `saya ada budget 150 juta untuk avanza 2018`.\n"
            "_Catatan: semua angka murni dari model.pkl. Aku tidak akan membuat angka di luar hasil model._"
        )
    }]

def history_summary(n_last: int = 6) -> str:
    msgs = st.session_state.messages[-n_last:]
    text = ""
    for m in msgs:
        who = "User" if m["role"] == "user" else "Asisten"
        text += f"{who}: {m['content']}\n"
    return textwrap.shorten(text, width=700, placeholder="…")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])

# ---------------- Intent & Slot Extraction ----------------
INTENT_PROMPT_TMPL = """
Anda adalah NLU untuk showroom mobil bekas. Klasifikasikan intent dan ekstrak fitur ke JSON satu baris.

Intent yang mungkin:
- "predict": user meminta estimasi harga berdasarkan deskripsi mobil
- "clarify": user memberikan lanjutan/penyempurnaan (mis. menyebut bahan bakar saja, atau transmisi saja)
- "chitchat": sapaan atau pertanyaan di luar prediksi harga
- "inventory": meminta daftar tipe/daftar semua model (jawab: tidak tersedia)

Keluaran HARUS JSON satu baris dengan kunci:
{{"intent": "...", "features": {{}}, "missing": []}}

`features` hanya boleh menggunakan kolom berikut: {cols}
Isi `missing` dengan kolom penting yang belum ada (mis. ["model","{year_key}","mileage","transmission","fuelType"]).
Teks: {text}
JSON:
"""

def nlu_intent_and_features(user_text: str) -> Dict[str, Any]:
    prompt = INTENT_PROMPT_TMPL.format(cols=feature_cols, year_key=YEAR_KEY, text=user_text)
    try:
        out = GMODEL.generate_content(prompt).text.strip()
        s, e = out.find("{"), out.rfind("}")
        js = json.loads(out[s:e+1]) if s != -1 and e != -1 else {}
        if not isinstance(js, dict): js = {}
    except Exception:
        js = {}
    # fallback kalau kosong
    js.setdefault("intent", "predict")
    js.setdefault("features", {})
    js.setdefault("missing", [])
    return js

def coerce_features_to_model_space(feat: Dict[str, Any]) -> Dict[str, Any]:
    """Map & normalisasi nilai agar cocok dengan kolom model."""
    cleaned = {}
    for c in feature_cols:
        if c in feat:
            v = feat[c]
            if c.lower() == "transmission": v = normalize_trans(v)
            if c.lower() == "fueltype":     v = normalize_fuel(v)
            cleaned[c] = v
    # Normalisasi khusus angka umum
    for k in list(cleaned.keys()):
        if k.lower() in {YEAR_KEY.lower(), "mileage", "tax", "mpg", "engine_size", "engineSize".lower()}:
            cleaned[k] = coerce_number(cleaned[k], default=0)
    return cleaned

def build_model_input_row(cleaned: Dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame([{}])
    # kolom urut mengikuti feature_names_in_ bila ada
    if hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)
    else:
        cols = feature_cols
    for c in cols:
        if c in cleaned:
            df[c] = cleaned[c]
        else:
            # kategori → "unknown", numerik → 0 (heuristik ringan)
            df[c] = np.nan
    # tipe data
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype("string").fillna("unknown")
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df[cols]

# ---------------- Respon Natural (tanpa bikin angka) ------
def natural_reply_from_result(user_text: str, pred_value: float, used_feat: Dict[str, Any], missing: List[str]) -> str:
    """Susun kalimat sendiri agar aman dari halusinasi angka."""
    parts = []
    # 1) Ringkasan fitur yang terpakai
    cue = []
    for k in ["model", YEAR_KEY, "transmission", "fuelType", "mileage", "engineSize"]:
        if k in used_feat and not is_null(used_feat[k]):
            val = used_feat[k]
            if k == "mileage":
                try: val = f"{int(val):,}".replace(",", ".") + " km"
                except: val = f"{val} km"
            if k == "engineSize": val = f"{val} L"
            cue.append(f"{k} **{val}**")
    if cue:
        parts.append("Detail yang saya gunakan: " + ", ".join(cue) + ".")
    # 2) Angka dari model
    parts.append(f"Perkiraan harga menurut model: **{rupiah(pred_value)}**.")
    # 3) Ajukan tanya singkat jika masih kurang
    if missing:
        yang_kurang = ", ".join(missing[:3])
        parts.append(f"Untuk hasil yang lebih presisi, mohon lengkapi: **{yang_kurang}**.")
    return "\n\n".join(parts)

# ----------------- Tampilkan chat input -------------------
user_text = st.chat_input("Tulis apa saja tentang mobilmu…")
if not user_text:
    st.stop()

if user_text.strip().lower() == "/reset":
    st.session_state.clear(); st.rerun()

with st.chat_message("user"):
    st.write(user_text)
st.session_state.messages.append({"role":"user","content":user_text})

# ----------------- NLU: intent + slot ---------------------
nlu = nlu_intent_and_features(user_text)
intent = str(nlu.get("intent","predict")).lower()
raw_feat = dict(nlu.get("features", {}))
missing = list(nlu.get("missing", []))

# beberapa intent non-prediksi tapi tetap "chatty"
if intent == "inventory":
    with st.chat_message("assistant"):
        text = (
            "Aku tidak menyimpan daftar inventori/tipe mobil. "
            "Tapi aku bisa **memperkirakan harga** untuk mobil apa pun kalau kamu sebutkan detailnya.\n"
            "Contoh: `avanza 2018 manual bensin 70rb km`."
        )
        st.write(text)
    st.session_state.messages.append({"role":"assistant","content":text})
    st.stop()

if intent == "chitchat":
    # Balas ramah; tidak mengeluarkan angka atau fakta di luar model
    with st.chat_message("assistant"):
        hist = history_summary()
        prompt = textwrap.dedent(f"""
            Kamu adalah asisten ramah. Balas singkat & natural untuk small talk.
            Jangan menyebut angka/daftar inventori. Ingatkan bahwa kamu fokus pada prediksi harga berbasis model pengguna.
            Ringkasan obrolan:
            {hist}

            Pesan: {user_text}
        """).strip()
        ans = stream_markdown(GMODEL.generate_content(prompt, stream=True))
    st.session_state.messages.append({"role":"assistant","content":ans})
    st.stop()

# ----------------- Normalisasi fitur ----------------------
cleaned = coerce_features_to_model_space(raw_feat)

# kalau user hanya menyebut "bensin"/"matic"/"2018" → intent akan 'clarify'
# kita gabungkan dengan konteks sebelumnya (ambil fitur dari msg assistant terakhir jika ada)
def last_used_features_from_history() -> Dict[str, Any]:
    # cari pola "Detail yang saya gunakan:" dari balasan sebelumnya (jika ada)
    used = {}
    for m in reversed(st.session_state.messages):
        if m["role"] != "assistant": continue
        txt = m["content"]
        if "Detail yang saya gunakan:" in txt:
            seg = txt.split("Detail yang saya gunakan:")[-1].strip()
            # contoh parsing sederhana "model **Yaris**, year **2018**"
            for token in seg.split(","):
                token = token.strip().strip(".")
                if "**" in token:
                    try:
                        k = token.split("**")[0].strip().split()[-1]
                        v = token.split("**")[1]
                        used[k] = v.replace(" km","").replace(" L","")
                    except:
                        pass
            break
    return used

if intent == "clarify":
    prev = last_used_features_from_history()
    # merge: yang baru override yang lama
    prev.update(cleaned)
    cleaned = prev

# ----------------- Siapkan input ke model -----------------
df = build_model_input_row(cleaned)

# Cek minimal fitur yang wajar untuk prediksi
minimal_keys = set([YEAR_KEY, "model"]) & set(df.columns)
if any(col not in df.columns for col in minimal_keys) or (is_null(cleaned.get("model")) and is_null(cleaned.get(YEAR_KEY))):
    # minta klarifikasi tanpa membuat angka
    need = [k for k in [ "model", YEAR_KEY, "mileage", "transmission", "fuelType" ] if k in feature_cols and is_null(cleaned.get(k))]
    need_text = ", ".join(need[:3]) if need else "model dan tahun"
    with st.chat_message("assistant"):
        st.write(f"Aku bisa bantu, tapi butuh detail minimal **model** dan **{YEAR_KEY}**. Boleh sebutkan {need_text}?")
    st.session_state.messages.append({"role":"assistant","content":f"Butuh detail minimal model & {YEAR_KEY}. Boleh sebutkan {need_text}?"})
    st.stop()

# ----------------- Prediksi dari model (ANGKA RESMI) ------
try:
    raw_pred = float(model.predict(df)[0])
except Exception as e:
    with st.chat_message("assistant"):
        st.error(f"Gagal memprediksi dari model: {e}")
    st.stop()

# Skala otomatis (jika model keluarkan 'juta' → kali 1e6)
pred_value = raw_pred * 1_000_000 if 1 <= raw_pred < 500_000 else raw_pred

# ----------------- Jawaban Natural TANPA bikin angka -------
reply_text = natural_reply_from_result(user_text, pred_value, cleaned, missing)

with st.chat_message("assistant"):
    # Untuk gaya lebih natural, minta Gemini memparafrase "reply_text" TANPA mengubah angka
    safety_prompt = textwrap.dedent(f"""
        Parafrase kalimat berikut agar natural & sopan. 
        PENTING: **Jangan ubah angka, satuan, atau fakta** di dalamnya; 
        fokus pada perapihan bahasa saja.
        Teks:
        {reply_text}
    """).strip()
    final_ans = stream_markdown(GMODEL.generate_content(safety_prompt, stream=True))

st.session_state.messages.append({"role":"assistant","content":final_ans})
