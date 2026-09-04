import os
import re
import json
import streamlit as st
import streamlit.components.v1
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine, text

# PyVis untuk network graph korelasi tema
try:
    from pyvis import network as pvnet
    import tempfile, os as _os
    def pv_static(g):
        """Render PyVis graph sebagai HTML di Streamlit tanpa stvis."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as _f:
            _fname = _f.name
        g.save_graph(_fname)
        with open(_fname, "r", encoding="utf-8") as _f:
            _html = _f.read()
        _os.unlink(_fname)
        # Inject CSS agar background transparan
        _html = _html.replace(
            "body {",
            "body { background: transparent !important; margin:0; padding:0; "
        )
        st.components.v1.html(_html, height=520, scrolling=False)
    PYVIS_OK = True
except ImportError:
    PYVIS_OK = False

# =====================================================
# =====================================================
# KONFIGURASI — Supabase REST API
# Tidak perlu port TCP — pakai HTTPS port 443
# =====================================================

SUPABASE_URL = st.secrets.get("SUPABASE_URL", os.environ.get("SUPABASE_URL", ""))
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", os.environ.get("SUPABASE_KEY", ""))
NAMA_TABEL   = "HI"

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error(
        "SUPABASE_URL / SUPABASE_KEY belum diisi. "
        "Isi di .streamlit/secrets.toml (lokal) atau menu Secrets di Streamlit Cloud."
    )
    st.stop()

import requests as _req

# =====================================================
# KONEKSI — Supabase REST API → SQLite in-memory
# =====================================================

def _sb_headers():
    return {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type":  "application/json",
    }

def _sb_fetch(tabel, limit=5000):
    """
    Fetch semua baris dari Supabase dengan pagination otomatis.
    Tidak ada batas jumlah baris — aman untuk data > 5.000 atau > 10.000.
    """
    PAGE   = 1000   # Supabase max per request
    semua  = []
    offset = 0

    while True:
        headers = {
            **_sb_headers(),
            "Range-Unit": "items",
            "Range":      f"{offset}-{offset + PAGE - 1}",
        }
        url  = f"{SUPABASE_URL}/rest/v1/{tabel}?select=*"
        resp = _req.get(url, headers=headers, timeout=20)

        # 206 Partial Content atau 200 OK — keduanya valid
        if resp.status_code not in (200, 206):
            resp.raise_for_status()

        data = resp.json()
        if isinstance(data, dict) and "code" in data:
            raise Exception(f"Supabase error: {data.get('message', data)}")
        if not isinstance(data, list) or len(data) == 0:
            break

        semua.extend(data)
        print(f"  Fetch offset {offset}: {len(data)} baris (total: {len(semua)})")

        # Jika data yang dikembalikan kurang dari PAGE, berarti sudah halaman terakhir
        if len(data) < PAGE:
            break
        offset += PAGE

    return pd.DataFrame(semua)

def _sb_count(tabel):
    h    = {**_sb_headers(), "Prefer": "count=exact", "Range-Unit": "items", "Range": "0-0"}
    resp = _req.head(f"{SUPABASE_URL}/rest/v1/{tabel}?select=*", headers=h, timeout=10)
    cr   = resp.headers.get("Content-Range", "0/0")
    return int(cr.split("/")[-1]) if "/" in cr else 0

try:
    _n      = _sb_count(NAMA_TABEL)
    _df_all = _sb_fetch(NAMA_TABEL)
    if _df_all.empty:
        raise Exception("Data kosong dari Supabase")

    # Normalisasi kolom: "Tema LKS Bipartit" → "tema", "Kode Area" → "kode_area" dll
    _df_all.columns = (
        _df_all.columns
               .str.strip()
               .str.lower()
               .str.replace(" ", "_", regex=False)
               .str.replace("(", "", regex=False)
               .str.replace(")", "", regex=False)
    )
    # Rename kolom — mendukung nama lama & nama baru di Supabase
    _rename_map = {
        # Nama kolom tema
        "tema_lks_bipartit":      "tema",
        # Nama kolom area — lama & baru
        "kode_area":              "kode_area",
        "personal_area_nama":     "kode_area",   # nama baru Supabase
        # Sub area — lama & baru
        "kode_sub_area":          "kode_sub_area",
        "personal_sub_area_nama": "kode_sub_area",  # nama baru Supabase
        # Kolom lain
        "tanggal_pelaksanaan":    "tanggal_pelaksanaan",
        "latar_belakang":         "latar_belakang",
        "tanggal_deadline":       "tanggal_deadline",
        "tindak_lanjut":          "tindak_lanjut",
        "tanggal_tindak_lanjut":  "tanggal_tindak_lanjut",
    }
    _df_all = _df_all.rename(columns=_rename_map)
    # Cetak kolom aktual untuk debug di terminal
    print(f"  Kolom setelah normalisasi: {list(_df_all.columns)}")

    # Tambah kolom status_tindak_lanjut jika belum ada
    if "status_tindak_lanjut" not in _df_all.columns and "tindak_lanjut" in _df_all.columns:
        _df_all["status_tindak_lanjut"] = _df_all["tindak_lanjut"].apply(
            lambda x: "Sudah" if pd.notna(x) and str(x).strip() != "" else "Belum"
        )

    engine    = create_engine("sqlite://")
    _df_all.to_sql("lks_bipartit", engine, if_exists="replace", index=False)
    print(f"✅ Kolom SQLite: {list(_df_all.columns)}")
    SUMBER    = f"Supabase REST API — {_n:,} baris"
    DATA_SIAP = True
    print(f"✅ {SUMBER}")
    # RAG index dibangun setelah semua fungsi didefinisikan (di bawah)

except Exception as _e_sb:
    DATA_SIAP  = False
    DATA_ERROR = str(_e_sb)
    print(f"❌ Gagal koneksi Supabase: {_e_sb}")



# =====================================================
# KONFIGURASI LLM
# =====================================================

GEMINI_KEY   = os.environ.get("GEMINI_API_KEY", "")
GROQ_KEY     = os.environ.get("GROQ_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"
GROQ_MODEL   = "llama-3.3-70b-versatile"

if GEMINI_KEY:
    PROVIDER = "gemini"
elif GROQ_KEY:
    PROVIDER = "groq"
else:
    PROVIDER = "mock"

USE_MOCK = (PROVIDER == "mock")

# =====================================================
# FUNGSI PANGGIL LLM
# =====================================================

def panggil_gemini(prompt: str) -> str:
    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_KEY)
        resp   = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return resp.text
    except Exception as e:
        return f"ERROR: {e}"


def panggil_groq(prompt: str) -> str:
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_KEY)
        resp   = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"ERROR: {e}"

# =====================================================
# HELPER: deteksi periode dari pertanyaan
# Mengembalikan string filter SQL atau "" jika tidak ada
# Contoh: "AND tanggal_pelaksanaan LIKE '2026-02%'"
# =====================================================

BULAN_MAP = {
    "januari": "01", "february": "02", "februari": "02",
    "maret": "03",   "april": "04",    "mei": "05",
    "juni": "06",    "juli": "07",     "agustus": "08",
    "september": "09","oktober": "10", "november": "11",
    "desember": "12",
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "jun": "06", "jul": "07", "agt": "08", "sep": "09",
    "okt": "10", "nov": "11", "des": "12",
}

def deteksi_periode(q: str) -> str:
    """
    Kembalikan klausa WHERE (tanpa WHERE) untuk filter periode,
    atau string kosong jika tidak ada periode disebut.
    """
    q_low = q.lower()

    # Cari tahun
    tahun_m = re.search(r"\b(202\d)\b", q_low)
    tahun   = tahun_m.group(1) if tahun_m else "2026"

    # Cari nama bulan
    bulan_no = None
    for nama, no in BULAN_MAP.items():
        if nama in q_low:
            bulan_no = no
            break

    # Cari bulan dalam format "bulan N" atau "bulan ke-N"
    if not bulan_no:
        m = re.search(r"bulan\s+ke[-\s]?(\d{1,2})", q_low)
        if m:
            bulan_no = f"{int(m.group(1)):02d}"

    if bulan_no:
        return f"tanggal_pelaksanaan LIKE '{tahun}-{bulan_no}%'"

    # Tidak ada periode → kosong (tampilkan semua)
    return ""


# =====================================================
# MOCK SQL (mode DEMO tanpa API key)
# =====================================================

def _mock_sql(pertanyaan: str) -> str:
    TBL    = NAMA_TABEL  # alias agar query pakai nama tabel aktual
    q      = pertanyaan.lower().strip()
    filter_periode = deteksi_periode(q)
    where_periode  = f"AND {filter_periode}" if filter_periode else ""
    # Untuk query yang dimulai dengan WHERE (bukan AND)
    where_periode_standalone = f"WHERE {filter_periode}" if filter_periode else ""

    m = re.search(r"top\s+(\d+)", q)
    if not m:
        m = re.search(r"(\d+)", q)
    top_n = int(m.group(1)) if m else 10

    # ── Belum ditindaklanjuti ────────────────────────
    if "belum" in q and ("tindak" in q or "ditindaklanjuti" in q):
        if filter_periode:
            return (
                f"SELECT kode_area, tema, latar_belakang, tanggal_deadline\n"
                f"FROM lks_bipartit\n"
                f"WHERE status_tindak_lanjut = 'Belum'\n"
                f"  AND {filter_periode}"
            )
        return (
            "SELECT kode_area, tema, latar_belakang, tanggal_deadline\n"
            "FROM lks_bipartit\n"
            "WHERE status_tindak_lanjut = 'Belum'"
        )

    # ── Deadline lewat ───────────────────────────────
    if "deadline" in q and ("lewat" in q or "terlewat" in q or "melewati" in q):
        if filter_periode:
            return (
                f"SELECT kode_area, tema, tanggal_deadline\n"
                f"FROM lks_bipartit\n"
                f"WHERE status_tindak_lanjut = 'Belum'\n"
                f"  AND tanggal_deadline < DATE('now')\n"
                f"  AND {filter_periode}\n"
                f"ORDER BY tanggal_deadline"
            )
        return (
            "SELECT kode_area, tema, tanggal_deadline\n"
            "FROM lks_bipartit\n"
            "WHERE status_tindak_lanjut = 'Belum'\n"
            "  AND tanggal_deadline < DATE('now')\n"
            "ORDER BY tanggal_deadline"
        )

    # ── Per tema ─────────────────────────────────────
    if "per tema" in q or "tiap tema" in q or ("berapa" in q and "tema" in q) \
            or "banyak diperbincangkan" in q or "paling banyak dibahas" in q \
            or "terbanyak" in q or "paling banyak" in q \
            or ("tema" in q and ("apa" in q or "banyak" in q
                                 or "dibahas" in q or "diperbincangkan" in q)):
        if filter_periode:
            return (
                f"SELECT tema, COUNT(*) AS jumlah\n"
                f"FROM lks_bipartit\n"
                f"WHERE {filter_periode}\n"
                f"GROUP BY tema\n"
                f"ORDER BY jumlah DESC"
            )
        return (
            "SELECT tema, COUNT(*) AS jumlah\n"
            "FROM lks_bipartit\n"
            "GROUP BY tema\n"
            "ORDER BY jumlah DESC"
        )

    # ── Per area ─────────────────────────────────────
    if "per area" in q or "tiap area" in q or ("berapa" in q and "area" in q):
        if filter_periode:
            return (
                f"SELECT kode_area, COUNT(*) AS jumlah\n"
                f"FROM lks_bipartit\n"
                f"WHERE {filter_periode}\n"
                f"GROUP BY kode_area\n"
                f"ORDER BY jumlah DESC"
            )
        return (
            "SELECT kode_area, COUNT(*) AS jumlah\n"
            "FROM lks_bipartit\n"
            "GROUP BY kode_area\n"
            "ORDER BY jumlah DESC"
        )

    # ── Detail / Rincian pembahasan tema tertentu ────
    # Contoh: "detail lain-lain juli", "rincian SPPD bulan maret", "pembahasan PKB di juni"
    # Semua tema dari data aktual — key = kata yang mungkin diketik user
    TEMA_KEYWORDS = {
        # Lain-lain
        "lain-lain": "lain-lain",  "lain lain": "lain-lain",
        "lainnya": "lain-lain",    "lain": "lain-lain",
        # LKS Bipartit
        "lks bipartit": "lks bipartit", "lks": "lks bipartit",
        "bipartit": "lks bipartit",
        # Kebijakan Fasilitas Kesehatan
        "kebijakan dan fasilitas kesehatan": "kebijakan dan fasilitas kesehatan",
        "kebijakan fasilitas": "kebijakan dan fasilitas kesehatan",
        "fasilitas kesehatan": "kebijakan dan fasilitas kesehatan",
        "kebijakan kesehatan": "kebijakan dan fasilitas kesehatan",
        "kesehatan": "kebijakan dan fasilitas kesehatan",
        # SPPD
        "sppd": "sppd", "perjalanan dinas": "sppd", "reimburse": "sppd",
        # PKB
        "pkb": "perjanjian kerja bersama (pkb)",
        "perjanjian kerja bersama": "perjanjian kerja bersama (pkb)",
        "perjanjian kerja": "perjanjian kerja bersama (pkb)",
        # Manajemen Talenta dan Pegawai
        "manajemen talenta": "manajemen talenta dan pegawai",
        "talenta dan pegawai": "manajemen talenta dan pegawai",
        "talenta pegawai": "manajemen talenta dan pegawai",
        "talenta": "manajemen talenta dan pegawai",
        # Manajemen Penghargaan Pegawai
        "manajemen penghargaan": "manajemen penghargaan pegawai",
        "penghargaan pegawai": "manajemen penghargaan pegawai",
        "penghargaan": "manajemen penghargaan pegawai",
        # Manajemen Pengembangan Pegawai
        "manajemen pengembangan": "manajemen pengembangan pegawai",
        "pengembangan pegawai": "manajemen pengembangan pegawai",
        # Manajemen Kinerja Pegawai
        "manajemen kinerja": "manajemen kinerja pegawai",
        "kinerja pegawai": "manajemen kinerja pegawai",
        "kinerja": "manajemen kinerja pegawai",
        # Manajemen Hubungan Industrial
        "manajemen hubungan industrial": "manajemen hubungan industrial",
        "hubungan industrial": "manajemen hubungan industrial",
        # Manajemen Akuisisi Pegawai
        "manajemen akuisisi": "manajemen akuisisi pegawai",
        "akuisisi pegawai": "manajemen akuisisi pegawai",
        "rekrutmen": "manajemen akuisisi pegawai",
        # Fasilitas Kantor
        "fasilitas kantor": "fasilitas kantor",
        "fasilitas": "fasilitas kantor",
        # Mutasi Pegawai
        "mutasi pegawai": "mutasi pegawai", "mutasi": "mutasi pegawai",
        # Jam Kerja Pegawai
        "jam kerja pegawai": "jam kerja pegawai",
        "jam kerja": "jam kerja pegawai",
        # Disiplin Pegawai
        "disiplin pegawai": "disiplin pegawai", "disiplin": "disiplin pegawai",
        # Absensi Pegawai
        "absensi pegawai": "absensi pegawai", "absensi": "absensi pegawai",
        # Pembinaan Pegawai
        "pembinaan pegawai": "pembinaan pegawai", "pembinaan": "pembinaan pegawai",
        # Wellbeing
        "wellbeing": "wellbeing", "kesejahteraan": "wellbeing",
        # K3L
        "k3l": "k3l", "keselamatan kerja": "k3l", "k3": "k3l",
        # FTK
        "ftk": "ftk",
        # Diklat
        "diklat": "diklat", "pelatihan": "diklat",
        # Kebijakan Lembur
        "kebijakan lembur": "kebijakan lembur",
        "lembur": "kebijakan lembur",
        # Serikat Pekerja
        "serikat pekerja": "serikat pekerja", "sp ": "serikat pekerja",
        "sp pln": "serikat pekerja",
        # Organisasi dan Tata Kerja
        "organisasi dan tata kerja": "organisasi dan tata kerja",
        "tata kerja": "organisasi dan tata kerja",
        "organisasi": "organisasi dan tata kerja",
        # Tenaga Alih Daya
        "tenaga alih daya": "tenaga alih daya (tad)",
        "tad": "tenaga alih daya (tad)",
        "alih daya": "tenaga alih daya (tad)",
        # Kegiatan Operasional Unit
        "kegiatan operasional": "kegiatan operasional unit",
        "operasional unit": "kegiatan operasional unit",
        # P2-1B Khusus
        "p2-1b": "p2-1b khusus", "p2 1b": "p2-1b khusus",
        # Data Kepegawaian
        "data kepegawaian": "data kepegawaian",
        # Kinerja Perusahaan
        "kinerja perusahaan": "kinerja perusahaan",
        # Employee Gathering
        "employee gathering": "employee gathering", "gathering": "employee gathering",
        # Budaya PLN
        "budaya pln": "budaya pln", "budaya": "budaya pln",
        # Inovasi
        "inovasi": "inovasi",
        # SBO
        "sbo": "sbo",
        # SMAP
        "smap": "smap",
        # Program Pendidikan Formal
        "program pendidikan": "program pendidikan formal",
        "pendidikan formal": "program pendidikan formal",
        # Clean Energy Day
        "clean energy": "clean energy day", "ced": "clean energy day",
        # Capacity Building
        "capacity building": "capacity building tim lks bipartit",
        # Evaluasi PKB
        "evaluasi pkb": "evaluasi pelaksanaan sosialisasi berita acara turunan pkb",
        "sosialisasi pkb": "evaluasi pelaksanaan sosialisasi berita acara turunan pkb",
        # Cascading KPI
        "cascading": "penyelarasan visi, target kinerj/cascading  kpi dan hubungan industrial tahun 2026",
        "cascading kpi": "penyelarasan visi, target kinerj/cascading  kpi dan hubungan industrial tahun 2026",
    }
    _is_detail = any(w in q for w in [
        "detail", "detil", "rincian", "rinci", "pembahasan",
        "tampilkan", "lihat", "apa saja", "list"
    ])
    _tema_match = None
    for _tk, _tv in TEMA_KEYWORDS.items():
        if _tk in q:
            _tema_match = _tv
            break
    # Jika ada tema + periode → otomatis detail
    if not _is_detail and _tema_match and filter_periode:
        _is_detail = True

    if _is_detail and _tema_match:
        _cols = (
            "kode_area, tema, latar_belakang, rekomendasi, "
            "tanggal_deadline, tindak_lanjut, tanggal_tindak_lanjut, "
            "status_tindak_lanjut"
        )
        if filter_periode:
            return (
                f"SELECT {_cols}\n"
                f"FROM lks_bipartit\n"
                f"WHERE LOWER(tema) LIKE '%{_tema_match}%'\n"
                f"  AND {filter_periode}\n"
                f"ORDER BY tanggal_deadline NULLS LAST"
            )
        return (
            f"SELECT {_cols}\n"
            f"FROM lks_bipartit\n"
            f"WHERE LOWER(tema) LIKE '%{_tema_match}%'\n"
            f"ORDER BY tanggal_deadline NULLS LAST"
        )

    # ── PKB ──────────────────────────────────────────
    if "pkb" in q or "perjanjian kerja" in q:
        if filter_periode:
            return (
                f"SELECT kode_area, tema, rekomendasi, tanggal_deadline\n"
                f"FROM lks_bipartit\n"
                f"WHERE (LOWER(tema) LIKE '%pkb%' OR LOWER(tema) LIKE '%perjanjian kerja%')\n"
                f"  AND {filter_periode}"
            )
        return (
            "SELECT kode_area, tema, rekomendasi, tanggal_deadline\n"
            "FROM lks_bipartit\n"
            "WHERE LOWER(tema) LIKE '%pkb%'\n"
            "   OR LOWER(tema) LIKE '%perjanjian kerja%'"
        )

    # ── Kesehatan ────────────────────────────────────
    if "kesehatan" in q:
        if filter_periode:
            return (
                f"SELECT kode_area, tema, latar_belakang, rekomendasi\n"
                f"FROM lks_bipartit\n"
                f"WHERE LOWER(tema) LIKE '%kesehatan%'\n"
                f"  AND {filter_periode}"
            )
        return (
            "SELECT kode_area, tema, latar_belakang, rekomendasi\n"
            "FROM lks_bipartit\n"
            "WHERE LOWER(tema) LIKE '%kesehatan%'"
        )

    # ── SPPD ─────────────────────────────────────────
    if "sppd" in q:
        if filter_periode:
            return (
                f"SELECT kode_area, tema, latar_belakang, rekomendasi\n"
                f"FROM lks_bipartit\n"
                f"WHERE LOWER(tema) LIKE '%sppd%'\n"
                f"  AND {filter_periode}"
            )
        return (
            "SELECT kode_area, tema, latar_belakang, rekomendasi\n"
            "FROM lks_bipartit\n"
            "WHERE LOWER(tema) LIKE '%sppd%'"
        )

    # ── Wellbeing ────────────────────────────────────
    if "wellbeing" in q:
        if filter_periode:
            return (
                f"SELECT kode_area, tema, latar_belakang, rekomendasi\n"
                f"FROM lks_bipartit\n"
                f"WHERE LOWER(tema) LIKE '%wellbeing%'\n"
                f"  AND {filter_periode}"
            )
        return (
            "SELECT kode_area, tema, latar_belakang, rekomendasi\n"
            "FROM lks_bipartit\n"
            "WHERE LOWER(tema) LIKE '%wellbeing%'"
        )

    # ── Total / jumlah ───────────────────────────────
    if "berapa" in q or "jumlah" in q or "total" in q:
        if filter_periode:
            return (
                f"SELECT COUNT(*) AS total_pertemuan\n"
                f"FROM lks_bipartit\n"
                f"WHERE {filter_periode}"
            )
        return "SELECT COUNT(*) AS total_pertemuan FROM lks_bipartit"

    # ── Default ──────────────────────────────────────
    if filter_periode:
        return (
            f"SELECT kode_area, tema, tanggal_pelaksanaan, status_tindak_lanjut\n"
            f"FROM lks_bipartit\n"
            f"WHERE {filter_periode}\n"
            f"LIMIT {top_n}"
        )
    return (
        f"SELECT kode_area, tema, tanggal_pelaksanaan, status_tindak_lanjut\n"
        f"FROM lks_bipartit\n"
        f"LIMIT {top_n}"
    )


def narasi_mock(pertanyaan: str) -> str:
    """Jawaban narasi DEMO untuk pertanyaan yang tidak bisa di-SQL."""
    q = pertanyaan.lower()
    if any(w in q for w in ["rekap", "summary", "ringkasan", "gambaran"]):
        return (
            "Data LKS Bipartit mencakup **1.263 pertemuan** dari Februari–April 2026, "
            "tersebar di **47 area** dengan **41 tema** berbeda. "
            "Tema terbanyak: Manajemen Penghargaan Pegawai, LKS Bipartit, dan Kesehatan. "
            "Dari total, sekitar **856 sudah ditindaklanjuti** dan **407 belum**."
        )
    return ""


def tanya_llm(prompt: str) -> str:
    if PROVIDER == "gemini":
        return panggil_gemini(prompt)
    elif PROVIDER == "groq":
        return panggil_groq(prompt)
    else:
        bagian = prompt.split("=== PERTANYAAN USER ===")
        pertanyaan = bagian[-1].strip() if len(bagian) > 1 else prompt
        return _mock_sql(pertanyaan)


def tanya_fallback_llm(pertanyaan: str) -> str:
    """
    Dipanggil saat query SQL gagal.
    LLM menganalisis pertanyaan dan menyarankan alternatif yang valid.
    """
    prompt_fb = f"""
Anda adalah asisten analitik data LKS Bipartit PLN yang ramah.
User mengajukan pertanyaan yang tidak bisa dijawab oleh sistem.

Pertanyaan user: "{pertanyaan}"

Data yang tersedia hanya memiliki kolom:
kode_area, kode_sub_area, tanggal_pelaksanaan, tema, latar_belakang,
rekomendasi, tanggal_deadline, tindak_lanjut, tanggal_tindak_lanjut,
status_tindak_lanjut (nilai: 'Sudah' atau 'Belum')

Tugas Anda:
1. Jelaskan secara singkat mengapa pertanyaan ini sulit dijawab (1 kalimat, TANPA menyebut SQL/database/teknis)
2. Berikan tepat 3 pertanyaan alternatif yang bisa dijawab sistem, dimulai dengan bullet "•"
3. Gunakan bahasa Indonesia ramah dan singkat

Contoh format jawaban:
Pertanyaan tersebut membutuhkan informasi yang belum tersedia dalam data saat ini.
Coba salah satu pertanyaan berikut:
• Tema apa yang paling banyak dibahas?
• Berapa pertemuan yang belum ditindaklanjuti?
• Rekomendasi terkait PKB di bulan Februari?
""".strip()

    if PROVIDER == "gemini":
        return panggil_gemini(prompt_fb)
    elif PROVIDER == "groq":
        return panggil_groq(prompt_fb)
    else:
        # Mode DEMO — saran statis berdasarkan kata kunci
        q = pertanyaan.lower()
        if any(w in q for w in ["siapa","nama","nip","jabatan","pegawai tertentu"]):
            saran = ["Berapa total pertemuan per area?",
                     "Tema yang paling banyak dibahas?",
                     "Permasalahan yang belum ditindaklanjuti?"]
        elif any(w in q for w in ["grafik","chart","visualisasi","diagram"]):
            saran = ["Tema yang paling banyak dibahas? (akan tampil sebagai chart)",
                     "Berapa pertemuan per area?",
                     "Permasalahan belum ditindaklanjuti per tema?"]
        elif any(w in q for w in ["predict","prediksi","forecast","masa depan"]):
            saran = ["Deadline yang sudah lewat dan belum selesai?",
                     "Berapa pertemuan belum ditindaklanjuti?",
                     "Tema terbanyak bulan April 2026?"]
        else:
            saran = ["Tema yang paling banyak dibahas?",
                     "Permasalahan yang belum ditindaklanjuti?",
                     "Rekomendasi terkait PKB?"]
        saran_str = "\n".join(f"• {s}" for s in saran)
        return f"Pertanyaan tersebut belum bisa dijawab dari data yang tersedia.\nCoba salah satu pertanyaan berikut:\n{saran_str}"

# =====================================================
# SKEMA & CONTOH
# =====================================================

DIALEK = "PostgreSQL (Supabase)"

SKEMA = """
lks_bipartit(
    kode_area            : kode/nama area PLN,
    kode_sub_area        : kode/nama sub area / unit kerja spesifik,
    tanggal_pelaksanaan  : tanggal pertemuan LKS Bipartit (YYYY-MM-DD),
    tema                 : tema permasalahan yang dibahas,
    latar_belakang       : uraian latar belakang permasalahan,
    rekomendasi          : rekomendasi hasil pertemuan,
    tanggal_deadline     : batas waktu penyelesaian (YYYY-MM-DD),
    tindak_lanjut        : uraian tindak lanjut yang sudah dilakukan,
    tanggal_tindak_lanjut: tanggal tindak lanjut dilakukan (YYYY-MM-DD),
    status_tindak_lanjut : 'Sudah' atau 'Belum'
)
""".strip()

CONTOH = """
Contoh query:

Q: Tema yang paling banyak dibahas?
A: SELECT tema, COUNT(*) AS jumlah FROM lks_bipartit GROUP BY tema ORDER BY jumlah DESC

Q: Tema yang paling banyak dibahas bulan Februari 2026?
A: SELECT tema, COUNT(*) AS jumlah FROM lks_bipartit WHERE tanggal_pelaksanaan LIKE '2026-02%' GROUP BY tema ORDER BY jumlah DESC

Q: Berapa pertemuan per area?
A: SELECT kode_area, COUNT(*) AS jumlah FROM lks_bipartit GROUP BY kode_area ORDER BY jumlah DESC

Q: Berapa pertemuan per area bulan Maret 2026?
A: SELECT kode_area, COUNT(*) AS jumlah FROM lks_bipartit WHERE tanggal_pelaksanaan LIKE '2026-03%' GROUP BY kode_area ORDER BY jumlah DESC

Q: Permasalahan yang belum ditindaklanjuti?
A: SELECT kode_area, tema, latar_belakang, tanggal_deadline FROM lks_bipartit WHERE status_tindak_lanjut = 'Belum'

Q: Permasalahan yang belum ditindaklanjuti bulan April 2026?
A: SELECT kode_area, tema, latar_belakang, tanggal_deadline FROM lks_bipartit WHERE status_tindak_lanjut = 'Belum' AND tanggal_pelaksanaan LIKE '2026-04%'

Q: Berapa total pertemuan?
A: SELECT COUNT(*) AS total_pertemuan FROM lks_bipartit

Q: Berapa total pertemuan bulan Februari?
A: SELECT COUNT(*) AS total_pertemuan FROM lks_bipartit WHERE tanggal_pelaksanaan LIKE '2026-02%'

ATURAN PERIODE:
- Jika pertanyaan menyebut bulan/periode → tambahkan WHERE tanggal_pelaksanaan LIKE 'YYYY-MM%'
- Jika tidak ada periode → jangan tambahkan filter tanggal (tampilkan semua data)
"""


def bangun_prompt(pertanyaan: str) -> str:
    return f"""
Anda adalah expert SQL untuk sistem HR / Hubungan Industrial PLN ({DIALEK}).

Aturan:
- Gunakan HANYA tabel dan kolom pada SKEMA
- Jangan mengarang kolom baru
- Hanya boleh 1 query SELECT
- Jika ada "terbanyak", "tertinggi", "paling" → gunakan ORDER BY (tanpa LIMIT kecuali diminta)
- Jika ada "berapa" → gunakan COUNT(*)
- Untuk pencarian teks → gunakan LOWER(...) LIKE '%keyword%'
- Kolom status_tindak_lanjut hanya bernilai 'Sudah' atau 'Belum'
- Format tanggal: YYYY-MM-DD
- PENTING: Jika pertanyaan menyebut bulan/periode → filter dengan tanggal_pelaksanaan LIKE 'YYYY-MM%'
- PENTING: Jika tidak ada periode → JANGAN tambahkan filter tanggal (tampilkan semua data)
- Jawab HANYA SQL query, tanpa penjelasan, tanpa markdown

=== SKEMA ===
{SKEMA}

=== CONTOH ===
{CONTOH}

=== PERTANYAAN USER ===
{pertanyaan}
""".strip()

# =====================================================
# BERSIHKAN SQL
# =====================================================

def bersihkan_sql(teks: str) -> str:
    teks = teks.strip()
    try:
        obj = json.loads(teks)
        if isinstance(obj, dict) and "sql" in obj:
            teks = obj["sql"].strip()
    except Exception:
        pass
    teks = re.sub(r"^```[a-zA-Z]*\n?", "", teks, flags=re.MULTILINE)
    teks = re.sub(r"```$", "", teks, flags=re.MULTILINE)
    teks = re.sub(r"</?[a-zA-Z]+>", "", teks)
    teks = re.sub(r"\bLIMT\b",   "LIMIT",  teks, flags=re.IGNORECASE)
    teks = re.sub(r"\bSEELCT\b", "SELECT", teks, flags=re.IGNORECASE)
    teks = re.sub(r"\bFRM\b",    "FROM",   teks, flags=re.IGNORECASE)
    return teks.strip()

# =====================================================
# GUARDRAIL — VALIDASI SQL
# =====================================================

TERLARANG = (
    "drop", "delete", "update", "insert",
    "alter", "truncate", "create", "replace",
    "grant", "revoke", "copy", "call", "exec",
)


def validasi_sql(sql: str, maks_baris: int = 1000) -> str:
    teks = sql.strip().rstrip(";").strip()
    low  = teks.lower()

    if not low.lstrip().startswith("select"):
        raise ValueError("❌ Hanya SELECT yang diizinkan")
    if ";" in teks and not teks.endswith(";"):
        raise ValueError("❌ Multi-statement tidak diizinkan")
    for kata in TERLARANG:
        if re.search(rf"\b{kata}\b", low):
            raise ValueError(f"❌ Operasi terlarang: {kata}")
    if "lks_bipartit" not in low and NAMA_TABEL.lower() not in low:
        raise ValueError(f"❌ Hanya tabel lks_bipartit / {NAMA_TABEL} yang diizinkan")

    teks = re.sub(r"\bLIMT\b",   "LIMIT",  teks, flags=re.IGNORECASE)
    teks = re.sub(r"\bSEELCT\b", "SELECT", teks, flags=re.IGNORECASE)
    teks = re.sub(r"\bFRM\b",    "FROM",   teks, flags=re.IGNORECASE)

    # Paksa LIMIT hanya untuk query detail (bukan agregasi)
    if "limit" not in teks.lower() and "count(" not in teks.lower() \
            and "group by" not in teks.lower():
        teks += f" LIMIT {maks_baris}"

    return teks

# =====================================================
# FUNGSI UTAMA
# =====================================================

def buat_sql(pertanyaan: str) -> str:
    prompt = bangun_prompt(pertanyaan)
    sql    = tanya_llm(prompt)
    return bersihkan_sql(sql)


def render_detail_pembahasan(df_hasil: "pd.DataFrame"):
    """
    Render kartu pembahasan yang bisa di-expand.
    Tiap kartu menampilkan tema + status, dan saat diklik expand:
    latar belakang, rekomendasi, tindak lanjut, deadline.
    """
    if "status_tindak_lanjut" not in df_hasil.columns:
        return
    if "latar_belakang" not in df_hasil.columns:
        return

    _n_sudah = (df_hasil["status_tindak_lanjut"] == "Sudah").sum()
    _n_belum = (df_hasil["status_tindak_lanjut"] == "Belum").sum()

    # Ringkasan scorecard
    _sc1, _sc2, _sc3 = st.columns(3)
    _sc1.metric("Total", f"{len(df_hasil):,}")
    _sc2.metric("✅ Sudah TL", f"{int(_n_sudah):,}")
    _sc3.metric("⏳ Belum TL", f"{int(_n_belum):,}")
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # Kartu expandable per baris
    for _idx, (_, _row) in enumerate(df_hasil.iterrows()):
        _sudah       = str(_row.get("status_tindak_lanjut","")) == "Sudah"
        _border      = "#2E7D32" if _sudah else "#C0392B"
        _bg          = "#F1F8F1" if _sudah else "#FDF0EE"
        _badge_color = "#2E7D32" if _sudah else "#C0392B"
        _badge_icon  = "✅" if _sudah else "⏳"
        _badge_text  = "Sudah" if _sudah else "Belum"
        _deadline    = str(_row.get("tanggal_deadline","") or "—")
        _area        = str(_row.get("kode_area","") or "")
        _tema        = str(_row.get("tema","") or "—")
        _lb          = str(_row.get("latar_belakang","") or "—")
        _rek         = str(_row.get("rekomendasi","") or "—")
        _tl          = str(_row.get("tindak_lanjut","") or "").strip()
        _tgl_tl      = str(_row.get("tanggal_tindak_lanjut","") or "")

        # Header kartu (selalu tampil)
        _header = (
            f"<div style='border-left:4px solid {_border};"
            f"background:{_bg};border-radius:0 6px 6px 0;"
            f"padding:10px 14px;cursor:pointer;'>"
            f"<div style='display:flex;justify-content:space-between;"
            f"align-items:center;'>"
            f"<div>"
            f"<span style='font-family:JetBrains Mono,monospace;"
            f"font-size:0.6rem;color:#6B5B4E;'>Area {_area} · Deadline: {_deadline}</span><br>"
            f"<span style='font-weight:700;font-size:0.9rem;color:#0A0204;'>{_tema}</span>"
            f"</div>"
            f"<span style='font-family:JetBrains Mono,monospace;"
            f"font-size:0.72rem;font-weight:700;color:{_badge_color};"
            f"white-space:nowrap;margin-left:12px;'>{_badge_icon} {_badge_text}</span>"
            f"</div></div>"
        )
        st.markdown(_header, unsafe_allow_html=True)

        # Detail (expand on click via st.expander)
        with st.expander("Lihat detail →", expanded=False):
            _d1, _d2 = st.columns(2)
            with _d1:
                st.markdown(
                    "<span style='font-family:JetBrains Mono,monospace;"
                    "font-size:0.6rem;text-transform:uppercase;"
                    "letter-spacing:0.08em;color:#6B5B4E;'>Latar Belakang</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div style='font-size:0.875rem;color:#1A0A0D;"
                    f"line-height:1.55;'>{_lb}</div>",
                    unsafe_allow_html=True,
                )
            with _d2:
                st.markdown(
                    "<span style='font-family:JetBrains Mono,monospace;"
                    "font-size:0.6rem;text-transform:uppercase;"
                    "letter-spacing:0.08em;color:#6B5B4E;'>Rekomendasi</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div style='font-size:0.875rem;color:#1A0A0D;"
                    f"line-height:1.55;'>{_rek}</div>",
                    unsafe_allow_html=True,
                )
            if _tl:
                st.markdown(
                    "<span style='font-family:JetBrains Mono,monospace;"
                    "font-size:0.6rem;text-transform:uppercase;"
                    "letter-spacing:0.08em;color:#2E7D32;margin-top:8px;"
                    "display:block;'>Tindak Lanjut</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div style='font-size:0.875rem;color:#2E7D32;"
                    f"font-style:italic;line-height:1.55;'>{_tl}"
                    f"{'  ·  ' + _tgl_tl if _tgl_tl else ''}</div>",
                    unsafe_allow_html=True,
                )
        st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)


def jalankan_query(pertanyaan: str):
    if not DATA_SIAP:
        return None, None, DATA_ERROR
    try:
        sql_mentah = buat_sql(pertanyaan)
        sql_aman   = validasi_sql(sql_mentah)
        with engine.connect() as conn:
            df = pd.read_sql(text(sql_aman), conn)
        return df, sql_aman, None
    except Exception as e:
        return None, None, str(e)

# =====================================================
# RAG — DOKUMEN via Supabase Storage
# Upload PDF ke bucket "DOKUMEN RAG" di Supabase.
# Sistem otomatis membaca semua file dari cloud.
# =====================================================

import io, pathlib, textwrap

# Supabase Storage config
RAG_BUCKET   = "DOKUMEN RAG"
RAG_BASE_URL = f"{SUPABASE_URL}/storage/v1/object/public/{RAG_BUCKET.replace(' ', '%20')}"
RAG_API_URL  = f"{SUPABASE_URL}/storage/v1/object/list/{RAG_BUCKET.replace(' ', '%20')}"

# Cache — diisi saat startup
_rag_index: dict = {}
_rag_urls:  dict = {}
_rag_files: list = []   # daftar nama file di bucket

# ── Daftar file manual (fallback jika API list kosong) ───────────────
# Tambahkan nama file baru di sini setelah upload ke Supabase Storage
RAG_FILE_MANUAL = [
    "EDIR 001 TENTANG SPPD.pdf",
    "PKB 2025.pdf",
    "PKB 2026.pdf",
]

def _get_public_url(file_info) -> str:
    """Buat public URL dari nama file atau dict info file."""
    if isinstance(file_info, str):
        nama = file_info
    else:
        nama = file_info.get("name", "")
    bkt = RAG_BUCKET.replace(" ", "%20")
    return f"{SUPABASE_URL}/storage/v1/object/public/{bkt}/{nama.replace(' ', '%20')}"

def _cek_url_exist(url: str) -> bool:
    """Cek apakah URL file bisa diakses (HEAD request)."""
    try:
        resp = _req.head(url, timeout=5)
        return resp.status_code == 200
    except Exception:
        return False

def _list_dokumen_supabase() -> list[dict]:
    """
    Ambil daftar file dari Supabase Storage.
    Prioritas: API list → fallback manual URL check.
    """
    # Coba API list dulu
    try:
        bkt  = RAG_BUCKET
        url  = f"{SUPABASE_URL}/storage/v1/object/list/{bkt}"
        resp = _req.post(
            url, headers=_sb_headers(),
            json={"limit": 100, "offset": 0, "prefix": ""},
            timeout=10,
        )
        print(f"  Storage [{bkt}]: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and data:
                hasil = [
                    {**f, "_bucket": bkt}
                    for f in data
                    if isinstance(f, dict)
                    and f.get("name", "").lower().endswith((".pdf", ".txt", ".md"))
                ]
                if hasil:
                    print(f"  ✅ API list: {len(hasil)} file")
                    return hasil
    except Exception as e:
        print(f"  ⚠️ API list gagal: {e}")

    # Fallback: cek manual URL per file yang sudah diketahui
    print("  ℹ️  Fallback ke daftar file manual...")
    hasil_manual = []
    for nama in RAG_FILE_MANUAL:
        pub_url = _get_public_url(nama)
        if _cek_url_exist(pub_url):
            hasil_manual.append({"name": nama, "_bucket": RAG_BUCKET, "_url": pub_url})
            print(f"  ✅ {nama}: accessible")
        else:
            print(f"  ⚠️ {nama}: tidak bisa diakses")
    print(f"  Total manual: {len(hasil_manual)} file")
    return hasil_manual

def _baca_pdf_dari_url(url: str) -> str:
    """Download PDF dari URL Supabase dan ekstrak teks."""
    try:
        resp = _req.get(url, timeout=20)
        resp.raise_for_status()
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
                return "\n".join(p.extract_text() or "" for p in pdf.pages)
        except ImportError:
            try:
                import PyPDF2
                r = PyPDF2.PdfReader(io.BytesIO(resp.content))
                return "\n".join(p.extract_text() or "" for p in r.pages)
            except Exception:
                return resp.text
    except Exception as e:
        print(f"⚠️ Gagal baca PDF {url}: {e}")
        return ""

def _chunk_teks(teks: str, ukuran: int = 400, overlap: int = 80) -> list[str]:
    """Potong teks menjadi chunk dengan overlap."""
    kata = teks.split()
    chunks, i = [], 0
    while i < len(kata):
        chunk = " ".join(kata[i:i+ukuran])
        if chunk.strip():
            chunks.append(chunk)
        i += ukuran - overlap
    return chunks

def bangun_rag_index() -> dict:
    """Download & index semua dokumen dari Supabase Storage."""
    idx, urls = {}, {}
    files = _list_dokumen_supabase()
    print(f"  RAG: {len(files)} file akan diindeks")
    for f in files:
        nama = f["name"]
        url  = f.get("_url") or _get_public_url(f)
        print(f"  Mengindeks: {nama}")
        teks = _baca_pdf_dari_url(url)
        if teks.strip():
            idx[nama]  = _chunk_teks(teks)
            urls[nama] = url
            print(f"  ✅ {nama}: {len(idx[nama])} chunks")
        else:
            print(f"  ⚠️ {nama}: teks kosong")
    _rag_urls.update(urls)
    return idx

def cari_dokumen(pertanyaan: str, top_k: int = 3) -> list[dict]:
    """
    Cari chunk paling relevan dari index RAG.
    Pakai TF sederhana + cek kata Indonesia & Inggris.
    """
    global _rag_index
    # Rebuild jika index kosong
    if not _rag_index:
        try:
            _rag_index = bangun_rag_index()
        except Exception as e:
            print(f"⚠️  RAG index gagal: {e}")
            return []

    # Gabung kata query Indonesia + Inggris (min 2 karakter)
    kata_query = set(re.findall(r"[a-zA-ZÀ-ÿ]{2,}", pertanyaan.lower()))
    hasil = []
    for fname, chunks in _rag_index.items():
        for i, chunk in enumerate(chunks):
            kata_chunk = set(re.findall(r"[a-zA-ZÀ-ÿ]{2,}", chunk.lower()))
            skor = len(kata_query & kata_chunk)
            # Boost skor jika nama file disebut dalam pertanyaan
            fname_clean = re.sub(r"[_.\- ]", " ", fname.lower())
            if any(w in fname_clean for w in kata_query if len(w) > 2):
                skor += 3
            if skor > 0:
                hasil.append({
                    "file":  fname,
                    "chunk": chunk,
                    "skor":  skor,
                    "bagian": f"Bag. {i+1}/{len(chunks)}",
                })
    hasil.sort(key=lambda x: x["skor"], reverse=True)
    return hasil[:top_k]

def jawab_dengan_rag(pertanyaan: str, konteks_chunks: list[dict]) -> str:
    """Minta LLM menjawab pertanyaan berdasarkan potongan dokumen."""
    if not konteks_chunks:
        return ""

    konteks_str = "\n\n---\n\n".join(
        f"[{c['file']} — {c['bagian']}]\n{c['chunk']}"
        for c in konteks_chunks
    )

    prompt = f"""
Anda adalah asisten PLN yang membantu menjawab pertanyaan berdasarkan dokumen internal.

Gunakan HANYA informasi dari dokumen berikut untuk menjawab:

{konteks_str}

---
Pertanyaan: {pertanyaan}

Instruksi:
- Jawab dalam Bahasa Indonesia yang ramah dan jelas
- Sebutkan nama file sumber di akhir jawaban dalam format: (Sumber: nama_file)
- Jika dokumen tidak memuat jawaban, katakan "Informasi ini belum tersedia dalam dokumen yang ada"
- Jangan mengarang informasi yang tidak ada di dokumen
""".strip()

    if PROVIDER == "gemini":
        return panggil_gemini(prompt)
    elif PROVIDER == "groq":
        return panggil_groq(prompt)
    else:
        # Mode DEMO: gabungkan ringkasan dari semua chunk relevan
        bagian = []
        for c in konteks_chunks[:2]:
            ringkas = textwrap.shorten(c["chunk"], width=400, placeholder="...")
            bagian.append(f"📄 **{c['file']}** ({c['bagian']}):\n{ringkas}")
        jawaban = "\n\n".join(bagian)
        return (
            f"Berdasarkan dokumen yang tersedia:\n\n{jawaban}\n\n"
            f"(Untuk jawaban lebih lengkap, aktifkan Gemini dengan set GEMINI_API_KEY)"
        )

def is_pertanyaan_dokumen(pertanyaan: str) -> bool:
    """
    Deteksi EKSPLISIT apakah pertanyaan minta isi dokumen/regulasi.
    Pertanyaan analitik biasa (tema, area, tindak lanjut) → False → SQL.
    Hanya yang jelas minta isi file → True → RAG.
    """
    q = pertanyaan.lower()

    # Pola eksplisit minta dokumen/regulasi
    kata_eksplisit = [
        "surat edaran", "edaran direksi", "perdir ", "peraturan direksi",
        "sk direksi", "edir ", "edir0", "edir 0",
        "isi dokumen", "isi edaran", "isi pkb", "isi perdir", "isi surat",
        "ada dokumen", "ada peraturan", "ada surat", "ada edaran",
        "menurut pkb", "menurut peraturan", "menurut edaran",
        "sesuai pkb", "sesuai peraturan", "diatur dalam pkb",
        "dasar hukum", "landasan hukum", "referensi regulasi",
        "pasal berapa", "bab berapa", "nomor surat", "nomor edaran",
        "bunyi pasal", "isi pasal", "ketentuan pkb", "ketentuan edaran",
        # Nama file eksplisit
        "edir 001", "pkb 2025", "pkb 2026",
    ]
    if any(k in q for k in kata_eksplisit):
        return True

    # Cek nama file bucket yang disebut persis dalam pertanyaan
    try:
        _files = _list_dokumen_supabase()
        for _f in _files:
            _base = _f["name"].lower().replace("_"," ").replace(".pdf","").replace(".txt","")
            if _base in q:
                return True
    except Exception:
        pass

    return False

# =====================================================
# HELPER: deteksi apakah hasil cocok untuk chart
# =====================================================

def deteksi_chart(df: pd.DataFrame):
    """
    Kembalikan tipe chart yang cocok berdasarkan struktur DataFrame:
    - 'metric' : 1 kolom 1 baris angka (hasil COUNT/SUM tunggal)
    - 'pie'    : 2 kolom, kolom ke-2 numerik, ≤ 6 baris
    - 'bar'    : 2 kolom, kolom ke-2 numerik, > 6 baris
    - None     : tampilkan tabel biasa
    """
    if df is None or df.empty:
        return None

    # Satu angka (COUNT(*) / total)
    if len(df.columns) == 1 and len(df) == 1:
        if pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
            return "metric"

    # Dua kolom: label + angka
    if len(df.columns) == 2:
        col_val = df.columns[1]
        if pd.api.types.is_numeric_dtype(df[col_val]):
            return "pie" if len(df) <= 6 else "bar"

    return None

# =====================================================
# STREAMLIT UI
# =====================================================

st.set_page_config(
    page_title="Chatbot LKS Bipartit PLN",
    page_icon="💼",
    layout="wide",
)

# ── Design System: PLN Institutional v2 ─────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@300;400;500&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --ink:     #0A0204;
  --burg:    #7A1428;
  --burg-2:  #5C0F1E;
  --gold:    #A67C2E;
  --gold-lt: #C9A84C;
  --sand:    #F7F2EA;
  --sand-2:  #EDE5D8;
  --muted:   #6B5B4E;
  --rule:    #D8CDBF;
  --serif: 'Playfair Display', Georgia, serif;
  --sans:  'Inter', Helvetica Neue, Arial, sans-serif;
  --mono:  'JetBrains Mono', Courier New, monospace;
}

html, body, .stApp { background: var(--sand) !important; }
.block-container {
  max-width: 100% !important;
  padding: 1.5rem 3rem 5rem !important;
  margin: 0 !important;
}

h1 {
  font-family: var(--serif) !important;
  font-size: 3rem !important;
  font-weight: 700 !important;
  color: var(--ink) !important;
  line-height: 1.08 !important;
  letter-spacing: -0.03em !important;
  margin: 0.6rem 0 0.5rem !important;
  padding: 0 !important;
  border: none !important;
}
h2, h3 {
  font-family: var(--sans) !important;
  font-size: 0.65rem !important;
  font-weight: 500 !important;
  color: var(--muted) !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  margin: 1.5rem 0 0.5rem !important;
  border: none !important;
}
p { font-family: var(--sans); font-size: 0.95rem; line-height: 1.6; color: var(--ink); }

[data-testid="stMetric"] {
  background: transparent !important;
  border: none !important;
  border-top: 1.5px solid var(--gold-lt) !important;
  padding: 10px 0 0 !important;
  border-radius: 0 !important;
  box-shadow: none !important;
}
[data-testid="stMetricLabel"] {
  font-family: var(--mono) !important;
  font-size: 0.6rem !important;
  color: var(--muted) !important;
  text-transform: uppercase !important;
  letter-spacing: 0.1em !important;
  font-weight: 500 !important;
}
[data-testid="stMetricValue"] {
  font-family: var(--serif) !important;
  font-size: 2.4rem !important;
  font-weight: 700 !important;
  color: var(--burg) !important;
  letter-spacing: -0.02em !important;
  line-height: 1.1 !important;
}

hr { border: none !important; border-top: 1px solid var(--rule) !important; margin: 1.25rem 0 !important; }

[data-testid="stTextArea"] textarea {
  font-family: var(--sans) !important;
  font-size: 0.95rem !important;
  background: #fff !important;
  border: 1px solid var(--rule) !important;
  border-radius: 4px !important;
  padding: 12px 14px !important;
  color: var(--ink) !important;
  box-shadow: 0 1px 3px rgba(10,2,4,.05) !important;
  transition: border-color .15s !important;
}
[data-testid="stTextArea"] textarea:focus {
  border-color: var(--burg) !important;
  box-shadow: 0 0 0 4px rgba(122,20,40,.12) !important;
}
[data-testid="stTextArea"] label {
  font-family: var(--serif) !important;
  font-size: 1.05rem !important;
  font-weight: 700 !important;
  color: var(--ink) !important;
  letter-spacing: 0 !important;
  text-transform: none !important;
  margin-bottom: 6px !important;
}

[data-testid="stButton"] button[kind="primary"] {
  font-family: var(--sans) !important;
  font-size: 0.82rem !important;
  font-weight: 500 !important;
  letter-spacing: 0.04em !important;
  background: var(--burg) !important;
  color: #F5EDD8 !important;
  border: none !important;
  border-radius: 4px !important;
  padding: 0.6rem 2rem !important;
  box-shadow: 0 1px 4px rgba(10,2,4,.2) !important;
  transition: background .15s, transform .1s !important;
}
[data-testid="stButton"] button[kind="primary"]:hover {
  background: var(--burg-2) !important;
  transform: translateY(-1px) !important;
}
[data-testid="stButton"] button[kind="secondary"] {
  font-family: var(--sans) !important;
  font-size: 0.82rem !important;
  background: transparent !important;
  border: 1px solid var(--rule) !important;
  color: var(--muted) !important;
  border-radius: 4px !important;
}
[data-testid="stButton"] button[kind="secondary"]:hover {
  border-color: var(--burg) !important;
  color: var(--burg) !important;
}

[data-testid="stExpander"] {
  border: 1px solid var(--rule) !important;
  border-radius: 4px !important;
  background: #fff !important;
  box-shadow: none !important;
}
[data-testid="stExpander"] summary {
  font-family: var(--sans) !important;
  font-size: 0.82rem !important;
  font-weight: 500 !important;
  color: var(--ink) !important;
}
[data-testid="stExpander"] summary:hover { color: var(--burg) !important; }

[data-testid="stCode"] {
  background: var(--ink) !important;
  border-radius: 4px !important;
  border-left: 2px solid var(--gold-lt) !important;
}
[data-testid="stCode"] code {
  font-family: var(--mono) !important;
  font-size: 0.78rem !important;
  color: #E8D5B0 !important;
}

[data-testid="stDataFrame"] {
  border: 1px solid var(--rule) !important;
  border-radius: 4px !important;
  overflow: hidden !important;
}
[data-testid="stDataFrame"] th {
  font-family: var(--mono) !important;
  font-size: 0.62rem !important;
  text-transform: uppercase !important;
  letter-spacing: 0.08em !important;
  color: var(--muted) !important;
  background: var(--sand-2) !important;
  font-weight: 500 !important;
}
[data-testid="stDataFrame"] td {
  font-family: var(--sans) !important;
  font-size: 0.875rem !important;
}

[data-testid="stCaptionContainer"] {
  font-family: var(--mono) !important;
  font-size: 0.62rem !important;
  color: var(--muted) !important;
  letter-spacing: 0.04em !important;
}

[data-testid="stSidebar"] {
  background: var(--burg-2) !important;
  border-right: 1px solid rgba(255,255,255,.06) !important;
}
[data-testid="stSidebar"] * { color: #E8D5B0 !important; }
[data-testid="stSidebar"] .stButton button {
  background: rgba(198,168,76,.1) !important;
  border: 1px solid rgba(198,168,76,.3) !important;
  color: var(--gold-lt) !important;
  font-size: 0.78rem !important;
  border-radius: 3px !important;
}
[data-testid="stSidebar"] .stButton button:hover {
  background: rgba(198,168,76,.22) !important;
}
[data-testid="collapsedControl"] { background: var(--burg-2) !important; }

[data-testid="stTab"] {
  font-family: var(--sans) !important;
  font-size: 0.82rem !important;
  color: var(--muted) !important;
}
[data-testid="stTab"][aria-selected="true"] {
  color: var(--burg) !important;
  border-bottom: 2px solid var(--burg) !important;
  font-weight: 500 !important;
}

[data-testid="stAlert"] { border-radius: 4px !important; font-family: var(--sans) !important; font-size: 0.88rem !important; }

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--rule); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--burg); }
</style>
""", unsafe_allow_html=True)

# ── Preload RAG index setelah semua fungsi terdefinisi ──
try:
    _rag_files_init = _list_dokumen_supabase()
    if _rag_files_init:
        _rag_index = bangun_rag_index()
        print(f"✅ RAG index: {len(_rag_index)} dokumen terindeks")
    else:
        print("⚠️  RAG: tidak ada dokumen di bucket atau list gagal")
except Exception as _e_rag_init:
    print(f"⚠️  RAG preload gagal: {_e_rag_init}")

# ── Sidebar ──────────────────────────────────────────
with st.sidebar:
    # ── Logo / Brand ─────────────────────────────────────────────────
    st.markdown(
        "<div style='text-align:center;padding:12px 0 4px 0;'>"
        "<span style='font-size:2.2rem;'>💼</span><br>"
        "<span style='font-size:1rem;font-weight:700;color:#C9A84C;"
        "letter-spacing:1px;font-family:Georgia,serif;'>LKS Bipartit</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<hr style='border-color:#C9A84C44;margin:8px 0;'/>",
        unsafe_allow_html=True,
    )

    # ── Status ringkas ────────────────────────────────────────────────
    _dot_db  = "🟢" if DATA_SIAP  else "🔴"
    _dot_llm = "🟢" if not USE_MOCK else "🟡"
    _dot_rag = "🟢" if _rag_index else "🟡"
    st.markdown(
        f"<div style='font-size:0.8rem;line-height:1.9;'>"
        f"{_dot_db} <b>Data</b>: {SUMBER if DATA_SIAP else 'Error'}<br>"
        f"{_dot_llm} <b>LLM</b>: {PROVIDER.upper()}<br>"
        f"{_dot_rag} <b>RAG</b>: {len(_rag_index)} dok terindeks"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Ringkasan data (compact) ──────────────────────────────────────
    if DATA_SIAP:
        st.markdown(
            "<hr style='border-color:#C9A84C44;margin:10px 0;'/>",
            unsafe_allow_html=True,
        )
        try:
            with engine.connect() as _c:
                _tot = _c.execute(text("SELECT COUNT(*) FROM lks_bipartit")).scalar()
                _bel = _c.execute(text(
                    "SELECT COUNT(*) FROM lks_bipartit WHERE status_tindak_lanjut='Belum'"
                )).scalar()
                _sud = _c.execute(text(
                    "SELECT COUNT(*) FROM lks_bipartit WHERE status_tindak_lanjut='Sudah'"
                )).scalar()
            st.markdown(
                f"<div style='font-size:0.82rem;text-align:center;'>"
                f"<div style='font-size:1.4rem;font-weight:700;color:#C9A84C;'>{_tot:,}</div>"
                f"<div style='color:#FAF5EE88;font-size:0.72rem;'>Total Pertemuan</div>"
                f"<div style='margin-top:8px;display:flex;gap:12px;justify-content:center;'>"
                f"<span>✅ {_sud:,}</span><span>⏳ {_bel:,}</span>"
                f"</div></div>",
                unsafe_allow_html=True,
            )
        except Exception:
            pass

    # ── Dokumen RAG ───────────────────────────────────────────────────
    st.markdown(
        "<hr style='border-color:#C9A84C44;margin:10px 0;'/>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-size:0.78rem;color:#C9A84C;font-weight:600;"
        "letter-spacing:0.5px;text-transform:uppercase;margin-bottom:6px;'>"
        "📁 Dokumen</div>",
        unsafe_allow_html=True,
    )

    # ── Upload Data Excel ke Supabase ────────────────────
    st.markdown(
        "<div style='font-family:JetBrains Mono,monospace;font-size:0.6rem;"
        "text-transform:uppercase;letter-spacing:0.1em;color:#C9A84C;"
        "margin:0 0 6px 0;'>📊 Update Data LKS Bipartit</div>",
        unsafe_allow_html=True,
    )
    _excel_up = st.file_uploader(
        "Upload Excel (.xlsx)",
        type=["xlsx", "xls"],
        key="excel_uploader",
        label_visibility="collapsed",
    )
    if _excel_up:
        with st.spinner("Memproses Excel..."):
            try:
                import io as _io
                _df_new = pd.read_excel(_io.BytesIO(_excel_up.getvalue()))

                # Konversi tanggal ke YYYY-MM-DD (aman, tidak ambigu)
                # Pakai nama kolom ASLI persis seperti di tabel Supabase
                for _tc in ["Tanggal Pelaksanaan", "Tanggal Deadline", "Tanggal Tindak Lanjut"]:
                    if _tc in _df_new.columns:
                        _df_new[_tc] = pd.to_datetime(
                            _df_new[_tc], errors="coerce"
                        ).dt.strftime("%Y-%m-%d")

                # Hanya kirim kolom yang ada di tabel Supabase (dari Excel)
                # Jangan tambah kolom baru yang tidak ada di skema tabel
                _kolom_supabase = [
                    "Kode Pembahasan", "Personal Area Nama", "Personal Sub Area Nama",
                    "Tanggal Pelaksanaan", "Tema LKS Bipartit", "Latar Belakang",
                    "Rekomendasi", "Tanggal Deadline", "Tindak Lanjut", "Tanggal Tindak Lanjut"
                ]
                _kolom_ada = [c for c in _kolom_supabase if c in _df_new.columns]
                _df_new = _df_new[_kolom_ada]

                st.success(f"✅ Excel terbaca: {len(_df_new):,} baris")
                st.caption(f"Rentang: {_df_new['Tanggal Pelaksanaan'].min()} s/d {_df_new['Tanggal Pelaksanaan'].max()}")

                if st.button("⬆️ Upload ke Supabase (ganti data lama)", key="btn_upload_excel"):
                    with st.spinner("Menghapus data lama..."):
                        # TRUNCATE via DELETE dengan filter semua baris
                        # Supabase REST: DELETE butuh filter — pakai kolom primary key != ''
                        _del = _req.delete(
                            f"{SUPABASE_URL}/rest/v1/{NAMA_TABEL}?"
                            f"Kode%20Pembahasan=neq.TIDAK_ADA_YANG_INI_XYZ",
                            headers={**_sb_headers(), "Prefer": "return=minimal"},
                            timeout=30,
                        )
                        # Jika DELETE gagal (butuh RPC), coba via SQL RPC
                        if _del.status_code not in (200, 204):
                            _req.post(
                                f"{SUPABASE_URL}/rest/v1/rpc/truncate_hi",
                                headers=_sb_headers(), json={}, timeout=10,
                            )
                        st.caption(f"Delete status: {_del.status_code}")

                    with st.spinner(f"Mengupload {len(_df_new):,} baris..."):
                        _headers_up = {**_sb_headers(), "Prefer": "resolution=merge-duplicates,return=minimal"}
                        _url_up     = f"{SUPABASE_URL}/rest/v1/{NAMA_TABEL}"
                        _batch_size = 500
                        _berhasil   = 0
                        for _i in range(0, len(_df_new), _batch_size):
                            _chunk = (
                                _df_new.iloc[_i:_i+_batch_size]
                                       .where(_df_new.iloc[_i:_i+_batch_size].notna(), None)
                                       .to_dict(orient="records")
                            )
                            _r = _req.post(_url_up, json=_chunk,
                                          headers=_headers_up, timeout=30)
                            if _r.status_code in (200, 201):
                                _berhasil += len(_chunk)
                            else:
                                st.warning(f"Batch {_i//500+1} error: {_r.text[:100]}")

                        if _berhasil > 0:
                            st.success(f"✅ {_berhasil:,} baris berhasil diupload ke Supabase!")
                            st.cache_resource.clear()
                            st.rerun()

            except Exception as _eu:
                st.error(f"❌ Gagal proses Excel: {_eu}")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-family:JetBrains Mono,monospace;font-size:0.6rem;"
        "text-transform:uppercase;letter-spacing:0.1em;color:#C9A84C;"
        "margin:0 0 6px 0;'>📁 Dokumen RAG</div>",
        unsafe_allow_html=True,
    )

    # ── Upload Dokumen RAG ────────────────────────────────
    _uploaded = st.file_uploader(
        "Upload PDF",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        key="rag_uploader",
        label_visibility="collapsed",
    )
    if _uploaded:
        _headers_up = {**_sb_headers(), "Content-Type": "application/octet-stream",
                       "x-upsert": "true"}
        for _uf in _uploaded:
            _url_up = (f"{SUPABASE_URL}/storage/v1/object/"
                       f"{RAG_BUCKET.replace(' ', '%20')}/{_uf.name.replace(' ', '%20')}")
            _r = _req.post(_url_up, data=_uf.getbuffer(),
                           headers=_headers_up, timeout=30)
            if _r.status_code in (200, 201):
                st.success(f"✅ {_uf.name}")
            else:
                st.error(f"❌ {_uf.name[:20]}")
        _rag_index.clear()
        st.rerun()

    _sb_docs = _list_dokumen_supabase()
    for _d in _sb_docs:
        _nama = _d["name"]
        _pub_url = _d.get("_url") or _get_public_url(_d)
        _short = _nama[:22] + "…" if len(_nama) > 22 else _nama
        _ca, _cb, _cc = st.columns([5, 2, 1])
        _ca.markdown(f"<span style='font-size:0.78rem;'>📄 {_short}</span>",
                     unsafe_allow_html=True)
        _cb.markdown(
            f"<a href='{_pub_url}' target='_blank' "
            f"style='font-size:0.75rem;color:#C9A84C;'>👁</a>",
            unsafe_allow_html=True,
        )
        if _cc.button("✕", key=f"del_{_nama}", help="Hapus"):
            _req.delete(
                f"{SUPABASE_URL}/storage/v1/object/"
                f"{RAG_BUCKET.replace(' ','%20')}/{_nama.replace(' ','%20')}",
                headers=_sb_headers(), timeout=10,
            )
            _rag_index.clear()
            st.rerun()

    # ── History ringkas ───────────────────────────────────────────────
    if st.session_state.get("history"):
        st.markdown(
            "<hr style='border-color:#C9A84C44;margin:10px 0;'/>"
            "<div style='font-size:0.78rem;color:#C9A84C;font-weight:600;"
            "letter-spacing:0.5px;text-transform:uppercase;margin-bottom:4px;'>"
            "🕘 Riwayat</div>",
            unsafe_allow_html=True,
        )
        for _hi in reversed(st.session_state.history[-5:]):
            _q = _hi["question"][:30] + "…" if len(_hi["question"]) > 30 else _hi["question"]
            st.markdown(
                f"<div style='font-size:0.75rem;padding:3px 0;"
                f"border-bottom:1px solid #C9A84C22;'>{_q}</div>",
                unsafe_allow_html=True,
            )

# ── Main ─────────────────────────────────────────────
# Pastikan cache bersih saat restart
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.cache_resource.clear()

# ── Header 1001-style ─────────────────────────────────────────────────
try:
    with engine.connect() as _hc:
        _tot = _hc.execute(text("SELECT COUNT(*) FROM lks_bipartit")).scalar()
        _bel = _hc.execute(text(
            "SELECT COUNT(*) FROM lks_bipartit WHERE status_tindak_lanjut='Belum'"
        )).scalar()
        _sud = _hc.execute(text(
            "SELECT COUNT(*) FROM lks_bipartit WHERE status_tindak_lanjut='Sudah'"
        )).scalar()
        _area = _hc.execute(text(
            "SELECT COUNT(DISTINCT kode_area) FROM lks_bipartit"
        )).scalar()
        _tema = _hc.execute(text(
            "SELECT COUNT(DISTINCT tema) FROM lks_bipartit"
        )).scalar()
except Exception:
    _tot = _bel = _sud = _area = _tema = "—"

st.markdown(
    "<div style='margin-bottom:0.15rem;'>"
    "<span style='font-family:DM Mono,monospace;font-size:0.7rem;"
    "letter-spacing:0.1em;color:#7A6A5A;text-transform:uppercase;'>"
    "PLN · Hubungan Industrial · Feb–Apr 2026</span>"
    "</div>",
    unsafe_allow_html=True,
)
st.title("LKS Bipartit Analytics")
st.markdown(
    "<p style='font-size:1.05rem;color:#7A6A5A;margin:0 0 1.5rem 0;"
    "max-width:560px;line-height:1.55;'>"
    "Tanya data pertemuan permasalahan pegawai — narasi, chart, atau dokumen. "
    "Satu pertanyaan, satu jawaban langsung dari data.</p>",
    unsafe_allow_html=True,
)

# ── Stat strip ────────────────────────────────────────────────────────
_c1, _c2, _c3, _c4 = st.columns(4)
_c1.metric("Pertemuan", f"{_tot:,}" if isinstance(_tot, int) else str(_tot))
_c2.metric("Sudah TL",  f"{_sud:,}" if isinstance(_sud, int) else str(_sud))
_c3.metric("Belum TL",  f"{_bel:,}" if isinstance(_bel, int) else str(_bel))
_c4.metric("Area",      f"{_area:,}" if isinstance(_area, int) else str(_area))

# ── Alert Deadline Realtime ───────────────────────────────────────────
if DATA_SIAP:
    try:
        from datetime import datetime as _dt, timedelta as _td
        _today = _dt.now().date()
        _batas = _today + _td(days=15)

        with engine.connect() as _ac:
            _df_dl = pd.read_sql(text("""
                SELECT kode_area, tema, tanggal_deadline, latar_belakang
                FROM lks_bipartit
                WHERE status_tindak_lanjut = 'Belum'
                AND tanggal_deadline IS NOT NULL
                AND tanggal_deadline != ''
            """), _ac)

        if len(_df_dl) > 0:
            _df_dl["_tgl"] = pd.to_datetime(_df_dl["tanggal_deadline"], errors="coerce").dt.date
            _df_dl = _df_dl.dropna(subset=["_tgl"])
            _df_dl["_sisa"] = _df_dl["_tgl"].apply(lambda d: (d - _today).days)
            _lewat  = _df_dl[_df_dl["_sisa"] < 0]
            _segera = _df_dl[(_df_dl["_sisa"] >= 0) & (_df_dl["_sisa"] <= 15)]

            if len(_lewat) > 0 or len(_segera) > 0:
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                _al1, _al2 = st.columns(2)

                with _al1:
                    _n = len(_lewat)
                    st.markdown(
                        f"<div style='border:1.5px solid #C0392B;border-radius:6px;"
                        f"background:#FDF0EE;padding:14px 18px;'>"
                        f"<span style='font-family:JetBrains Mono,monospace;font-size:0.6rem;"
                        f"text-transform:uppercase;letter-spacing:0.1em;color:#C0392B;'>"
                        f"⚠️ Deadline Terlewat</span><br>"
                        f"<span style='font-family:Playfair Display,Georgia,serif;"
                        f"font-size:2rem;font-weight:700;color:#C0392B;line-height:1.1;'>"
                        f"{_n}</span>"
                        f"<span style='font-size:0.85rem;color:#6B5B4E;margin-left:6px;'>"
                        f"bahasan belum selesai</span></div>",
                        unsafe_allow_html=True,
                    )
                    if _n > 0:
                        with st.expander(f"Lihat {_n} bahasan yang terlewat →"):
                            for _, _r in _lewat.sort_values("_sisa").iterrows():
                                _hari = abs(int(_r["_sisa"]))
                                st.markdown(
                                    f"<div style='padding:8px 0;border-bottom:1px solid #F0E0DC;'>"
                                    f"<span style='font-family:JetBrains Mono,monospace;"
                                    f"font-size:0.65rem;color:#C0392B;'>"
                                    f"Area {_r['kode_area']} · terlambat {_hari} hari</span><br>"
                                    f"<b style='font-size:0.88rem;'>{_r['tema']}</b><br>"
                                    f"<span style='font-size:0.8rem;color:#6B5B4E;'>"
                                    f"Deadline: {_r['tanggal_deadline']}</span></div>",
                                    unsafe_allow_html=True,
                                )

                with _al2:
                    _ns = len(_segera)
                    st.markdown(
                        f"<div style='border:1.5px solid #A67C2E;border-radius:6px;"
                        f"background:#FBF7EE;padding:14px 18px;'>"
                        f"<span style='font-family:JetBrains Mono,monospace;font-size:0.6rem;"
                        f"text-transform:uppercase;letter-spacing:0.1em;color:#A67C2E;'>"
                        f"⏳ Segera Ditindaklanjuti</span><br>"
                        f"<span style='font-family:Playfair Display,Georgia,serif;"
                        f"font-size:2rem;font-weight:700;color:#A67C2E;line-height:1.1;'>"
                        f"{_ns}</span>"
                        f"<span style='font-size:0.85rem;color:#6B5B4E;margin-left:6px;'>"
                        f"bahasan · deadline ≤ 15 hari</span></div>",
                        unsafe_allow_html=True,
                    )
                    _label_segera = f"Lihat {_ns} bahasan mendesak →" if _ns > 0 else "Tidak ada bahasan mendesak saat ini"
                    with st.expander(_label_segera):
                        if _ns == 0:
                            st.markdown(
                                "<div style='padding:12px 0;text-align:center;color:#A67C2E;"
                                "font-family:JetBrains Mono,monospace;font-size:0.78rem;'>"
                                "✅ Tidak ada pembahasan yang perlu segera ditindaklanjuti"
                                "</div>",
                                unsafe_allow_html=True,
                            )
                        else:
                            for _, _r in _segera.sort_values("_sisa").iterrows():
                                _si = int(_r["_sisa"])
                                _c  = "#C0392B" if _si <= 3 else "#A67C2E"
                                st.markdown(
                                    f"<div style='padding:8px 0;border-bottom:1px solid #F0E8D8;'>"
                                    f"<span style='font-family:JetBrains Mono,monospace;"
                                    f"font-size:0.65rem;color:{_c};'>"
                                    f"Area {_r['kode_area']} · sisa {_si} hari</span><br>"
                                    f"<b style='font-size:0.88rem;'>{_r['tema']}</b><br>"
                                    f"<span style='font-size:0.8rem;color:#6B5B4E;'>"
                                    f"Deadline: {_r['tanggal_deadline']}</span></div>",
                                    unsafe_allow_html=True,
                                )
    except Exception:
        pass

# ── Top 3 Tema — Trend 3 Bulan Terakhir ──────────────────────────────
if DATA_SIAP:
    try:
        with engine.connect() as _nc:
            _df_trend = pd.read_sql(
                text("""
                    SELECT tema, tanggal_pelaksanaan
                    FROM lks_bipartit
                    WHERE tema IS NOT NULL
                    AND tanggal_pelaksanaan IS NOT NULL
                """),
                _nc,
            )

        # Parse tanggal & ambil 3 bulan terakhir
        # Parse tanggal — support format DD/MM/YYYY, YYYY-MM-DD, MM/DD/YYYY
        def _parse_tgl(s):
            for _fmt in ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"]:
                try:
                    return pd.to_datetime(s, format=_fmt, errors="raise")
                except Exception:
                    continue
            return pd.to_datetime(s, errors="coerce", dayfirst=True)

        _df_trend["tgl"] = _df_trend["tanggal_pelaksanaan"].apply(_parse_tgl)
        _df_trend = _df_trend.dropna(subset=["tgl"])
        _df_trend["bulan"] = _df_trend["tgl"].dt.to_period("M")

        # (debug dihapus)

        from datetime import datetime as _dtnow
        _bulan_skrg = pd.Period(_dtnow.now(), freq="M")

        # Hanya ambil bulan yang <= bulan sekarang (tidak boleh future)
        _bulan_valid = sorted([b for b in _df_trend["bulan"].unique() if b <= _bulan_skrg])
        print(f"  Bulan valid (tidak future): {_bulan_valid}")

        if not _bulan_valid:
            raise Exception("Tidak ada data bulan valid di tanggal_pelaksanaan")

        _bulan_list = _bulan_valid[-3:]  # 3 bulan terakhir yang tidak future
        print(f"  Bulan ditampilkan: {_bulan_list}")
        _df_3bln    = _df_trend[_df_trend["bulan"].isin(_bulan_list)].copy()

        # Top 3 tema berdasarkan total 3 bulan
        _top3_tema = (
            _df_3bln["tema"].value_counts()
            .head(3).index.tolist()
        )

        # Hitung jumlah per tema per bulan
        _df_pivot = (
            _df_3bln[_df_3bln["tema"].isin(_top3_tema)]
            .groupby(["bulan", "tema"])
            .size()
            .reset_index(name="jumlah")
        )
        _df_pivot["bulan_str"] = _df_pivot["bulan"].astype(str).apply(
            lambda x: ["Jan","Feb","Mar","Apr","Mei","Jun",
                        "Jul","Agu","Sep","Okt","Nov","Des"][int(x.split("-")[1])-1]
                      + " " + x.split("-")[0]
        )

        # Label section
        st.markdown(
            "<p style='font-family:Playfair Display,Georgia,serif;font-size:1.4rem;"
            "font-weight:700;color:#0A0204;letter-spacing:-0.01em;"
            "margin:1.5rem 0 2px 0;'>Tren Tema — 3 Bulan Terakhir</p>",
            unsafe_allow_html=True,
        )
        _bulan_labels = [
            ["Jan","Feb","Mar","Apr","Mei","Jun","Jul","Agu","Sep","Okt","Nov","Des"][int(str(b).split("-")[1])-1]
            + " " + str(b).split("-")[0]
            for b in _bulan_list
        ]
        st.markdown(
            f"<p style='font-size:0.95rem;color:#6B5B4E;margin:0 0 1rem 0;'>"
            f"Top 3 tema terbanyak &nbsp;·&nbsp; {' &nbsp;·&nbsp; '.join(_bulan_labels)}</p>",
            unsafe_allow_html=True,
        )

        # Warna 3 tema
        _warna3 = ["#7A1428", "#A67C2E", "#5C6B2E"]
        _tema_colors = {t: _warna3[i] for i, t in enumerate(_top3_tema)}

        # 3 kolom — satu per bulan
        _cols3 = st.columns(3)
        for _bi, _bln in enumerate(_bulan_list):
            _bln_str = _bulan_labels[_bi]
            _df_bln  = _df_pivot[_df_pivot["bulan"] == _bln].copy()
            _df_bln  = _df_bln.set_index("tema").reindex(_top3_tema).fillna(0).reset_index()

            with _cols3[_bi]:
                # Mini metric total bulan
                _total_bln = int(_df_bln["jumlah"].sum())
                st.markdown(
                    f"<div style='border-top:1.5px solid #C9A84C;padding-top:8px;"
                    f"margin-bottom:10px;'>"
                    f"<span style='font-family:JetBrains Mono,monospace;"
                    f"font-size:0.6rem;text-transform:uppercase;color:#6B5B4E;"
                    f"letter-spacing:0.1em;'>{_bln_str}</span><br>"
                    f"<span style='font-family:Playfair Display,Georgia,serif;"
                    f"font-size:1.8rem;font-weight:700;color:#7A1428;"
                    f"line-height:1.1;'>{_total_bln}</span><br>"
                    f"<span style='font-family:JetBrains Mono,monospace;"
                    f"font-size:0.58rem;color:#6B5B4E;'>pertemuan (top 3)</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # Bar chart mini per bulan
                _fig_mini = px.bar(
                    _df_bln,
                    x="tema", y="jumlah",
                    text="jumlah",
                    color="tema",
                    color_discrete_map=_tema_colors,
                )
                _fig_mini.update_traces(
                    textposition="outside",
                    hovertemplate="<b>%{x}</b><br>%{y} pertemuan<extra></extra>",
                    width=0.55,
                )
                _fig_mini.update_layout(
                    showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0, r=10, t=35, b=60),
                    height=250,
                    xaxis=dict(
                        tickfont=dict(size=9, family="Inter"),
                        tickangle=-20,
                        title="",
                        showgrid=False,
                        ticktext=[t[:14]+"…" if len(t)>14 else t for t in _top3_tema],
                        tickvals=_top3_tema,
                    ),
                    yaxis=dict(
                        showgrid=True, gridcolor="#E8DDD0",
                        gridwidth=0.5, tickfont=dict(size=9),
                        title="",
                        rangemode="tozero",
                        range=[0, _df_bln["jumlah"].max() * 1.25],
                    ),
                )
                st.plotly_chart(_fig_mini, use_container_width=True,
                                key=f"trend_chart_{_bi}")

        # Legenda tema di bawah
        _leg_html = "<div style='display:flex;gap:20px;margin-top:4px;flex-wrap:wrap;'>"
        for _i, _t in enumerate(_top3_tema):
            _leg_html += (
                f"<span style='font-family:JetBrains Mono,monospace;"
                f"font-size:0.65rem;color:#6B5B4E;'>"
                f"<span style='display:inline-block;width:10px;height:10px;"
                f"border-radius:2px;background:{_warna3[_i]};"
                f"margin-right:5px;vertical-align:middle;'></span>"
                f"{_t}</span>"
            )
        _leg_html += "</div>"
        st.markdown(_leg_html, unsafe_allow_html=True)



    except Exception as _ne:
        st.caption(f"Gagal load trend: {_ne}")
st.markdown("<hr/>", unsafe_allow_html=True)

if "query" not in st.session_state:
    st.session_state.query = ""

# ── Input pertanyaan (besar & menonjol) ──────────────
st.markdown(
    "<p style='font-family:Playfair Display,Georgia,serif;font-size:1.4rem;"
    "font-weight:700;color:#0A0204;letter-spacing:-0.01em;"
    "margin:0.5rem 0 6px 0;'>Tanya apa saja tentang data LKS Bipartit</p>",
    unsafe_allow_html=True,
)
query = st.text_area(
    "",
    value=st.session_state.query,
    height=130,
    label_visibility="collapsed",
    placeholder=(
        "Contoh:\n"
        "• tema apa yang paling banyak dibahas bulan Februari 2026?\n"
        "• permasalahan terkait SPPD yang belum ditindaklanjuti\n"
        "• berapa pertemuan per area bulan Maret?"
    ),
)

col1, col2 = st.columns([1, 4])
with col1:
    jalankan_btn = st.button("🚀 Jalankan", type="primary", use_container_width=True)
with col2:
    if st.button("🗑️ Clear", use_container_width=False):
        st.session_state.query = ""
        st.rerun()

# ── FAQ — definisi per kategori ──────────────────────
FAQ_DATA = [
    {
        "kategori": "📊 Analisis Tema",
        "items": [
            ("Tema apa yang paling banyak dibahas?",        "Tema apa yang paling banyak dibahas?"),
            ("Tema terbanyak bulan Februari 2026?",         "Tema yang paling banyak dibahas bulan Februari 2026?"),
            ("Tema terbanyak bulan Maret 2026?",            "Tema yang paling banyak dibahas bulan Maret 2026?"),
            ("Tema terbanyak bulan April 2026?",            "Tema yang paling banyak dibahas bulan April 2026?"),
        ],
    },
    {
        "kategori": "⏳ Tindak Lanjut",
        "items": [
            ("Permasalahan yang belum ditindaklanjuti?",    "Permasalahan yang belum ditindaklanjuti?"),
            ("Belum ditindaklanjuti bulan Februari?",       "Permasalahan belum ditindaklanjuti bulan Februari 2026?"),
            ("Deadline sudah lewat & belum selesai?",       "Deadline yang sudah lewat dan belum selesai?"),
        ],
    },
    {
        "kategori": "🔍 Cari Topik",
        "items": [
            ("Permasalahan terkait SPPD?",                  "Permasalahan terkait SPPD?"),
            ("Rekomendasi terkait PKB?",                    "Rekomendasi terkait PKB?"),
            ("Permasalahan terkait kesehatan pegawai?",     "Permasalahan terkait kesehatan pegawai?"),
        ],
    },
    {
        "kategori": "📍 Statistik",
        "items": [
            ("Berapa pertemuan per area?",                  "Berapa pertemuan per area?"),
            ("Berapa pertemuan per tema?",                  "Berapa pertemuan per tema?"),
            ("Total pertemuan keseluruhan?",                "Berapa total pertemuan?"),
        ],
    },
]

FAQ_FLAT = [(lbl, prm) for kat in FAQ_DATA for lbl, prm in kat["items"]]

# ── FAQ — tampilan ─────────────────────────────────────
_q_now = (st.session_state.get("query") or "").strip().lower()
_words  = [w for w in _q_now.split() if len(w) > 2]

if not _q_now:
    # Mode default: accordion 4 kolom per kategori
    st.markdown(
        "<p style='font-family:DM Mono,monospace;font-size:0.68rem;"
        "letter-spacing:0.08em;text-transform:uppercase;"
        "color:#7A6A5A;margin:0 0 10px 0;'>Pertanyaan yang sering ditanyakan</p>",
        unsafe_allow_html=True,
    )
    _kcat = st.columns(len(FAQ_DATA))
    for _ki, kat in enumerate(FAQ_DATA):
        with _kcat[_ki]:
            with st.expander(kat["kategori"], expanded=False):
                for lbl, prm in kat["items"]:
                    if st.button(lbl, key=f"faq_{hash(prm)}", use_container_width=True):
                        st.session_state.query    = prm
                        st.session_state.auto_run = True
                        st.rerun()
else:
    # Mode filter: tampilkan suggestion yang relevan
    _filtered = [
        (lbl, prm) for lbl, prm in FAQ_FLAT
        if any(w in lbl.lower() or w in prm.lower() for w in _words)
    ] or FAQ_FLAT[:6]

    st.markdown(
        "<p style='font-family:DM Mono,monospace;font-size:0.68rem;"
        "letter-spacing:0.08em;text-transform:uppercase;"
        "color:#7A6A5A;margin:0 0 6px 0;'>Mungkin ini yang dicari</p>",
        unsafe_allow_html=True,
    )
    _fc = st.columns(min(len(_filtered[:6]), 3))
    for _fi, (lbl, prm) in enumerate(_filtered[:6]):
        with _fc[_fi % 3]:
            if st.button(
                f"↗ {lbl}",
                key=f"faq_f_{hash(prm)}_{_fi}",
                use_container_width=True,
            ):
                st.session_state.query    = prm
                st.session_state.auto_run = True
                st.rerun()

st.markdown("---")

if "history" not in st.session_state:
    st.session_state.history = []

# Jalankan otomatis jika dari klik FAQ
_auto_run = st.session_state.pop("auto_run", False)

if (jalankan_btn or _auto_run) and query:
    if not DATA_SIAP:
        st.error(f"⚠️ Data tidak tersedia: {DATA_ERROR}")
    else:
        # ── Deteksi apakah pertanyaan tentang dokumen ───────
        # Init default agar tidak NoneType error
        df, sql, error = None, None, None

        _is_dok     = is_pertanyaan_dokumen(query)
        _dok_chunks = cari_dokumen(query) if _is_dok else []

        # Info RAG kecil (hanya jika relevan)
        if _is_dok and _dok_chunks:
            st.caption(
                f"📄 RAG aktif — {len(_rag_index)} dok terindeks, "
                f"{len(_dok_chunks)} bagian relevan ditemukan"
            )

        if _dok_chunks:
            # ── Mode RAG: jawab dari dokumen ──────────────────
            with st.spinner("📄 Mencari di dokumen..."):
                jawaban_rag = jawab_dengan_rag(query, _dok_chunks)

            st.markdown("<p style='font-family:DM Mono,monospace;font-size:0.68rem;letter-spacing:0.08em;text-transform:uppercase;color:#7A6A5A;margin:1.5rem 0 0.75rem 0;'>Jawaban dari Dokumen</p>", unsafe_allow_html=True)
            st.markdown(jawaban_rag)

            # ── Preview PDF langsung di app ──────────────────
            file_unik = list(dict.fromkeys(c["file"] for c in _dok_chunks))
            if file_unik:
                st.markdown("<p style='font-family:DM Mono,monospace;font-size:0.68rem;letter-spacing:0.08em;text-transform:uppercase;color:#7A6A5A;margin:1.5rem 0 0.75rem 0;'>Preview Dokumen</p>", unsafe_allow_html=True)
                _tab_labels = file_unik[:3]
                _tabs = st.tabs(_tab_labels)
                for _ti, _fname in enumerate(_tab_labels):
                    with _tabs[_ti]:
                        # Cari URL file
                        _furl = _rag_urls.get(_fname)
                        if not _furl:
                            # Cari dari _dok_chunks
                            for _c in _dok_chunks:
                                if _c["file"] == _fname:
                                    _furl = _get_public_url(_c)
                                    break
                        if _furl:
                            st.markdown(
                                f"<iframe src='{_furl}' width='100%' height='600px' "
                                f"style='border:1px solid #ddd;border-radius:8px;'>"
                                f"</iframe>",
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f"<a href='{_furl}' target='_blank' "
                                f"style='font-size:0.85rem;'>↗ Buka di tab baru</a>",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.info("URL dokumen tidak tersedia")

            with st.expander("📎 Lihat potongan teks yang digunakan", expanded=False):
                for c in _dok_chunks:
                    st.markdown(f"**{c['file']}** — {c['bagian']} (relevansi: {c['skor']})")
                    isi = c["chunk"].strip()
                    if len(isi) < 50:
                        st.warning(
                            f"⚠️ Teks sangat sedikit ({len(isi)} karakter). "
                            "PDF ini mungkin berupa scan/gambar — "
                            "coba upload versi PDF yang bisa di-copy teksnya."
                        )
                    else:
                        st.caption(isi[:500] + ("..." if len(isi) > 500 else ""))
                    st.divider()

            st.session_state.history.append({
                "question": query,
                "sql":      f"[RAG: {', '.join(c['file'] for c in _dok_chunks)}]",
                "rows":     0,
            })

        elif _is_dok and not _dok_chunks:
            # ── Dokumen terdeteksi tapi index kosong ──────────
            st.markdown(
                "<div style='background:#FFF8E1;border-left:4px solid #FFA000;"
                "border-radius:8px;padding:14px 18px;margin:8px 0;'>"
                "<b>📂 Dokumen ditemukan di Supabase tapi belum bisa dibaca.</b><br>"
                "Kemungkinan: PDF terenkripsi atau butuh pdfplumber.<br>"
                f"File di bucket: {', '.join(f['name'] for f in _list_dokumen_supabase()) or 'tidak ada'}"
                "</div>",
                unsafe_allow_html=True,
            )

        else:
            # ── Mode SQL: query ke database ───────────────────
            with st.spinner("🤖 Sedang menganalisis data..."):
                df, sql, error = jalankan_query(query)


            # ── ERROR + FALLBACK LLM ──────────────────────────
            if error:
                with st.spinner("🤔 Menganalisis pertanyaan Anda..."):
                    saran = tanya_fallback_llm(query)
                st.markdown(
                    "<div style='background:linear-gradient(135deg,#FFF8E1,#FFF3E0);"
                    "border-left:5px solid #FFA000;border-radius:8px;"
                    "padding:18px 22px;margin:12px 0;'>"
                    "<p style='font-size:1.1rem;font-weight:700;"
                    "color:#E65100;margin:0 0 10px 0;'>"
                    "🤔 Hmm, saya belum bisa menjawab pertanyaan itu...</p>"
                    f"<p style='font-size:0.93rem;color:#444;margin:0;"
                    f"white-space:pre-line;line-height:1.7;'>{saran}</p>"
                    "</div>",
                    unsafe_allow_html=True,
                )

            # ── SUKSES ───────────────────────────────────────
            elif df is not None:
                st.session_state.history.append({
                    "question": query,
                    "sql":      sql,
                    "rows":     len(df),
                })

        # ── Tampilkan hasil SQL jika ada ─────────────────────
        if df is not None and sql is not None and error is None:
            # ── 🤖 RINGKASAN ─────────────────────────────────
            tipe_chart = deteksi_chart(df)

            st.markdown("<p style='font-family:DM Mono,monospace;font-size:0.68rem;letter-spacing:0.08em;text-transform:uppercase;color:#7A6A5A;margin:1.5rem 0 0.75rem 0;'>Ringkasan</p>", unsafe_allow_html=True)
            if len(df) == 0:
                st.info("Query berhasil dijalankan namun tidak menemukan data.")
            elif tipe_chart == "metric":
                nilai = df.iloc[0, 0]
                nama  = df.columns[0].replace("_", " ").title()
                st.metric(nama, f"{nilai:,}")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("📋 Total Baris",  f"{len(df):,}")
                c2.metric("🗂️ Kolom",        f"{len(df.columns)}")
                # Kolom numerik → tampilkan sum/rata-rata
                num_cols = df.select_dtypes(include="number").columns.tolist()
                if num_cols:
                    col_n = num_cols[0]
                    c3.metric(f"∑ {col_n.replace('_',' ').title()}", f"{df[col_n].sum():,.0f}")
                elif "status_tindak_lanjut" in df.columns:
                    n_belum = (df["status_tindak_lanjut"] == "Belum").sum()
                    c3.metric("⏳ Belum TL", f"{n_belum:,}")
                elif "kode_area" in df.columns:
                    c3.metric("📍 Area", f"{df['kode_area'].nunique():,}")

            # ── 🗄️ SQL (tersembunyi default) ─────────────────
            with st.expander("🗄️ Lihat SQL Query", expanded=False):
                st.code(sql, language="sql")

            # ── 🕸️ Network Graph korelasi tema (pyvis) ───────────────
            if "tema" in df.columns and len(df) > 0 and PYVIS_OK:
                _tema_cnt = df["tema"].value_counts()
                if len(_tema_cnt) >= 2:
                    with st.expander("🕸️ Network Graph — Korelasi Antar Tema", expanded=False):
                        st.caption("Ukuran node = jumlah kemunculan · Tebal edge = area yang sama")

                        # Buat network
                        _g = pvnet.Network(
                            height="520px", width="100%",
                            bgcolor="#FAF2EA", font_color="#0A0204",
                            directed=False,
                        )
                        _g.set_options("""{
                          "nodes": {
                            "font": {"size": 13, "face": "Inter, sans-serif"},
                            "borderWidth": 1.5,
                            "shadow": {"enabled": true, "color": "rgba(0,0,0,0.1)", "size": 6}
                          },
                          "edges": {
                            "color": {"color": "#D8CDBF", "highlight": "#7A1428"},
                            "smooth": {"type": "continuous"},
                            "shadow": false
                          },
                          "physics": {
                            "forceAtlas2Based": {
                              "gravitationalConstant": -50,
                              "centralGravity": 0.01,
                              "springLength": 120
                            },
                            "solver": "forceAtlas2Based",
                            "stabilization": {"iterations": 150}
                          },
                          "interaction": {"hover": true, "tooltipDelay": 100}
                        }""")

                        # Warna node berdasarkan jumlah
                        _max_cnt = _tema_cnt.max()
                        for _tema, _cnt in _tema_cnt.items():
                            # Gradasi warna: sedikit=gold, banyak=burgundy
                            _ratio   = _cnt / _max_cnt
                            _r = int(122 + (10 - 122) * _ratio)
                            _g2 = int(20 + (2 - 20) * _ratio)
                            _b = int(40 + (4 - 40) * _ratio)
                            _color   = f"#{_r:02x}{_g2:02x}{_b:02x}"
                            _size    = max(18, min(50, 18 + int(_ratio * 32)))
                            _label   = str(_tema)[:22] + ("…" if len(str(_tema)) > 22 else "")
                            _g.add_node(
                                str(_tema), label=_label,
                                title=f"{_tema}: {_cnt} pertemuan",
                                color=_color, size=_size,
                                font={"color": "#FAF2EA" if _ratio > 0.5 else "#0A0204"},
                            )

                        # Edge: hubungkan tema yang muncul di area yang sama
                        if "kode_area" in df.columns:
                            _area_tema = df.groupby("kode_area")["tema"].apply(list)
                            _edge_w = {}
                            for _area, _tlist in _area_tema.items():
                                _uniq = list(set(_tlist))
                                for _i in range(len(_uniq)):
                                    for _j in range(_i+1, len(_uniq)):
                                        _k = (str(_uniq[_i]), str(_uniq[_j]))
                                        _k = (_k[0], _k[1]) if _k[0] < _k[1] else (_k[1], _k[0])
                                        _edge_w[_k] = _edge_w.get(_k, 0) + 1
                            # Ambil top 40 edge terkuat
                            for (_a, _b2), _w in sorted(
                                _edge_w.items(), key=lambda x: -x[1]
                            )[:40]:
                                if _a in [str(t) for t in _tema_cnt.index] and                                    _b2 in [str(t) for t in _tema_cnt.index]:
                                    _g.add_edge(
                                        _a, _b2,
                                        value=_w,
                                        title=f"Bersama di {_w} area",
                                        width=max(1, min(6, _w // 2)),
                                    )

                        pv_static(_g)

            # ── 🕸️ Mermaid: hubungan antar tema (selalu tampil jika ada tema) ──
            if "tema" in df.columns and len(df) > 0:
                _tema_counts = df["tema"].value_counts()
                if len(_tema_counts) >= 2:
                    with st.expander("🕸️ Lihat Hubungan Antar Tema", expanded=False):
                        # Bangun diagram Mermaid mindmap
                        _top_tema = _tema_counts.head(8)
                        _mermaid_lines = ["mindmap", "  root((LKS Bipartit))"]
                        _kategori = {
                            "Manajemen": ["manajemen", "penghargaan", "talenta", "rekrutmen", "promosi"],
                            "Hubungan Industrial": ["lks", "bipartit", "pkb", "perjanjian", "sp "],
                            "Fasilitas": ["fasilitas", "sppd", "perjalanan", "reimburse", "hardware"],
                            "Kesehatan": ["kesehatan", "wellbeing", "bpjs", "cuti", "melahirkan"],
                            "Lainnya": [],
                        }
                        _grouped = {k: [] for k in _kategori}
                        for _tema, _cnt in _top_tema.items():
                            _t_low = str(_tema).lower()
                            _placed = False
                            for _kat, _kw in _kategori.items():
                                if _kw and any(k in _t_low for k in _kw):
                                    _grouped[_kat].append((_tema, _cnt))
                                    _placed = True
                                    break
                            if not _placed:
                                _grouped["Lainnya"].append((_tema, _cnt))

                        for _kat, _items in _grouped.items():
                            if not _items:
                                continue
                            _safe_kat = _kat
                            _mermaid_lines.append(f"    {_safe_kat}")
                            for _tema, _cnt in _items:
                                _safe = str(_tema)[:30].replace('"', "'")
                                _mermaid_lines.append(f'      {_safe} [{_cnt}x]')

                        _mermaid_code = "\n".join(_mermaid_lines)
                        st.markdown(
                            f"<div class='mermaid'>{_mermaid_code}</div>"
                            "<script src='https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js'></script>"
                            "<script>mermaid.initialize({startOnLoad:true, theme:'base',"
                            "themeVariables:{primaryColor:'#4A0E1F',primaryTextColor:'#FAF5EE',"
                            "primaryBorderColor:'#C9A84C',lineColor:'#C9A84C',"
                            "secondaryColor:'#6B1527',tertiaryColor:'#FDF3DC'}});</script>",
                            unsafe_allow_html=True,
                        )
                        st.caption(
                            "Diagram otomatis dari hasil query. "
                            "Angka dalam [ ] menunjukkan jumlah kemunculan tema."
                        )

            # ── 📊 HASIL + VISUALISASI INTERAKTIF ────────────
            _skip_table = "latar_belakang" in df.columns if df is not None and len(df) > 0 else False
            if len(df) > 0 and tipe_chart != "metric":
                if not _skip_table:
                    st.markdown("<p style='font-family:DM Mono,monospace;font-size:0.68rem;letter-spacing:0.08em;text-transform:uppercase;color:#7A6A5A;margin:1.5rem 0 0.75rem 0;'>Hasil</p>", unsafe_allow_html=True)

                if tipe_chart in ("bar", "pie"):
                    col_label = df.columns[0]
                    col_val   = df.columns[1]

                    # ── Bar chart interaktif + klik untuk detail ──
                    if tipe_chart == "bar":
                        # Siapkan data detail per tema (jika ada)
                        _has_detail = col_label in ("tema", "kode_area")

                        fig = px.bar(
                            df,
                            x=col_val,
                            y=col_label,
                            orientation="h",
                            text=col_val,
                            color=col_val,
                            color_continuous_scale=[
                                [0, "#FDF3DC"], [0.4, "#C9A84C"],
                                [0.7, "#8B2035"], [1, "#2D0812"]
                            ],
                            custom_data=[col_label],
                        )
                        fig.update_traces(
                            textposition="outside",
                            hovertemplate=(
                                "<b>%{y}</b><br>"
                                f"{col_val.replace('_',' ').title()}: %{{x:,}}<br>"
                                "<i>Klik untuk lihat detail</i>"
                                "<extra></extra>"
                            ),
                        )
                        fig.update_layout(
                            yaxis={"categoryorder": "total ascending",
                                   "tickfont": {"size": 11}},
                            xaxis={"title": col_val.replace("_", " ").title()},
                            height=max(420, len(df) * 32),
                            showlegend=False,
                            coloraxis_showscale=False,
                            margin=dict(l=10, r=80, t=40, b=30),
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#2D0812"),
                            hoverlabel=dict(
                                bgcolor="#4A0E1F",
                                font_color="#FAF5EE",
                                font_size=13,
                            ),
                        )
                        fig.update_xaxes(showgrid=True, gridcolor="#E8D5B0",
                                         gridwidth=0.5)
                        fig.update_yaxes(showgrid=False)

                        st.plotly_chart(fig, use_container_width=True,
                                        key="main_chart")

                        # Detail langsung dari hasil query (tidak perlu dropdown)

                    # ── Donut chart untuk data kecil ─────────────
                    elif tipe_chart == "pie":
                        fig = px.pie(
                            df,
                            names=col_label,
                            values=col_val,
                            hole=0.45,
                            color_discrete_sequence=px.colors.sequential.RdPu_r,
                        )
                        fig.update_traces(
                            textposition="inside",
                            textinfo="percent+label",
                            hovertemplate=(
                                "<b>%{label}</b><br>"
                                "Jumlah: %{value:,}<br>"
                                "Porsi: %{percent}<extra></extra>"
                            ),
                        )
                        fig.update_layout(
                            height=420,
                            showlegend=True,
                            paper_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#2D0812"),
                        )
                        st.plotly_chart(fig, use_container_width=True)

                else:
                    # Tabel detail biasa — skip jika kartu detail sudah tampil
                    if not _skip_table:
                        st.dataframe(df, use_container_width=True, height=350)

                # Download selalu ada
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="hasil_lks.csv",
                    mime="text/csv",
                )

                # ── Kartu pembahasan berwarna jika ada kolom relevan ──
                _has_detail_cols = (
                    "latar_belakang" in df.columns
                    and tipe_chart not in ("metric", "pie", "bar")
                )
                if _has_detail_cols and len(df) > 0:
                    render_detail_pembahasan(df)

            # ── 🌳 TREE VIEW ──────────────────────────────────
            KOLOM_TEKS = ["latar_belakang", "rekomendasi", "tindak_lanjut"]
            ada_teks   = [c for c in KOLOM_TEKS if c in df.columns]

            # Jika ada kolom teks tapi data terbatas (LIMIT 10), re-query tanpa limit
            if ada_teks and len(df) > 0 and len(df) <= 10:
                try:
                    with engine.connect() as _rc:
                        _sql_full = re.sub(
                            r"\bLIMIT\s+\d+", "LIMIT 2000", sql, flags=re.IGNORECASE
                        )
                        df = pd.read_sql(text(_sql_full), _rc)
                except Exception:
                    pass  # pakai df yang ada jika re-query gagal

            if ada_teks and len(df) > 0:
                st.markdown("<p style='font-family:DM Mono,monospace;font-size:0.68rem;letter-spacing:0.08em;text-transform:uppercase;color:#7A6A5A;margin:1.5rem 0 0.75rem 0;'>Pengelompokkan Topik</p>", unsafe_allow_html=True)
                st.caption(
                    "Baris dikelompokkan berdasarkan topik/frasa yang sering muncul. "
                    "Pilih kolom lalu klik grup untuk melihat detailnya."
                )

                STOPWORDS = {
                    "yang","dan","di","ke","dari","dengan","untuk","pada","dalam",
                    "adalah","ini","itu","atau","juga","sudah","telah","akan",
                    "tidak","ada","oleh","para","agar","dapat","serta","karena",
                    "namun","sebagai","sesuai","terkait","terdapat","bahwa","hal",
                    "kami","unit","pln","bersama","bidang","lks","bipartit",
                    "mengenai","menindaklanjuti","pembahasan","menyampaikan",
                    "berdasarkan","saat","masih","belum","kepada","antara","tim",
                    "anggota","pihak","dinas","meminta","memiliki","dilakukan",
                    "tersebut","lebih","proses","pegawai","perlu","nomor","surat",
                    "adanya","hasil",
                }
                FRASA_PRIORITAS = [
                    "aplikasi e-sppd","esppd","e-sppd","e sppd",
                    "reimburse","reimbursement","perjalanan dinas","optimasi biaya",
                    "non diklat","non-diklat","tugas belajar","emergency exit",
                    "printer","laptop","hardware","pkb","perjanjian kerja",
                    "rekrutmen","rekrut","kesehatan","wellbeing","cuti","melahirkan",
                    "lembur","overtime","pagu","anggaran","kompetensi",
                    "pelatihan","diklat","jabatan","promosi","mutasi","pensiun","purna",
                ]

                def cari_frasa(teks):
                    t = str(teks).lower()
                    return [f for f in FRASA_PRIORITAS if f in t]

                def ekstrak_topik(seri, max_topik=8):
                    from collections import Counter
                    fc = Counter()
                    for t in seri.dropna():
                        for f in cari_frasa(t):
                            fc[f] += 1
                    topik = [f for f, _ in fc.most_common(max_topik)]
                    if len(topik) < 3:
                        semua = []
                        for t in seri.dropna():
                            kata = [k for k in re.findall(r"\b[a-zA-Z]{4,}\b", str(t).lower())
                                    if k not in STOPWORDS]
                            semua += [f"{kata[i]} {kata[i+1]}" for i in range(len(kata)-1)]
                            semua += kata
                        topik += [b for b, _ in Counter(semua).most_common(20)
                                  if b not in topik][: max_topik - len(topik)]
                    return topik[:max_topik]

                def tag_baris(teks, topik_list):
                    t = str(teks).lower()
                    for tp in topik_list:
                        if tp in t:
                            return tp
                    return "lainnya"

                pilih_kolom = st.selectbox(
                    "Tampilkan kolom:",
                    options=ada_teks,
                    format_func=lambda x: x.replace("_", " ").title(),
                    key="tree_kolom",
                )

                df_isi = df[
                    df[pilih_kolom].notna() &
                    (df[pilih_kolom].str.strip() != "")
                ].copy()

                if df_isi.empty:
                    st.info("Tidak ada data untuk ditampilkan.")
                else:
                    topik_list = ekstrak_topik(df_isi[pilih_kolom])
                    df_isi["_topik"] = df_isi[pilih_kolom].apply(lambda t: tag_baris(t, topik_list))
                    urutan = [t for t in topik_list if t in df_isi["_topik"].values]
                    if "lainnya" in df_isi["_topik"].values:
                        urutan += ["lainnya"]

                    WARNA = ["#1565C0","#2E7D32","#6A1B9A","#BF360C","#00695C",
                             "#E65100","#4527A0","#283593","#558B2F","#AD1457","#888"]

                    for i_tp, topik in enumerate(urutan):
                        subset = df_isi[df_isi["_topik"] == topik]
                        if subset.empty:
                            continue
                        n     = len(subset)
                        w     = WARNA[i_tp % len(WARNA)] if topik != "lainnya" else "#888"
                        areas = ""
                        if "kode_area" in subset.columns:
                            alist = sorted(subset["kode_area"].dropna().unique())[:12]
                            areas = " ".join(
                                f"<span style='background:#EEE;color:#555;padding:0 5px;"
                                f"border-radius:3px;font-size:0.72rem;'>{a}</span>"
                                for a in alist
                            )
                        st.markdown(
                            f"<div style='margin:12px 0 4px 0;'>"
                            f"<span style='background:{w};color:white;padding:3px 12px;"
                            f"border-radius:12px;font-size:0.82rem;font-weight:700;'>"
                            f"📌 {topik.upper()}</span>"
                            f"<span style='color:#888;font-size:0.8rem;margin-left:8px;'>"
                            f"{n} baris</span> &nbsp;{areas}</div>",
                            unsafe_allow_html=True,
                        )
                        with st.expander(f"Lihat {n} detail →", expanded=False):
                            for _, row in subset.iterrows():
                                ab = ""
                                if "kode_area" in row and pd.notna(row["kode_area"]):
                                    ab = (f"<span style='background:#E3F2FD;color:#1565C0;"
                                          f"padding:1px 7px;border-radius:4px;font-size:0.74rem;"
                                          f"margin-right:6px;font-weight:600;'>{row['kode_area']}</span>")
                                raw   = str(row[pilih_kolom])
                                shown = raw[:300] + "…" if len(raw) > 300 else raw
                                st.markdown(
                                    f"<div style='margin:4px 0;padding:6px 10px;"
                                    f"border-left:3px solid {w};background:#FAFAFA;"
                                    f"font-size:0.875rem;color:#333;border-radius:0 4px 4px 0;'>"
                                    f"{ab}{shown}</div>",
                                    unsafe_allow_html=True,
                                )
                            st.markdown("")

# ── 👍👎 FEEDBACK (di luar if jalankan_btn agar selalu dirender) ──
if st.session_state.get("history"):
    fb_idx = len(st.session_state.history) - 1
    fb_key = f"fb_{fb_idx}"
    if fb_key not in st.session_state:
        st.session_state[fb_key] = None

    st.markdown("---")
    st.markdown("**Apakah jawaban ini membantu?**")

    if st.session_state[fb_key] is None:
        fb1, fb2, _ = st.columns([1, 1, 8])
        if fb1.button("👍 Ya", key=f"btn_yes_{fb_key}"):
            st.session_state[fb_key] = "yes"
            st.rerun()
        if fb2.button("👎 Tidak", key=f"btn_no_{fb_key}"):
            st.session_state[fb_key] = "no"
            st.rerun()
    elif st.session_state[fb_key] == "yes":
        st.success("Terima kasih! Senang bisa membantu 😊")
    elif st.session_state[fb_key] == "no":
        st.markdown(
            "<div style='background:linear-gradient(135deg,#FFF8E1,#FFF3E0);"
            "border-left:5px solid #FFA000;border-radius:8px;"
            "padding:16px 20px;margin:8px 0;'>"
            "<p style='font-size:1.05rem;font-weight:700;"
            "color:#E65100;margin:0 0 6px 0;'>😔 Maaf, jawaban belum sesuai!</p>"
            "<p style='font-size:0.9rem;color:#555;margin:0;'>"
            "Terima kasih sudah memberi tahu kami. "
            "Coba ulangi pertanyaan dengan kata yang lebih spesifik, "
            "atau pilih pertanyaan pemantik di atas. "
            "Kami akan terus belajar untuk memberikan hasil yang lebih baik! 🙏"
            "</p></div>",
            unsafe_allow_html=True,
        )

# ── History ──────────────────────────────────────────
st.markdown("---")
st.subheader("🕘 History Query")

if st.session_state.history:
    for idx, item in enumerate(reversed(st.session_state.history[-10:])):
        with st.expander(
            f"#{len(st.session_state.history) - idx}: "
            f"{item['question']} ({item['rows']} rows)"
        ):
            st.code(item["sql"], language="sql")
else:
    st.info("Belum ada query yang dijalankan")

st.markdown("---")
st.caption("🔒 Query hanya SELECT  •  Guardrail aktif  •  Data: LKS Bipartit Feb–Apr 2026")
