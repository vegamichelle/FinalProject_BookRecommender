"""Streamlit user app: pick favourite books, get recommendations.

Talks to the FastAPI backend over HTTP (BACKEND_URL). All inference goes
through the API — the frontend never loads the model or touches the DB.
"""
import os

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Book Recommender", page_icon="📚")
st.title("📚 Personalized Book Recommender")
st.caption(f"Backend: {BACKEND_URL}")


@st.cache_data(ttl=60)
def get_catalog() -> list[str]:
    try:
        return requests.get(f"{BACKEND_URL}/catalog", params={"n": 40}, timeout=5).json()[
            "titles"
        ]
    except Exception:
        return []


# Health indicator.
try:
    h = requests.get(f"{BACKEND_URL}/health", timeout=5).json()
    st.success("Backend online · model loaded") if h.get("model_loaded") else st.warning(
        "Backend online · model NOT loaded"
    )
except Exception as exc:  # noqa: BLE001
    st.error(f"Cannot reach backend: {exc}")

catalog = get_catalog()

st.subheader("Tell us a few books you love")
picked = st.multiselect("Pick from the catalogue", options=catalog)
freetext = st.text_input("…or type titles (comma-separated)", "")
user_id = st.text_input("User ID (optional — enables caching)", "")
n = st.slider("How many recommendations?", 3, 20, 10)

favorites = list(picked)
if freetext.strip():
    favorites += [t.strip() for t in freetext.split(",") if t.strip()]

if st.button("Recommend books", type="primary"):
    if not favorites:
        st.warning("Add at least one favourite book.")
    else:
        payload = {"favorite_titles": favorites, "n": n}
        if user_id.strip():
            payload["user_id"] = user_id.strip()
        try:
            r = requests.post(f"{BACKEND_URL}/recommend", json=payload, timeout=10)
            if r.status_code == 200:
                data = r.json()
                st.session_state["last_request_id"] = data["request_id"]
                tag = "cache" if data["cache_hit"] else data["source"]
                st.caption(f"Source: {tag} · latency {data['latency_ms']:.1f} ms")
                for i, rec in enumerate(data["recommendations"], 1):
                    st.markdown(
                        f"**{i}. {rec['title']}** — *{rec['author']}* "
                        f"· {rec['genre']}  \n"
                        f"<span style='color:gray'>score {rec['score']}</span>",
                        unsafe_allow_html=True,
                    )
            else:
                st.error(f"Error {r.status_code}: {r.json().get('detail')}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Request failed: {exc}")

# Feedback for live relevance tracking.
if "last_request_id" in st.session_state:
    st.divider()
    st.subheader("Were these recommendations helpful?")
    c1, c2 = st.columns(2)

    def _send_feedback(helpful: bool) -> None:
        try:
            requests.post(
                f"{BACKEND_URL}/feedback",
                json={
                    "request_id": st.session_state["last_request_id"],
                    "helpful": helpful,
                },
                timeout=10,
            )
            st.success("Thanks — feedback recorded for monitoring.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Request failed: {exc}")

    with c1:
        if st.button("👍 Helpful"):
            _send_feedback(True)
    with c2:
        if st.button("👎 Not helpful"):
            _send_feedback(False)
