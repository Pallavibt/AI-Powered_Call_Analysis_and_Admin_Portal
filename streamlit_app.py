
import os
from urllib.parse import urljoin

import pandas as pd
import requests
import streamlit as st


API_URL = os.getenv(
    "API_URL", "http://127.0.0.1:8000"
).rstrip("/")

st.set_page_config(
    page_title="AI Call Analysis Admin",
    page_icon="📞",
    layout="wide",
)


def api_get(path, timeout=180):
    url = urljoin(API_URL + "/", path.lstrip("/"))
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response


@st.cache_data(ttl=30)
def fetch_calls():
    return api_get("/api/calls").json()["calls"]


@st.cache_data(ttl=300)
def fetch_analysis(call_id):
    return api_get(f"/api/calls/{call_id}/analyze").json()


st.title("AI-Powered Call Analysis")
st.caption("Admin portal for customer call analysis")

with st.sidebar:
    st.header("Admin controls")
    st.write("Backend API")
    st.code(API_URL)

    if st.button("Refresh data"):
        fetch_calls.clear()
        fetch_analysis.clear()
        st.rerun()

    st.caption("All sample calls are synthetic.")


try:
    calls = fetch_calls()
except requests.RequestException as exc:
    st.error("Cannot connect to the FastAPI backend.")
    st.code("python -m uvicorn app.main:app --reload")
    st.write(str(exc))
    st.stop()


if not calls:
    st.warning("No calls found.")
    st.stop()


analyses = []
errors = []

with st.spinner("Analyzing sample calls..."):
    for call in calls:
        try:
            analyses.append(fetch_analysis(call["id"]))
        except requests.RequestException as exc:
            errors.append(f"{call['id']}: {exc}")


if errors:
    st.warning("Some calls could not be analyzed.")
    with st.expander("View errors"):
        for error in errors:
            st.write(error)


if analyses:
    positive = sum(a["sentiment"] == "Positive" for a in analyses)
    neutral = sum(a["sentiment"] == "Neutral" for a in analyses)
    negative = sum(a["sentiment"] == "Negative" for a in analyses)

    average_probability = sum(
        a["purchase_probability"] for a in analyses
    ) / len(analyses)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total calls", len(calls))
    c2.metric("Positive", positive)
    c3.metric("Neutral", neutral)
    c4.metric("Negative", negative)
    c5.metric("Average purchase estimate", f"{average_probability:.1f}%")

    st.subheader("Call overview")

    overview = pd.DataFrame([
        {
            "Call ID": a["call_id"],
            "Customer": a["customer"],
            "Sentiment": a["sentiment"],
            "Purchase estimate (%)": a["purchase_probability"],
        }
        for a in analyses
    ])

    st.dataframe(overview, use_container_width=True, hide_index=True)


st.divider()
st.subheader("Individual call analysis")

options = {
    f"{call['id']} — {call['title']}": call
    for call in calls
}

selected_label = st.selectbox("Select a call", list(options.keys()))
selected_call = options[selected_label]

try:
    analysis = fetch_analysis(selected_call["id"])
except requests.RequestException as exc:
    st.error(f"Could not analyze this call: {exc}")
    st.stop()


left, right = st.columns(2)

with left:
    st.markdown(f"### {selected_call['title']}")
    st.write(f"**Call ID:** {selected_call['id']}")
    st.write(f"**Customer:** {selected_call['customer']}")
    st.write(f"**Scenario:** {selected_call['scenario']}")

    try:
        audio_response = api_get(
            f"/static/audio/{selected_call['id']}.wav"
        )
        st.audio(audio_response.content, format="audio/wav")
    except requests.RequestException:
        st.warning("Audio unavailable. Regenerate the WAV files.")

    st.markdown("#### Call transcript")
    st.write(analysis["transcript"])


with right:
    st.markdown("### AI analysis")

    sentiment = analysis["sentiment"]

    if sentiment == "Positive":
        st.success(f"Sentiment: {sentiment}")
    elif sentiment == "Negative":
        st.error(f"Sentiment: {sentiment}")
    else:
        st.info(f"Sentiment: {sentiment}")

    st.write(
        "Sentiment model confidence: "
        f"**{analysis['sentiment_confidence']:.2f}%**"
    )

    probability = analysis["purchase_probability"]
    st.metric("Estimated purchase probability", f"{probability:.2f}%")
    st.progress(max(0.0, min(1.0, probability / 100)))

    st.caption(
        "Illustrative model output, not a validated real-world forecast."
    )


st.markdown("### Detailed insights")

for index, insight in enumerate(analysis["insights"], start=1):
    with st.expander(f"Insight {index}", expanded=True):
        st.write(insight)


st.markdown("### Extracted customer signals")

features_df = pd.DataFrame([
    {"Feature": name.replace("_", " ").title(), "Value": value}
    for name, value in analysis["features"].items()
])

st.dataframe(features_df, use_container_width=True, hide_index=True)


st.markdown("### Recommended next actions")

for recommendation in analysis["recommendations"]:
    st.markdown(f"- {recommendation}")


st.markdown("### Download call report")

try:
    report = api_get(
        f"/api/calls/{selected_call['id']}/report"
    )

    st.download_button(
        label="Download PDF report",
        data=report.content,
        file_name=f"{selected_call['id']}_report.pdf",
        mime="application/pdf",
    )
except requests.RequestException as exc:
    st.error(f"Could not generate report: {exc}")


with st.expander("Model explanation and limitations"):
    st.write(
        "A pretrained Transformer classifies sentiment. Logistic "
        "Regression estimates purchase likelihood from extracted "
        "features. The purchase model uses synthetic training examples."
    )
    st.write(analysis["model_note"])
