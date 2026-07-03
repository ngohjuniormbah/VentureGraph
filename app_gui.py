#!/usr/bin/env python3
"""
VentureGraph Streamlit dashboard — the user-facing demo.

This is deliberately a thin client: it never imports Docling, Instructor,
or the Anthropic SDK directly, and it never sees `ANTHROPIC_API_KEY` or any
search API key. Every button here makes an HTTP call to the VentureGraph
API (`app_api.py`), which does the actual PDF parsing, LLM calls, and web
search, and holds all the secrets. That split is what makes it possible to
deploy this dashboard on a lightweight host like Hugging Face Spaces while
the heavier backend runs elsewhere (see `deployment_guide.md`).

Run locally:
    streamlit run app_gui.py

Configuration:
    Set the backend's URL either via the `VENTUREGRAPH_API_URL` environment
    variable, or a `[connection] api_url` entry in `.streamlit/secrets.toml`;
    it also has a sidebar field so it can be changed at runtime for a quick demo.
"""

import os

import requests
import streamlit as st

DEFAULT_API_URL = os.environ.get("VENTUREGRAPH_API_URL", "http://localhost:8000")
# Investment Thesis generation makes several Claude + web-search calls back
# to back; give it much more room than the default requests timeout.
THESIS_TIMEOUT_SECONDS = 300
DEFAULT_TIMEOUT_SECONDS = 120

st.set_page_config(page_title="VentureGraph", page_icon="🔬", layout="wide")


def _get_api_url() -> str:
    """
    Resolve the backend API URL to call.

    Data flow:
        Checks `st.secrets["connection"]["api_url"]` first (the standard
        place for a deployed Streamlit app's config), falls back to the
        `VENTUREGRAPH_API_URL` environment variable, and finally to
        `http://localhost:8000` for local development. The sidebar text
        input (in `main()`) lets this be overridden per-session on top of
        all of that, without needing a restart.

    Returns:
        The backend base URL, with no trailing slash.
    """
    try:
        api_url = st.secrets["connection"]["api_url"]
    except Exception:
        api_url = DEFAULT_API_URL
    return api_url.rstrip("/")


def _post_pdf(api_url: str, endpoint: str, files, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> dict:
    """
    POST one or more PDFs to a VentureGraph API endpoint and return the JSON body.

    Data flow:
        Sends a multipart/form-data POST request to `f"{api_url}{endpoint}"`
        with `files` (in the shape `requests` expects for multipart
        uploads), and either returns the parsed JSON response or raises a
        `RuntimeError` with the backend's error detail so callers can show
        it to the user via `st.error`.

    Args:
        api_url: Backend base URL (no trailing slash).
        endpoint: API path, e.g. `"/parse"`.
        files: A `requests`-compatible files payload (list of tuples).
        timeout: Request timeout in seconds.

    Returns:
        The parsed JSON response body as a dict.

    Raises:
        RuntimeError: If the request fails or the backend returns a non-2xx
            status; the message includes the backend's error detail when
            available.
    """
    try:
        response = requests.post(f"{api_url}{endpoint}", files=files, timeout=timeout)
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Could not reach the VentureGraph API at {api_url}: {exc}") from exc

    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"API error ({response.status_code}): {detail}")

    return response.json()


def _render_scientific_essence(essence: dict) -> None:
    st.subheader("Scientific Essence")
    st.markdown(f"**Title:** {essence['paper_title']}")
    authors = ", ".join(a["name"] for a in essence.get("authors", [])) or "Unknown"
    st.markdown(f"**Authors:** {authors}")
    st.markdown(f"**Problem Statement:** {essence['problem_statement']}")
    st.markdown(f"**Methodology:** {essence['methodology']}")
    st.markdown(f"**Key Results:** {essence['key_results']}")


def _render_orkg_triples(contribution: dict) -> None:
    st.subheader("ORKG Contribution")
    st.markdown(f"**Research Problem:** {contribution['research_problem']}")
    st.markdown(f"**Contribution:** {contribution['contribution_label']}")
    triples = contribution.get("triples", [])
    if triples:
        st.table(
            [{"Subject": t["subject"], "Predicate": t["predicate"], "Object": t["object"]} for t in triples]
        )
    else:
        st.info("No grounded triples survived the hallucination filter for this paper.")


def _render_parse_tab(api_url: str) -> None:
    st.header("Parse a Paper")
    st.caption("Docling PDF -> Markdown conversion, plus a quick heuristic Scientific Essence. No LLM call.")
    uploaded = st.file_uploader("Upload a scientific PDF", type="pdf", key="parse_upload")

    if uploaded and st.button("Parse", key="parse_button"):
        with st.spinner("Converting PDF to Markdown..."):
            try:
                result = _post_pdf(api_url, "/parse", files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")})
            except RuntimeError as exc:
                st.error(str(exc))
                return

        _render_scientific_essence(result["essence"])
        with st.expander("Show extracted Markdown"):
            st.markdown(result["markdown"])


def _render_orkg_tab(api_url: str) -> None:
    st.header("ORKG Contribution Triples")
    st.caption("Extracts Subject-Predicate-Object triples via Claude, grounded in verbatim quotes from the paper.")
    uploaded = st.file_uploader("Upload a scientific PDF", type="pdf", key="orkg_upload")

    if uploaded and st.button("Extract ORKG Triples", key="orkg_button"):
        with st.spinner("Parsing PDF and extracting contribution triples (calls Claude)..."):
            try:
                result = _post_pdf(api_url, "/orkg", files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")})
            except RuntimeError as exc:
                st.error(str(exc))
                return

        _render_orkg_triples(result["contribution"])
        with st.expander("Show extracted Markdown"):
            st.markdown(result["markdown"])


def _render_compare_tab(api_url: str) -> None:
    st.header("Compare Papers")
    st.caption("Upload two or more papers to compare their reported benchmark results.")
    uploaded_files = st.file_uploader(
        "Upload two or more scientific PDFs", type="pdf", accept_multiple_files=True, key="compare_upload"
    )

    if uploaded_files and len(uploaded_files) >= 2 and st.button("Compare", key="compare_button"):
        with st.spinner(f"Parsing {len(uploaded_files)} papers and extracting benchmarks (calls Claude)..."):
            files = [("files", (f.name, f.getvalue(), "application/pdf")) for f in uploaded_files]
            try:
                result = _post_pdf(api_url, "/compare", files=files, timeout=THESIS_TIMEOUT_SECONDS)
            except RuntimeError as exc:
                st.error(str(exc))
                return

        st.markdown(result["comparison_markdown"])
    elif uploaded_files and len(uploaded_files) < 2:
        st.warning("Upload at least two PDFs to compare.")


def _render_tam_estimate(tam: dict) -> None:
    st.markdown(f"**Estimated TAM: ${tam['tam_usd']:,.0f}**")
    if tam.get("assumptions"):
        st.markdown("**Assumptions:**")
        for assumption in tam["assumptions"]:
            st.markdown(f"- {assumption}")
    with st.expander("Show TAM calculation"):
        st.code(tam["calculation_code"], language="python")
        st.text(tam["calculation_output"])
    if tam.get("data_sources"):
        st.caption("Sources: " + ", ".join(tam["data_sources"]))


def _render_thesis_tab(api_url: str) -> None:
    st.header("Investment Thesis")
    st.caption(
        "Full pipeline: ORKG contribution + 3 startup ideas, each with live competitor research and a "
        "TAM calculation. This calls Claude several times and runs live web searches, so it can take a "
        "minute or two."
    )
    uploaded = st.file_uploader("Upload a scientific PDF", type="pdf", key="thesis_upload")

    if uploaded and st.button("Generate Investment Thesis", key="thesis_button"):
        with st.spinner("Running the full Venture Catalyst pipeline (this can take a minute or two)..."):
            try:
                result = _post_pdf(
                    api_url,
                    "/thesis",
                    files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                    timeout=THESIS_TIMEOUT_SECONDS,
                )
            except RuntimeError as exc:
                st.error(str(exc))
                return

        thesis = result["thesis"]
        _render_scientific_essence(thesis["scientific_essence"])
        _render_orkg_triples(thesis["orkg_contribution"])

        st.subheader("Startup Ideas & Market Analysis")
        for i, dossier in enumerate(thesis["idea_dossiers"], start=1):
            idea = dossier["idea"]
            with st.container(border=True):
                st.markdown(f"### {i}. {idea['name']}")
                st.markdown(f"**Description:** {idea['description']}")
                st.markdown(f"**Target Customer:** {idea['target_customer']}")
                st.markdown(f"**Commercial Angle:** {idea['commercial_angle']}")
                st.caption(f"Grounded in methodology: \"{idea['methodology_basis']}\"")

                competitors = dossier["competitors"]
                st.markdown("**Competitive Landscape**")
                st.markdown(competitors["summary"])
                if competitors.get("competitors"):
                    st.table(
                        [
                            {"Name": c["name"], "Description": c["description"], "URL": c.get("url") or "—"}
                            for c in competitors["competitors"]
                        ]
                    )

                st.markdown("**Total Addressable Market**")
                _render_tam_estimate(dossier["tam"])

        with st.expander("Download full report as Markdown"):
            st.download_button(
                "Download Investment Thesis (.md)",
                data=result["report_markdown"],
                file_name=f"{thesis['paper_title']}.thesis.md",
                mime="text/markdown",
            )
            st.markdown(result["report_markdown"])


def main() -> None:
    """
    Streamlit entry point: render the sidebar (API URL config) and the four
    pipeline tabs (Parse, ORKG, Compare, Investment Thesis).

    Data flow:
        Resolves the backend API URL via `_get_api_url()`, lets the user
        override it in the sidebar, and renders one tab per VentureGraph
        pipeline stage. Each tab collects a file upload, calls the
        corresponding backend endpoint via `_post_pdf`, and renders the
        JSON response - no pipeline logic lives in this file itself.
    """
    st.title("🔬 VentureGraph")
    st.caption("From scientific papers to ORKG contributions and startup Investment Theses.")

    with st.sidebar:
        st.header("Settings")
        api_url = st.text_input("Backend API URL", value=_get_api_url())
        if st.button("Check connection"):
            try:
                response = requests.get(f"{api_url.rstrip('/')}/health", timeout=10)
                if response.ok:
                    st.success("Connected to the VentureGraph API.")
                else:
                    st.error(f"API responded with status {response.status_code}.")
            except requests.exceptions.RequestException as exc:
                st.error(f"Could not reach the API: {exc}")

    api_url = api_url.rstrip("/")

    tab_parse, tab_orkg, tab_compare, tab_thesis = st.tabs(
        ["📄 Parse", "🕸️ ORKG Contribution", "⚖️ Compare Papers", "💰 Investment Thesis"]
    )
    with tab_parse:
        _render_parse_tab(api_url)
    with tab_orkg:
        _render_orkg_tab(api_url)
    with tab_compare:
        _render_compare_tab(api_url)
    with tab_thesis:
        _render_thesis_tab(api_url)


if __name__ == "__main__":
    main()
