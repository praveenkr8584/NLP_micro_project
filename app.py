"""Interactive web app: CBOW vs Skip-gram word similarity.

Run it from the project folder:

    streamlit run app.py

The first start loads the saved models (models/) or trains them (about 1 minute).
"""
import json
import random
from types import SimpleNamespace

import altair as alt
import pandas as pd
import streamlit as st

from src import compare, config, preprocess, train, viz

st.set_page_config(page_title="CBOW vs Skip-gram", page_icon=None, layout="wide")

# --------------------------------------------------------------------------- colours
# Validated chart palette. Blue = CBOW and orange = Skip-gram in EVERY chart, so a
# colour always means the same model. Text always uses the ink colours, never a series colour.
LIGHT = SimpleNamespace(
    dark=False, surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
    grid="#e1e0d9", baseline="#c3c2b7", cbow="#2a78d6", skip="#eb6834", panel="#f1f0ec",
)
DARK = SimpleNamespace(
    dark=True, surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
    grid="#2c2c2a", baseline="#383835", cbow="#3987e5", skip="#d95926", panel="#242422",
)
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def palette() -> SimpleNamespace:
    """Pick the colours that match the theme the viewer sees (light or dark)."""
    try:
        return DARK if st.context.theme.type == "dark" else LIGHT
    except Exception:
        return LIGHT


def style(chart: alt.Chart, pal: SimpleNamespace) -> alt.Chart:
    """Quiet chart chrome: hairline solid grid, muted axes, no frame."""
    return (
        chart.configure(background="rgba(0,0,0,0)", font=FONT)
        .configure_view(stroke=None)
        .configure_axis(
            labelColor=pal.ink2, titleColor=pal.ink2, gridColor=pal.grid,
            domainColor=pal.baseline, tickColor=pal.baseline,
            labelFontSize=12, titleFontSize=12, titleFontWeight="normal", titlePadding=10,
        )
        .configure_legend(
            labelColor=pal.ink2, titleColor=pal.ink2, labelFontSize=12, orient="bottom",
            symbolType="circle", symbolSize=110, padding=0, columnPadding=16,
        )
        .configure_title(color=pal.ink, fontSize=13, fontWeight="normal", anchor="start")
    )


def swatch(colour: str) -> str:
    return (
        f"<span style='display:inline-block;width:11px;height:11px;border-radius:3px;"
        f"background:{colour};margin-right:8px;vertical-align:baseline'></span>"
    )


# --------------------------------------------------------------------------- data (cached)
@st.cache_resource(show_spinner="Loading the models. If this is the first start, training both takes about 1 minute ...")
def load_everything():
    """Corpus, models and vocabulary. Models are loaded from models/ or trained once."""
    sentences = preprocess.load_sentences() if config.PROCESSED_TXT.exists() else None
    if sentences is None:
        sentences, stats = preprocess.build_corpus()
        preprocess.save_sentences(sentences)
        preprocess.save_stats(stats)
    models = train.load_models()
    trained_now = False
    if models is None:
        cbow, skip, _ = train.train_models(sentences)
        train.save_models(cbow, skip)
        models, trained_now = (cbow, skip), True
    cbow, skip = models
    return cbow, skip, list(cbow.wv.index_to_key), trained_now


@st.cache_data(show_spinner=False)
def corpus_stats() -> dict:
    path = config.RESULTS_DIR / "corpus_stats.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    _, stats = preprocess.build_corpus()
    return stats


@st.cache_data(show_spinner="Comparing the 200 most frequent words ...")
def evidence(_cbow, _skip, topn: int) -> dict:
    return {
        "overlap": compare.overlap_summary(_cbow, _skip, k=topn),
        "profile": {
            "CBOW": compare.neighbour_profile(_cbow, k=topn),
            "Skip-gram": compare.neighbour_profile(_skip, k=topn),
        },
        "wordsim": {
            "CBOW": compare.wordsim_evaluation(_cbow),
            "Skip-gram": compare.wordsim_evaluation(_skip),
        },
    }


@st.cache_data(show_spinner=False)
def agreement_for(_cbow, _skip, words: tuple) -> dict:
    return {w: compare.similarity_agreement(_cbow, _skip, w) for w in words}


# --------------------------------------------------------------------------- charts
def neighbour_bars(df: pd.DataFrame, colour: str, pal: SimpleNamespace, title: str):
    df = df.copy()
    df["label"] = [
        f"{c:.2f}" + ("   ◆ also in the other model" if s else "")
        for c, s in zip(df.cosine, df.shared)
    ]
    base = alt.Chart(df).encode(
        y=alt.Y("word:N", sort=None, title=None,
                axis=alt.Axis(ticks=False, domain=False, labelLimit=220, labelFontSize=13, labelPadding=8))
    )
    bars = base.mark_bar(size=16, cornerRadiusEnd=4, color=colour).encode(
        x=alt.X("cosine:Q", title="cosine similarity (1 = same direction)",
                scale=alt.Scale(domain=[0, 1.0]), axis=alt.Axis(tickCount=5, format=".1f")),
        tooltip=[alt.Tooltip("word:N", title="Word"), alt.Tooltip("cosine:Q", title="Cosine"),
                 alt.Tooltip("count:Q", title="Count in corpus", format=","),
                 alt.Tooltip("shared:N", title="Also in other model")],
    )
    labels = base.mark_text(align="left", dx=7, fontSize=12, color=pal.ink2).encode(
        x="cosine:Q", text="label:N"
    )
    return style((bars + labels).properties(height=max(150, 38 * len(df)), title=title), pal)


def word_map(df: pd.DataFrame, colour: str, pal: SimpleNamespace, topn: int, label_others: bool = False):
    kinds = ["Target word", f"Top {topn} neighbours", "Other nearby words"]
    scale = alt.Scale(domain=kinds, range=[pal.ink, colour, pal.muted])
    size = alt.Size("kind:N", scale=alt.Scale(domain=kinds, range=[320, 170, 100]), legend=None)
    x = alt.X("x:Q", axis=None, scale=alt.Scale(nice=True, padding=70))
    y = alt.Y("y:Q", axis=None, scale=alt.Scale(nice=True, padding=24))
    base = alt.Chart(df).encode(x=x, y=y)
    points = base.mark_circle(opacity=1, stroke=pal.surface, strokeWidth=2).encode(
        color=alt.Color("kind:N", scale=scale, legend=alt.Legend(title=None)),
        size=size,
        tooltip=[alt.Tooltip("word:N", title="Word"), alt.Tooltip("cosine:Q", title="Cosine to target"),
                 alt.Tooltip("count:Q", title="Count in corpus", format=","),
                 alt.Tooltip("in_other_top:N", title="Also top neighbour in other model")],
    )
    # Labels are selective: the target and the top-N neighbours always, the other nearby
    # words only on request (they are still in the hover tooltip and in the table view).
    shown = (df[df.kind != "Target word"] if label_others else df[df.kind == kinds[1]]).copy()
    # Labels point towards the middle of the map, so none is cut off at the edge.
    middle = (df.x.min() + df.x.max()) / 2
    shown["side"] = ["right" if px < middle else "left" for px in shown.x]
    label_base = alt.Chart(shown).encode(x=x, y=y, text="label:N")
    text = (
        label_base.transform_filter(alt.datum.side == "right")
        .mark_text(align="left", dx=9, dy=-1, fontSize=12, color=pal.ink2)
        + label_base.transform_filter(alt.datum.side == "left")
        .mark_text(align="right", dx=-9, dy=-1, fontSize=12, color=pal.ink2)
    )
    target = base.transform_filter(alt.datum.kind == "Target word").mark_text(
        align="left", dx=12, fontSize=14, fontWeight="bold", color=pal.ink).encode(text="word:N")
    chart = (points + text + target).properties(height=430)
    return style(chart.interactive(), pal)


def simple_bars(df: pd.DataFrame, x: str, y: str, pal, colour=None, colour_field=None,
                height=210, fmt=",.0f", title=None, x_title=None, domain=None):
    """Horizontal single-measure bars; colour follows the model (colour_field) or is neutral."""
    enc_x = alt.X(f"{x}:Q", title=x_title, axis=alt.Axis(format=fmt, tickCount=5),
                  scale=alt.Scale(domain=domain) if domain else alt.Undefined)
    base = alt.Chart(df).encode(
        y=alt.Y(f"{y}:N", sort=None, title=None, axis=alt.Axis(ticks=False, domain=False, labelFontSize=13, labelLimit=240)))
    if colour_field:
        col = alt.Color(f"{colour_field}:N", legend=None,
                        scale=alt.Scale(domain=["CBOW", "Skip-gram"], range=[pal.cbow, pal.skip]))
        bars = base.mark_bar(size=18, cornerRadiusEnd=4).encode(x=enc_x, color=col)
    else:
        bars = base.mark_bar(size=18, cornerRadiusEnd=4, color=colour or pal.muted).encode(x=enc_x)
    text = base.mark_text(align="left", dx=6, fontSize=12, color=pal.ink2).encode(
        x=f"{x}:Q", text=alt.Text(f"{x}:Q", format=fmt))
    return style((bars + text).properties(height=height, title=title or ""), pal)


def chips(words, colour_border: str, pal) -> str:
    if not words:
        return f"<span style='color:{pal.muted}'>none</span>"
    return " ".join(
        f"<span style='display:inline-block;padding:2px 9px;margin:2px 4px 2px 0;border-radius:10px;"
        f"border:1px solid {colour_border};color:{pal.ink}'>{w}</span>"
        for w in words
    )


# --------------------------------------------------------------------------- page
pal = palette()
cbow, skip, vocab, trained_now = load_everything()

st.title("CBOW vs Skip-gram: word similarity")
st.markdown(
    "**Question:** do CBOW and Skip-gram learn similar relationships between words?  \n"
    f"{swatch(pal.cbow)}**CBOW** predicts a word from its neighbours &nbsp;&nbsp;&nbsp; "
    f"{swatch(pal.skip)}**Skip-gram** predicts the neighbours from a word",
    unsafe_allow_html=True,
)
if trained_now:
    st.info("The saved models were missing, so both were trained now and saved in the models/ folder.")

# ---- sidebar: one control row that scopes every tab
if "word" not in st.session_state:
    st.session_state.word = "school"
if "topn" not in st.session_state:
    st.session_state.topn = config.TOPN


def pick(word: str):
    st.session_state.word = word


def pick_random():
    st.session_state.word = random.choice(vocab[:3000])


with st.sidebar:
    st.header("Choose a word")
    word = st.selectbox("Search or select a word", vocab, key="word",
                        help="Sorted by frequency. Type to search the whole vocabulary.")
    st.caption("The five words used in the report")
    cols = st.columns(2)
    for i, w in enumerate(config.SELECTED_WORDS):
        cols[i % 2].button(w, key=f"quick_{w}", on_click=pick, args=(w,), width="stretch")
    cols[1].button("random word", key="quick_random", on_click=pick_random, width="stretch")
    st.divider()
    topn = st.slider("Neighbours to compare (top N)", 3, 15, key="topn")
    st.divider()
    st.caption("Praveen Kumar  \nReg. No. 231FA04G42  \nMicro project: Word Similarity Using Word2Vec Models")

tab_explore, tab_table, tab_map, tab_evidence, tab_how = st.tabs(
    ["Explore a word", "Assignment table", "Word map", "Evidence", "How it works"]
)

# =========================================================================== tab 1
with tab_explore:
    res = viz.compare_word(cbow, skip, word, topn)
    c_med = int(res["cbow"]["count"].median())
    s_med = int(res["skipgram"]["count"].median())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Word", word, help=f"Appears {res['count']:,} times in the cleaned corpus (frequency rank {res['rank']:,}).")
    m2.metric("Shared top neighbours", f"{res['overlap']} of {topn}",
              help="How many of the top-N neighbours are the same in both models.")
    m3.metric("Agreement, whole vocabulary", f"{res['agreement']:.2f}",
              help="Spearman correlation between the two models' similarity scores of this word to ALL other words. "
                   "1 = identical ordering, 0 = unrelated.")
    m4.metric("Median neighbour count (CBOW vs Skip-gram)", f"{c_med:,} vs {s_med:,}",
              help="How often the neighbours occur in the corpus. A lower number means rarer, more specific words.")

    left, right = st.columns(2, gap="large")
    for col, key, label, colour in ((left, "cbow", "CBOW", pal.cbow), (right, "skipgram", "Skip-gram", pal.skip)):
        with col:
            st.markdown(f"#### {swatch(colour)}{label}", unsafe_allow_html=True)
            st.altair_chart(neighbour_bars(res[key], colour, pal, f"Top {topn} words most similar to “{word}”"),
                            theme=None, width="stretch")
            with st.expander("Table view"):
                st.dataframe(res[key].rename(columns={"count": "count in corpus"}), hide_index=True,
                             width="stretch")

    st.markdown(
        f"**Both models:** {chips(res['shared'], pal.baseline, pal)}  \n"
        f"**Only CBOW:** {chips(res['only_cbow'], pal.cbow, pal)}  \n"
        f"**Only Skip-gram:** {chips(res['only_skipgram'], pal.skip, pal)}",
        unsafe_allow_html=True,
    )
    st.caption(
        "Cosine scores come from two different models, so compare the order of the words more than the exact "
        "numbers. Both charts use the same 0 to 1 axis."
    )

# =========================================================================== tab 2
with tab_table:
    st.subheader("Comparison table")
    st.caption("The table required by the assignment. Add or remove words to explore more.")
    chosen = st.multiselect("Words in the table", vocab, default=config.SELECTED_WORDS)
    if chosen:
        table = compare.comparison_table(cbow, skip, chosen, topn).rename(
            columns={"CBOW top 5": f"CBOW top {topn}", "Skip-gram top 5": f"Skip-gram top {topn}"}
        )
        # st.table wraps long text, so the whole neighbour lists stay visible on any screen width.
        shown_table = table.copy()
        shown_table["Common words"] = shown_table["Common words"].replace("-", "none")
        st.table(shown_table.set_index("Word"))
        st.download_button("Download as CSV", table.to_csv(index=False).encode("utf-8"),
                           file_name="comparison_table.csv", mime="text/csv")

        ag = agreement_for(cbow, skip, tuple(chosen))
        ov = pd.DataFrame({"word": chosen,
                           "common neighbours": [compare.overlap_at_k(cbow, skip, w, topn) for w in chosen]})
        st.markdown(f"##### Common neighbours per word (out of {topn})")
        st.altair_chart(
            simple_bars(ov, "common neighbours", "word", pal, height=max(120, 36 * len(ov)), fmt="d",
                        domain=[0, topn]),
            theme=None, width="stretch")
        with st.expander("Agreement over the whole vocabulary (Spearman)"):
            st.dataframe(pd.DataFrame({"word": list(ag), "Spearman": [round(v, 3) for v in ag.values()]}),
                         hide_index=True, width="stretch")
    else:
        st.warning("Select at least one word.")

# =========================================================================== tab 3
with tab_map:
    st.subheader(f"Word map for “{word}”")
    st.caption(
        "Each panel is a 2D picture (PCA) of the word and its nearest neighbours in one model. "
        "Close points are similar. Positions can only be compared inside one panel, because the two "
        "models have different vector spaces. Scroll to zoom and drag to move."
    )
    o1, o2 = st.columns([3, 1], vertical_alignment="bottom")
    n_near = o1.slider("Nearby words to draw", max(topn, 8), 40, max(topn, 15))
    label_others = o2.checkbox("Label the other nearby words", value=False)
    c_top = [w for w, _ in cbow.wv.most_similar(word, topn=topn)]
    s_top = [w for w, _ in skip.wv.most_similar(word, topn=topn)]
    left, right = st.columns(2, gap="large")
    for col, model, other_top, label, colour in (
        (left, cbow, s_top, "CBOW", pal.cbow), (right, skip, c_top, "Skip-gram", pal.skip)
    ):
        with col:
            df_map, explained = viz.pca_map(model, word, n_near, topn, other_top)
            st.markdown(f"#### {swatch(colour)}{label}", unsafe_allow_html=True)
            st.altair_chart(word_map(df_map, colour, pal, topn, label_others), theme=None, width="stretch")
            st.caption(f"The two axes keep {explained:.0%} of the information in the vectors.")
            with st.expander("Table view"):
                st.dataframe(df_map[["word", "kind", "cosine", "count", "in_other_top"]]
                             .rename(columns={"count": "count in corpus",
                                              "in_other_top": "also top N in other model"}),
                             hide_index=True, width="stretch")
    st.caption("◆ after a word = this word is also a top-N neighbour in the other model.")

# =========================================================================== tab 4
with tab_evidence:
    ev = evidence(cbow, skip, topn)
    ov = ev["overlap"]
    st.subheader("Evidence from the whole corpus, not only five words")
    st.caption(f"The {ov['words_checked']} most frequent words, top {topn} neighbours each.")

    e1, e2, e3, e4 = st.columns(4)
    e1.metric(f"Mean overlap (of {topn})", f"{ov['mean_overlap']:.2f}",
              help=f"Random neighbour lists would overlap by only {ov['chance_level_mean_overlap']:.3f}.")
    e2.metric("Words with at least 1 shared", f"{ov['share_with_at_least_1_common']:.0%}")
    e3.metric("WordSim-353, CBOW", f"{ev['wordsim']['CBOW']['spearman']:.2f}",
              help="Spearman correlation with human similarity ratings (higher is better).")
    e4.metric("WordSim-353, Skip-gram", f"{ev['wordsim']['Skip-gram']['spearman']:.2f}")

    a, b = st.columns(2, gap="large")
    with a:
        st.markdown(f"##### Shared neighbours among the top {topn}")
        dist = pd.DataFrame({"shared": [str(k) for k in ov["distribution"]],
                             "words": list(ov["distribution"].values())})
        st.altair_chart(
            style(alt.Chart(dist).mark_bar(size=34, cornerRadiusEnd=4, color=pal.muted).encode(
                x=alt.X("shared:N", sort=None, title=f"shared neighbours (out of {topn})", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("words:Q", title="number of words"),
                tooltip=[alt.Tooltip("shared:N", title="Shared"), alt.Tooltip("words:Q", title="Words")],
            ).properties(height=260), pal),
            theme=None, width="stretch")
        st.caption(f"Chance level would put almost all words at 0. Mean overlap here is {ov['mean_overlap']:.2f}.")
        with st.expander("Table view"):
            st.dataframe(dist, hide_index=True, width="stretch")
    with b:
        st.markdown("##### How common are the neighbours in the corpus?")
        prof = pd.DataFrame({
            "model": list(ev["profile"]),
            "median count": [v["median_neighbour_count_in_corpus"] for v in ev["profile"].values()],
        })
        st.altair_chart(simple_bars(prof, "median count", "model", pal, colour_field="model", height=150,
                                    x_title="median number of times the neighbour occurs in the corpus"),
                        theme=None, width="stretch")
        st.caption("A smaller number means the model prefers rarer, more specific neighbours.")
        st.markdown("##### Agreement with human ratings (WordSim-353)")
        ws = pd.DataFrame({"model": list(ev["wordsim"]),
                           "Spearman": [v["spearman"] for v in ev["wordsim"].values()]})
        st.altair_chart(simple_bars(ws, "Spearman", "model", pal, colour_field="model", height=150, fmt=".3f",
                                    domain=[0, 1], x_title="Spearman correlation with people (higher is better)"),
                        theme=None, width="stretch")
        cov = 100 - ev["wordsim"]["CBOW"]["pairs_not_covered_percent"]
        st.caption(f"Only the {cov:.0f}% of word pairs whose words are in this corpus vocabulary are counted.")

    st.divider()
    s1, s2 = st.columns(2, gap="large")
    with s1:
        st.markdown("##### Corpus")
        all_stats = corpus_stats()
        st.caption(f"Source: {all_stats.get('source', 'Brown Corpus')}")
        stats = {k: v for k, v in all_stats.items() if k not in ("sentences_per_genre_letter", "source")}
        st.dataframe(pd.DataFrame({"measure": [k.replace("_", " ") for k in stats],
                                   "value": [f"{v:,}" if isinstance(v, int) else str(v) for v in stats.values()]}),
                     hide_index=True, width="stretch")
    with s2:
        st.markdown("##### Model settings (only `sg` differs)")
        st.caption("sg = 0 is CBOW, sg = 1 is Skip-gram.")
        c_set, s_set = train.model_settings(cbow), train.model_settings(skip)
        fmt = lambda v: f"{v:,}" if isinstance(v, int) else str(v)
        st.dataframe(pd.DataFrame({"setting": list(c_set), "CBOW": [fmt(v) for v in c_set.values()],
                                   "Skip-gram": [fmt(s_set[k]) for k in c_set]}),
                     hide_index=True, width="stretch")

# =========================================================================== tab 5
with tab_how:
    st.subheader("How the two models learn")
    st.caption("Type a sentence and choose a window. The tables show the training examples each model would get.")
    sentence = st.text_input("Sentence", "Students learn programming in college.")
    window = st.slider("Window (words on each side)", 1, 4, 2)
    tokens = viz.demo_tokens(sentence)
    if len(tokens) < 2:
        st.warning("Please type a sentence with at least two words.")
    else:
        centre = st.selectbox("Centre (target) word", tokens, index=min(2, len(tokens) - 1))
        cbow_ex, skip_ex = viz.training_examples(tokens, window)
        idx = tokens.index(centre)
        one = next(e for e in cbow_ex if e["centre_index"] == idx)
        pairs = [e["context_word"] for e in skip_ex if e["centre_index"] == idx]

        ctx_html = " ".join(
            f"<span style='display:inline-block;padding:3px 11px;margin:2px;border-radius:10px;"
            f"border:1px solid {pal.baseline}'>{w}</span>" for w in one["context"])
        tgt = lambda colour: (f"<span style='display:inline-block;padding:3px 11px;margin:2px;border-radius:10px;"
                              f"border:2px solid {colour};font-weight:600'>{centre}</span>")
        arrow = f"<span style='color:{pal.muted}'>&nbsp;&rarr;&nbsp;</span>"

        a, b = st.columns(2, gap="large")
        with a:
            st.markdown(f"#### {swatch(pal.cbow)}CBOW", unsafe_allow_html=True)
            st.markdown("**One** training example for this centre word:", unsafe_allow_html=True)
            st.markdown(f"{ctx_html}{arrow}{tgt(pal.cbow)}", unsafe_allow_html=True)
            st.caption("The surrounding words together predict the centre word.")
        with b:
            st.markdown(f"#### {swatch(pal.skip)}Skip-gram", unsafe_allow_html=True)
            st.markdown(f"**{len(pairs)}** training examples for this centre word:")
            st.markdown("<br>".join(
                f"{tgt(pal.skip)}{arrow}<span style='display:inline-block;padding:3px 11px;margin:2px;"
                f"border-radius:10px;border:1px solid {pal.baseline}'>{p}</span>" for p in pairs),
                unsafe_allow_html=True)
            st.caption("The centre word predicts each surrounding word, one at a time.")

        t1, t2, t3 = st.columns(3)
        t1.metric("CBOW examples in this sentence", len(cbow_ex))
        t2.metric("Skip-gram examples in this sentence", len(skip_ex))
        t3.metric("Skip-gram makes", f"{len(skip_ex) / max(len(cbow_ex), 1):.1f}x as many")
        note = "Skip-gram makes more examples, so it trains more slowly."
        settings_file = config.RESULTS_DIR / "model_settings.json"
        if settings_file.exists():
            timing = json.loads(settings_file.read_text(encoding="utf-8")).get("timings")
            if timing:
                note += (f" Measured in this project: CBOW {timing['cbow_seconds']} s, "
                         f"Skip-gram {timing['skipgram_seconds']} s.")
        st.caption(note + " gensim also shrinks the window randomly at each step, so the window is a maximum.")
