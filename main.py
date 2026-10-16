"""Run the whole project from the command line:

    python main.py

Steps: clean corpus -> train CBOW + Skip-gram -> compare -> write results/.
The notebook does the same thing step by step with explanations.
"""
import json

from src import compare, config, preprocess, train


def main() -> None:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("1/4 Cleaning corpus ...")
    sentences, stats = preprocess.build_corpus()
    preprocess.save_sentences(sentences)
    preprocess.save_stats(stats)
    print(f"    {stats['sentences_kept']} sentences, {stats['tokens_after_cleaning']} tokens")

    print("2/4 Training CBOW and Skip-gram (same settings, only sg differs) ...")
    cbow, skip, timings = train.train_models(sentences)
    train.save_models(cbow, skip)
    settings = {"cbow": train.model_settings(cbow), "skipgram": train.model_settings(skip),
                "timings": timings}
    (config.RESULTS_DIR / "model_settings.json").write_text(json.dumps(settings, indent=2))
    print(f"    CBOW {timings['cbow_seconds']}s, Skip-gram {timings['skipgram_seconds']}s")

    print("3/4 Comparing the selected words ...")
    table = compare.comparison_table(cbow, skip, config.SELECTED_WORDS)
    table.to_csv(config.RESULTS_DIR / "comparison_table.csv", index=False)
    compare.neighbours_with_scores(cbow, skip, config.SELECTED_WORDS).to_csv(
        config.RESULTS_DIR / "neighbours_with_scores.csv", index=False
    )
    print(table.to_string(index=False))

    print("4/4 Extra evidence ...")
    extra = {
        "overlap_top200": compare.overlap_summary(cbow, skip),
        "agreement_selected_words": {
            w: round(compare.similarity_agreement(cbow, skip, w), 3)
            for w in config.SELECTED_WORDS
        },
        "neighbour_profile_top200": {
            "cbow": compare.neighbour_profile(cbow),
            "skipgram": compare.neighbour_profile(skip),
        },
        "wordsim353": {
            "cbow": compare.wordsim_evaluation(cbow),
            "skipgram": compare.wordsim_evaluation(skip),
        },
    }
    (config.RESULTS_DIR / "extra_evidence.json").write_text(json.dumps(extra, indent=2))
    print(json.dumps(extra, indent=2))
    print("Done. Files are in results/")


if __name__ == "__main__":
    main()
