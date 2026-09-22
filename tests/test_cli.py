def test_real_deps_segmenter_is_sam2_plus_spine_detector(monkeypatch):
    """Regression test: `libpipe process` (the actual command that would run against a real app upload)
    used to construct Deps with ReplicateSAM2 alone, silently skipping the trained spine detector that
    run_demo.py's --spine flag already used - so step 6 would fall back to the untrained
    split_by_spine_edges heuristic for every mask, book or not, on a real production run."""
    import libpipe.identity as identity_mod
    import libpipe.llm as llm_mod
    import libpipe.pieces as pieces_mod
    import libpipe.pricing as pricing_mod
    from libpipe import cli

    class Fake:
        def __init__(self, *a, **kw):
            pass

    for mod, name in [(pieces_mod, "ReplicateSAM2"), (pieces_mod, "SpineDetector"),
                      (llm_mod, "Fable"), (llm_mod, "Astra"), (llm_mod, "JevJudge"),
                      (identity_mod, "ISBNdb"), (pricing_mod, "Pricer"), (pricing_mod, "WebSearchPricer")]:
        monkeypatch.setattr(mod, name, Fake)

    deps = cli.real_deps()
    assert isinstance(deps.segmenter, list) and len(deps.segmenter) == 2
    assert isinstance(deps.segmenter[0], pieces_mod.ReplicateSAM2)
    assert isinstance(deps.segmenter[1], pieces_mod.SpineDetector)
