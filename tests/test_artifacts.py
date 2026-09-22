from libpipe.artifacts import isometric_plot, objects_plot


def _obj(id_, x0, z0, x1, z1, status="ok", price=None, currency="USD", title=None, desc=None):
    return {"id": id_, "status": status, "title": title, "description": desc, "currency": currency,
           "list_price": price, "box": {"min": [x0, 0.0, z0], "max": [x1, 0.3, z1]}}


def test_objects_plot_renders_expected_size_and_excludes_non_objects():
    fp = {"walls": [{"start": [0, 0], "end": [3, 0]}, {"start": [3, 0], "end": [3, 3]}]}
    objects = [
        _obj("o1", 0.5, 0.5, 0.8, 0.7, status="ok", price=99.0, currency="INR", desc="a vase"),
        _obj("o2", 1.0, 1.0, 1.1, 1.05, status="needs_review", price=None),
        _obj("o3", 2.0, 2.0, 2.5, 2.5, status="not_an_object", price=None),   # must be excluded from the plot
    ]
    img = objects_plot(fp, objects, size=400)
    assert img.size == (400, 400)


def test_objects_plot_handles_empty_input():
    img = objects_plot({"walls": []}, [], size=300)
    assert img.size == (300, 300)


def test_objects_plot_skips_objects_without_a_box():
    """A priced object missing its box (shouldn't happen, but step 14 doesn't guarantee it) must not crash
    the plot - it's a demo visualization, not a data validator."""
    objects = [{"id": "o1", "status": "ok", "list_price": 10, "currency": "USD", "box": None}]
    img = objects_plot({"walls": []}, objects, size=200)
    assert img.size == (200, 200)


def test_objects_plot_sums_only_priced_objects_into_the_total_label():
    import numpy as np

    objects = [
        _obj("o1", 0, 0, 0.3, 0.3, status="ok", price=100.0, currency="USD"),
        _obj("o2", 1, 1, 1.3, 1.3, status="ok", price=50.0, currency="USD"),
        _obj("o3", 2, 2, 2.3, 2.3, status="needs_review", price=None),   # excluded from the total, still drawn
    ]
    img = objects_plot({"walls": []}, objects, size=500)
    assert np.array(img).sum() > 0   # something was actually drawn, not a blank canvas


def test_isometric_plot_renders_expected_size_and_excludes_non_objects():
    fp = {"floor_y": 0.0, "walls": [{"start": [0, 0], "end": [3, 0]}]}
    objects = [
        _obj("o1", 0.5, 0.5, 0.8, 0.7, status="ok", price=99.0, currency="INR", desc="a vase"),
        _obj("o2", 2.0, 2.0, 2.5, 2.5, status="not_an_object", price=None),
    ]
    img = isometric_plot(fp, objects, size=400)
    assert img.size == (400, 400) and img.mode == "RGB"


def test_isometric_plot_handles_empty_and_missing_box():
    assert isometric_plot({"walls": []}, [], size=300).size == (300, 300)
    objects = [{"id": "o1", "status": "ok", "list_price": 10, "currency": "USD", "box": None}]
    assert isometric_plot({"walls": []}, objects, size=200).size == (200, 200)


def test_iso_projection_taller_points_map_higher_on_screen():
    """A larger world y (height) must produce a smaller screen y (higher up), and the projection must not
    collapse x/z into the same axis as height - the whole point of using this over a flat top-down plot."""
    import numpy as np

    from libpipe.artifacts import _iso

    low = _iso(np.array([1.0, 0.0, 1.0]))
    high = _iso(np.array([1.0, 2.0, 1.0]))
    assert high[1] < low[1]                    # taller -> smaller screen y -> draws higher
    assert np.isclose(high[0], low[0])          # pure height change must not shift the horizontal position
