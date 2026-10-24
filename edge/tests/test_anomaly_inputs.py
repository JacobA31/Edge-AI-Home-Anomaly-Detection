"""The module's inputs: YAMNet class names, category mapping and frame-stream files."""

import numpy as np
import pytest

from e3_edge.anomaly.categories import DEFAULT_CATEGORIES, CategoryMapper, class_names, resolve
from e3_edge.anomaly.config import AnomalyConfig, CategoryOverride
from e3_edge.anomaly.frames import FrameStream, frame_level_dbfs, iter_streams, observed_seconds


def test_class_map_is_yamnets_521_classes():
    names = class_names()
    assert len(names) == 521
    assert names[0] == "Speech" and names[494] == "Silence" and names[364] == "Water tap, faucet"


def test_every_default_category_uses_real_yamnet_classes():
    known = set(class_names())
    for name, spec in DEFAULT_CATEGORIES.items():
        assert set(spec.classes) <= known, name
    CategoryMapper(DEFAULT_CATEGORIES)  # also validates thresholds


def test_category_score_is_the_strongest_member_class():
    mapper = CategoryMapper(DEFAULT_CATEGORIES)
    scores = np.zeros(521, dtype=np.float32)
    index = {n: i for i, n in enumerate(class_names())}
    scores[index["Sink (filling or washing)"]] = 0.7
    scores[index["Water"]] = 0.4
    scores[index["Thump, thud"]] = 0.15  # below the impact "on" threshold
    by_name = dict(zip(mapper.names, mapper.scores(scores), strict=True))
    assert by_name["water"] == pytest.approx(0.7)
    assert by_name["impact"] == pytest.approx(0.15)
    assert mapper.dominant(scores) == "water"
    assert mapper.dominant(np.zeros(521)) is None
    np.testing.assert_allclose(mapper.scores_batch(scores[None, :])[0], mapper.scores(scores))


def test_category_overrides():
    specs = resolve({"water": CategoryOverride(on=0.5), "kettle": CategoryOverride(classes=["Boiling"])})
    assert specs["water"].on == 0.5 and specs["water"].classes == DEFAULT_CATEGORIES["water"].classes
    assert specs["kettle"].classes == ("Boiling",)
    with pytest.raises(ValueError, match="needs a list of classes"):
        resolve({"new": CategoryOverride(on=0.3)})
    with pytest.raises(ValueError, match="not YAMNet class names"):
        CategoryMapper(resolve({"x": CategoryOverride(classes=["Elephant"])}))


def test_config_from_yaml_shaped_dict():
    cfg = AnomalyConfig.from_dict({"timezone": "America/Toronto", "water": {"min_seconds": 60},
                                   "categories": {"water": {"on": 0.3}}})
    assert cfg.water.min_seconds == 60 and cfg.water.margin == 1.5
    assert cfg.categories["water"].on == 0.3
    assert AnomalyConfig.from_dict(cfg.to_dict()) == cfg
    with pytest.raises(ValueError, match="unknown anomaly setting"):
        AnomalyConfig.from_dict({"water": {"typo": 1}})
    with pytest.raises(ValueError, match="contexts"):
        AnomalyConfig.from_dict({"embedding": {"contexts": "nope"}})
    with pytest.raises(ValueError, match="periods"):
        AnomalyConfig.from_dict({"periods": {"day": 6}})


def test_example_config_has_a_valid_anomaly_section():
    from pathlib import Path

    from e3_edge import config

    cfg = config.load(Path(__file__).resolve().parents[1] / "config.example.yaml")
    anomaly = AnomalyConfig.from_dict(cfg.anomaly)
    assert anomaly.timezone == "America/Toronto"
    assert anomaly.calibration.stats_seconds == 3 * 86400 and anomaly.embedding.contexts == "category"


def test_level_dbfs():
    t = np.arange(15360) / 16000
    assert frame_level_dbfs(np.sin(2 * np.pi * 440 * t)) == pytest.approx(-3.01, abs=0.01)
    assert frame_level_dbfs(np.zeros(100)) == -120.0


def stream(n=10, start=0.0, hop=0.48, dim=4, subset=True):
    t = start + np.arange(n) * hop
    rng = np.random.default_rng(0)
    if subset:
        return FrameStream(t, rng.random((n, dim)), rng.random((n, 2)), [364, 494], rng.random(n) - 60)
    return FrameStream(t, rng.random((n, dim)))


def test_stream_round_trip_with_a_subset_of_classes(tmp_path):
    s = stream()
    s.save(tmp_path / "s.npz")
    loaded = FrameStream.load(tmp_path / "s.npz")
    frame = loaded.frame(3)
    assert frame.scores.shape == (521,)
    assert frame.scores[364] == pytest.approx(s.scores[3, 0]) and frame.scores[0] == 0
    assert frame.level_dbfs == pytest.approx(float(s.level_dbfs[3]))
    np.testing.assert_array_equal(loaded.embeddings, s.embeddings)
    assert stream(subset=False).frame(0).scores is None


def test_stream_validation():
    with pytest.raises(ValueError, match="increasing"):
        FrameStream([0.0, 1.0, 1.0], np.zeros((3, 2)))
    with pytest.raises(ValueError, match="one row per timestamp"):
        FrameStream([0.0, 1.0], np.zeros((3, 2)))
    with pytest.raises(ValueError, match="521"):
        FrameStream([0.0], np.zeros((1, 2)), np.zeros((1, 5)))
    with pytest.raises(ValueError, match="class indices"):
        FrameStream([0.0], np.zeros((1, 2)), np.zeros((1, 1)), [600])


def test_streams_are_replayed_in_time_order_and_must_not_overlap():
    later, earlier = stream(start=100.0), stream(start=0.0)
    times = [f.t for f in iter_streams([later, earlier])]
    assert times == sorted(times) and len(times) == 20
    with pytest.raises(ValueError, match="overlap"):
        list(iter_streams([stream(start=0.0), stream(start=1.0)]))


def test_observed_seconds_skips_gaps():
    t = np.concatenate([np.arange(10) * 0.5, 100 + np.arange(10) * 0.5])
    # two runs of 4.5 s of steps, each gap counted as one frame, plus the last frame
    assert observed_seconds(t, 0.96, 5.0) == pytest.approx(4.5 + 0.96 + 4.5 + 0.96)
