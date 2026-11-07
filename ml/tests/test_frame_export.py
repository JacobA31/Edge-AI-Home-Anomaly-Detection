"""Check real WAV decoding and the exact format consumed by the anomaly evaluator.

YAMNet inference is mocked here so these tests need neither TensorFlow nor a download.
"""

import csv
import wave
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest
from e3_edge.anomaly.frames import FrameStream, frame_level_dbfs

from src import export_frames, yamnet
from src.features import SAMPLE_RATE, prepare_audio


@pytest.fixture
def fake_yamnet(monkeypatch):
    def infer(waveform):
        count = yamnet.window_count(len(waveform))
        return (
            np.full((count, 1024), waveform.mean(), dtype=np.float32),
            np.full((count, 521), 0.25, dtype=np.float32),
        )

    monkeypatch.setattr(export_frames, "extract_features", infer)


def write_wav(path, rate=8000, samples=None):
    if samples is None:
        samples = np.tile([8192, 8192], (rate, 1))
    samples = np.asarray(samples).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setparams((samples.shape[1], 2, rate, 0, "NONE", "not compressed"))
        output.writeframes(samples.tobytes())


def write_manifest(path, rows):
    with path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["path", "start_time"])
        writer.writerows(rows)


def test_preparation_handles_microphone_layout_and_rejects_invalid_inputs():
    # zero-mean channels, so only the averaging and clipping act here
    samples = np.array([[0.25, 0.75], [-0.25, -0.75], [1.5, 1.5], [-1.5, -1.5]])
    result = prepare_audio(samples, SAMPLE_RATE)
    np.testing.assert_array_equal(result, [0.5, -0.5, 1.0, -1.0])
    assert result.dtype == np.float32
    assert samples[2, 0] == 1.5  # the input buffer stays available to its caller
    with pytest.raises(ValueError, match="16000"):
        prepare_audio(samples, 8000)
    with pytest.raises(ValueError, match="finite"):
        prepare_audio(np.array([np.nan]), SAMPLE_RATE)
    with pytest.raises(ValueError, match="floating point"):
        prepare_audio(np.array([32767], dtype=np.int16), SAMPLE_RATE)


def test_preparation_removes_each_channels_dc_offset_in_float64():
    # Like SINS: four channels at about half full scale, the room a -60 dBFS ripple on top,
    # a minute long. A float32 mean leaves an offset as loud as the ripple behind.
    n = 61 * SAMPLE_RATE
    ripple = 0.001 * np.sin(2 * np.pi * 440 * np.arange(n) / SAMPLE_RATE)
    offsets = np.array([0.498, 0.523, 0.508, 0.499])
    samples = (ripple[:, None] + offsets).astype(np.float32)
    result = prepare_audio(samples, SAMPLE_RATE)
    assert abs(float(result.mean())) < 1e-6
    rms_dbfs = 20 * np.log10(np.sqrt(np.mean(result.astype(np.float64) ** 2)))
    assert rms_dbfs == pytest.approx(20 * np.log10(0.001 / np.sqrt(2)), abs=0.1)
    # a mono buffer loses its offset too
    np.testing.assert_allclose(prepare_audio(np.full(100, 0.5), SAMPLE_RATE), 0.0, atol=1e-7)


def test_wav_loader_resamples_stereo_and_preserves_duration(tmp_path):
    path = tmp_path / "audio.wav"
    tone = 4096 * np.sin(2 * np.pi * 440 * np.arange(8000) / 8000)
    write_wav(path, samples=np.stack([tone + 8192, tone + 8192], axis=1))  # DC offset of 0.25
    result = export_frames.load_audio(path)
    assert result.dtype == np.float32 and result.shape == (16000,)
    assert abs(float(result.mean())) < 1e-4
    np.testing.assert_allclose(np.sqrt(np.mean(result[1000:-1000] ** 2)), 0.125 / np.sqrt(2), rtol=0.01)


@pytest.mark.parametrize(("length", "count"), [(1, 1), (15600, 1), (15601, 2), (48000, 6)])
def test_model_window_count_includes_padding(monkeypatch, length, count):
    embeddings = np.ones((count, 1024), dtype=np.float32)
    scores = np.ones((count, 521), dtype=np.float32)
    monkeypatch.setattr(yamnet, "_load_yamnet_model", lambda: lambda audio: (scores, embeddings, None))
    actual_embeddings, actual_scores = yamnet.extract_features(np.zeros(length, dtype=np.float32))
    np.testing.assert_array_equal(actual_embeddings, embeddings)
    np.testing.assert_array_equal(actual_scores, scores)


def test_model_checks_row_alignment(monkeypatch):
    monkeypatch.setattr(yamnet, "_load_yamnet_model", lambda: lambda audio: (
        np.zeros((1, 521)), np.zeros((2, 1024)), None,
    ))
    with pytest.raises(ValueError, match="shapes"):
        yamnet.extract_features(np.zeros(100, dtype=np.float32))


@pytest.mark.parametrize("explicit_ca", [False, True])
def test_certificate_fallback_runs_before_hub_import(monkeypatch, explicit_ca):
    monkeypatch.setenv("SSL_CERT_FILE", "configured-by-user.pem")
    if not explicit_ca:
        monkeypatch.delenv("SSL_CERT_FILE")
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    monkeypatch.setattr(
        yamnet.ssl, "get_default_verify_paths", lambda: SimpleNamespace(cafile=None, capath=None),
    )
    monkeypatch.setattr(yamnet.certifi, "where", lambda: "bundle.pem")

    def fake_import(name):
        assert name == "tensorflow_hub"
        expected = "configured-by-user.pem" if explicit_ca else "bundle.pem"
        assert yamnet.os.environ["SSL_CERT_FILE"] == expected
        return SimpleNamespace(load=lambda url: "loaded-model")

    monkeypatch.setattr(yamnet, "import_module", fake_import)
    yamnet._load_yamnet_model.cache_clear()
    try:
        assert yamnet._load_yamnet_model() == "loaded-model"
    finally:
        yamnet._load_yamnet_model.cache_clear()


def test_window_timestamps_and_loudness_use_matching_recorded_samples(fake_yamnet):
    waveform = np.linspace(0.1, 0.8, 48000, dtype=np.float32)
    stream = export_frames.make_stream(waveform, 1000.0)
    np.testing.assert_allclose(stream.timestamps, 1000 + np.arange(6) * 0.48)
    expected = [
        frame_level_dbfs(waveform[i * 7680:i * 7680 + 15600]) for i in range(6)
    ]
    np.testing.assert_allclose(stream.level_dbfs, expected)
    # Short recordings must not look quieter because of YAMNet's internal padding.
    short = export_frames.make_stream(np.full(100, 0.5, dtype=np.float32), 1000.0)
    np.testing.assert_allclose(short.level_dbfs, [20 * np.log10(0.5)])


def test_safe_cli_groups_labels_and_round_trips_with_clip_gaps(tmp_path, fake_yamnet):
    audio_dir = tmp_path / "safe"
    audio_dir.mkdir()
    for name in ("a-01.wav", "b-01.wav", "c-02.wav"):
        write_wav(audio_dir / name)
    output = tmp_path / "output"
    assert export_frames.main(["safe", str(audio_dir), "--output-dir", str(output)]) == 0
    falls = FrameStream.load(output / "falls.npz")
    normal = FrameStream.load(output / "non_falls.npz")
    assert falls.embeddings.shape == (4, 1024)
    assert normal.embeddings.shape == (2, 1024)
    assert falls.scores.shape == (4, 521)
    assert falls.level_dbfs.shape == (4,)
    np.testing.assert_array_equal(falls.score_classes, np.arange(521))
    np.testing.assert_allclose(falls.timestamps[[1, 3]] - falls.timestamps[[0, 2]], 0.48)
    assert falls.timestamps[2] - falls.timestamps[1] > 5.0
    assert falls.timestamps.dtype == np.float64
    assert falls.embeddings.dtype == falls.scores.dtype == falls.level_dbfs.dtype == np.float32


def test_safe_rejects_unlabeled_files(tmp_path):
    wav_path = tmp_path / "unknown.wav"
    wav_path.touch()
    with pytest.raises(ValueError, match="-01.wav or -02.wav"):
        export_frames.export_safe(wav_path, tmp_path / "out")


def test_sins_groups_by_local_day_and_preserves_utc_order(tmp_path, monkeypatch, fake_yamnet):
    for name in ("early.wav", "late.wav"):
        (tmp_path / name).touch()
    monkeypatch.setattr(export_frames, "load_audio", lambda path: np.full(48000, 0.5, dtype=np.float32))
    csv_path = tmp_path / "recordings.csv"
    write_manifest(csv_path, [
        ("late.wav", "2017-07-02T00:00:04+02:00"),
        ("early.wav", "2017-07-01T23:59:59.700+02:00"),
    ])
    output = tmp_path / "days"
    assert export_frames.main(["sins", str(csv_path), "--output-dir", str(output)]) == 0
    first = FrameStream.load(output / "2017-07-01.npz")
    second = FrameStream.load(output / "2017-07-02.npz")
    assert len(first) == 1 and len(second) == 11
    expected_start = datetime.fromisoformat("2017-07-01T23:59:59.700+02:00").timestamp()
    assert first.timestamps[0] == expected_start
    assert second.timestamps[0] == pytest.approx(expected_start + 0.48)
    assert np.all(np.diff(second.timestamps) > 0)
    assert second.scores.shape == (11, 521) and second.level_dbfs.shape == (11,)


def test_sins_requires_explicit_timezone_and_rejects_long_overlap(tmp_path, monkeypatch, fake_yamnet):
    for name in ("a.wav", "b.wav"):
        (tmp_path / name).touch()
    monkeypatch.setattr(export_frames, "load_audio", lambda path: np.full(48000, 0.5, dtype=np.float32))
    manifest = tmp_path / "recordings.csv"
    write_manifest(manifest, [("a.wav", "2017-07-01T08:00:00")])
    with pytest.raises(ValueError, match="timezone offset"):
        export_frames.export_sins(manifest, tmp_path / "out")
    write_manifest(manifest, [  # b repeats 2.5 s of a's 3 s: wrong start times, not a SINS overlap
        ("a.wav", "2017-07-01T08:00:00+02:00"),
        ("b.wav", "2017-07-01T08:00:00.500+02:00"),
    ])
    with pytest.raises(ValueError, match="Overlapping.*2.50 s"):
        export_frames.export_sins(manifest, tmp_path / "out")


def test_sins_trims_short_overlap_so_audio_is_counted_once(tmp_path, monkeypatch, fake_yamnet):
    for name in ("a.wav", "b.wav"):
        (tmp_path / name).touch()
    monkeypatch.setattr(export_frames, "load_audio", lambda path: np.full(48000, 0.5, dtype=np.float32))
    manifest = tmp_path / "recordings.csv"
    write_manifest(manifest, [  # b starts 0.92 s before a's 3 s end, like consecutive SINS files
        ("a.wav", "2017-07-01T08:00:00+02:00"),
        ("b.wav", "2017-07-01T08:00:02.080+02:00"),
    ])
    export_frames.export_sins(manifest, tmp_path / "out")
    stream = FrameStream.load(tmp_path / "out" / "2017-07-01.npz")
    a_start = datetime.fromisoformat("2017-07-01T08:00:00+02:00").timestamp()
    a_end = a_start + 3.0
    from_b = stream.timestamps[stream.timestamps >= a_end]
    assert len(stream) == 6 + 4 and len(from_b) == 4  # b keeps 2.08 s of audio: 4 windows
    assert from_b[0] == pytest.approx(a_end)  # b now starts where a ends
    assert np.all(np.diff(stream.timestamps) > 0)


def test_trim_overlap_cuts_only_the_repeated_samples():
    waveform = np.arange(48000, dtype=np.float32)
    start, trimmed = export_frames.trim_overlap(100.0, waveform, 100.92, "b.wav")
    assert start == pytest.approx(100.92) and len(trimmed) == 48000 - 14720
    assert trimmed[0] == 14720
    unchanged = export_frames.trim_overlap(101.0, waveform, 100.92, "b.wav")
    assert unchanged[0] == 101.0 and unchanged[1] is waveform


def test_failed_save_preserves_previous_output(tmp_path, monkeypatch, fake_yamnet):
    stream = export_frames.make_stream(np.full(100, 0.5, dtype=np.float32), 1000.0)
    output = tmp_path / "stream.npz"
    stream.save(output)
    previous = output.read_bytes()

    def interrupted_save(path):
        path.write_bytes(b"interrupted")
        raise OSError("disk error")

    monkeypatch.setattr(stream, "save", interrupted_save)
    with pytest.raises(OSError, match="disk error"):
        export_frames._save_stream(stream, output)
    assert output.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [output]
