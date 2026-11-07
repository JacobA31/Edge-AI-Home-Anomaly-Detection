# models/ (contents git-ignored)

Trained models are written here. Publish the ones the Pi should use as GitHub Release assets.

## Pretrained YAMNet export

From `ml/`, with its Python environment activated and requirements installed:

```bash
python -m src.export_yamnet
```

This downloads the pretrained Hub YAMNet if needed, converts it and writes to `models/yamnet.tflite`,
and verifies its outputs with LiteRT before saving. It does not train YAMNet.
An existing output file is replaced only after conversion and verification succeed.
Use `--output models/another-name.tflite` to choose a different destination.

The exported model has a `serving_default` signature:

- Input `waveform`: float32, shape `(15600,)`, mono 16 kHz audio in `[-1, 1]`.
- Output `embeddings`: float32, shape `(1, 1024)`.
- Output `scores`: float32, shape `(1, 521)`, in the original AudioSet class order.

One call covers 0.975 seconds of waveform for one 0.96-second spectrogram patch.
The caller supplies successive windows 0.48 seconds (7,680 samples) apart.
The model computes the spectrogram internally. It uses LiteRT builtin operations;
the Pi does not need full TensorFlow to execute it.

The exporter compares silence, a 440 Hz tone, and repeatable noise against the original
Hub model. Both outputs must be finite float32 arrays with the expected shapes and agree
within `rtol=1e-3`, `atol=1e-4`. It prints `Verified` for each signal and then
`Saved verified model to ...` on success. A failed check exits without replacing the model.

Run the download-free export tests from `ml/`:

```bash
python -m pytest tests/test_yamnet_export.py -q
```

The tests use simulated outputs and a small locally created model, not downloaded
YAMNet weights. The real conversion test is skipped if TensorFlow or LiteRT is absent.
Running the export command above performs the check with the actual pretrained YAMNet.

The generated `.tflite` file is ignored by Git. Publishing it to a GitHub Release and
downloading it onto the Pi are separate deployment steps.
