# DustVision-KSA — Model Integration (Detection + Classification)

This part of the project **integrates the two separately-trained models** into a single
end-to-end pipeline for solar panel inspection.

```
photo ─▶ STAGE 1: YOLOv11 detects panel(s) ─▶ crop each panel
      ─▶ STAGE 2: ResNet50 classifies condition ─▶ annotated image + verdict
```

While my teammates trained the detection model (YOLOv11) and the classification model (ResNet50)
independently, **my contribution was the integration**: connecting the two models so that the
output of the detector feeds directly into the classifier, and the whole system works as one.

---

## What the integration does

For every input image, the pipeline runs the following steps:

1. **Detect** — YOLOv11 locates each solar panel and returns its bounding box.
2. **Crop / zoom** — each detected panel is cropped out of the image (with a small padding margin
   for context).
3. **Preprocess** — the crop is converted to RGB, resized to 224×224, and passed through the
   **ResNet50-specific preprocessing** required by the classifier.
4. **Classify** — ResNet50 predicts the panel's condition: one of
   `Clean · Dusty · Bird-drop · Physical-Damage · Electrical-damage`.
5. **Map to a verdict** — each condition is folded into a simple **Clean vs Defective** result.
6. **Annotate** — the original image is drawn with color-coded boxes and labels, and a structured
   report (per-panel condition, confidence, and verdict) is produced.

---

## Why integration is not trivial

The two models were trained on different data and have different requirements, so connecting them
correctly required care:

- **Matching the class order.** The classifier assigns label indices in a fixed order. The
  integration uses the exact same order, otherwise predictions would be silently mislabeled.
- **Matching the preprocessing.** ResNet50 needs its specific `preprocess_input` function (not a
  plain `/255` rescale). Using the wrong preprocessing does not throw an error — it quietly lowers
  accuracy — so this had to match the classifier's training exactly.
- **Handling the hand-off.** The detector outputs pixel-coordinate boxes; the classifier expects a
  fixed-size normalized image. The integration bridges this with the crop → resize → preprocess
  step.
- **Handling edge cases.** Degenerate or empty crops are skipped, and images where the detector
  finds no panel are handled gracefully.

---

## How to run

The integration lives in the notebook (after both models are loaded/trained).

```python
# load both trained models
DETECTOR   = YOLO("best_detector.pt")
CLASSIFIER = tf.keras.models.load_model("best_classifier.keras")

# run the full pipeline on one image
result = inspect_image("panel_photo.jpg")
```

`inspect_image()` returns a dictionary with the number of panels found, a per-panel report, and an
annotated image.

---

## Evaluating the integrated pipeline

The integration is evaluated **as a whole** (both models working together), not as two separate
models. Each test image is passed through the full detect → crop → classify pipeline, and the
predicted condition is compared against the ground-truth label.

The evaluation reports:
- **Integrated accuracy** and **macro-F1** across the 5 classes
- A **per-class precision / recall / F1** report
- A **confusion matrix**
- A **Clean vs Defective** binary summary

Results are saved to `outputs/integrated_eval_results.json` and
`outputs/integrated_confusion_matrix.png`.

---

## Classes

| Condition | Verdict |
|---|---|
| Clean | Clean |
| Dusty | Defective |
| Bird-drop | Defective |
| Physical-Damage | Defective |
| Electrical-damage | Defective |

---

## Notes

- **Inputs:** the system performs best on clear images of solar panels similar to its training
  data. Performance is reported on the held-out test split.
- **Models:** the trained model files (`best_detector.pt`, `best_classifier.keras`) are not stored
  in this repository due to size; they are produced by the training notebooks.

---

*Part of the ITS69204 — Computer Vision and Natural Language Processing group project.*
