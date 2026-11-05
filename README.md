# Laboratory Sample Accessioning – Cylindrical Barcode & Computer Vision

## Project Overview

This project explores a computer-vision-based approach to improving barcode visibility, localization, and recognition on cylindrical laboratory sample containers such as test tubes and vials.

The system investigates the use of cylindrical image unwrapping, two-mirror reflection geometry, image calibration, and barcode detection to capture information that may otherwise be difficult to read from curved sample surfaces.

The long-term objective is to support laboratory sample accessioning by making barcode identification more reliable and reducing difficulties caused by cylindrical surfaces and limited camera visibility.

**Project status:** Research and prototype development.

---

## 1. Project Objectives

- Capture barcode labels on cylindrical laboratory containers.
- Flatten curved barcode-label images into a more readable representation.
- Investigate two-mirror arrangements to capture additional portions of cylindrical surfaces.
- Develop mathematical models for mirror reflections and camera geometry.
- Calibrate mirror positions and camera geometry using image-based measurements.
- Compare theoretical predictions against actual photographic measurements.
- Establish a foundation for integrating barcode detection and decoding.

---

## 2. Development Phases

### Phase 1 – Cylindrical Image Flattening

Developed an image-processing approach to unwrap labels attached to cylindrical surfaces.

Key techniques:
- Region of interest selection.
- Perspective correction.
- Cylindrical coordinate transformation.
- OpenCV image remapping.
- Image enhancement and contrast adjustment.

Important parameters include:
- Cylinder radius.
- Centre shift.
- Width scaling.
- Curvature adjustment.
- Angular coverage.

### Phase 2 – Mirror Image Processing and Flattening

Investigated combining the real central image with reflected images captured through two mirrors.

The objective was to invert and align mirror reflections and combine them with the central image to improve the visible barcode area.

### Phase 3 – Two-Mirror Calibration

Developed calibration scripts to identify mirror boundaries and convert image coordinates into approximate physical measurements.

Key techniques:
- Manual mirror-edge selection.
- Line intersection for mirror corner estimation.
- Perspective transformation using homography.
- Independent rectification of mirror surfaces.
- Millimetre-based coordinate measurement.

### Phase 4 – Two-Mirror Reflection Geometry

Developed a top-down 2D mathematical model to estimate the apparent positions of cylindrical objects reflected by two mirrors.

The model considers:
- Camera position.
- Camera-to-hinge distance.
- Mirror included angle.
- Gap between inner mirror edges.
- Flask position relative to the physical hinge.

The model produces theoretical apparent mirror-plane positions, which are compared with measurements from rectified photographs.

---

## 3. Physical Setup

Current recorded experimental configuration:

| Component | Specification |
|---|---|
| Mirrors | Approximately 150 × 100 mm each |
| Mirror included angle | 99° nominal |
| Inner mirror gap | 11 mm at top, 8 mm at bottom |
| Mean gap used in model | 9.5 mm |
| Flask outer diameter | Approximately 13 mm |
| Flask inner diameter | Approximately 11 mm |
| Cap diameter | Approximately 15 mm |
| Camera distance from hinge | 240 mm |
| Camera lateral offset | 0 mm assumed |
| Camera | 1× main camera |

Flask positions investigated:

- 30 mm
- 50 mm
- 70 mm
- 100 mm

Distances are referenced to the physical hinge point where the mirrors meet.

---

## 4. Phase 4 Validation Results

Four photographs were used to compare measured apparent reflection positions against theoretical predictions.

| Flask Position | Measured Left | Measured Right | Predicted |
|---|---:|---:|---:|
| 30 mm | 27.29 mm | 25.82 mm | 28.39 mm |
| 50 mm | 46.88 mm | 46.12 mm | 47.50 mm |
| 70 mm | 62.24 mm | 62.53 mm | 64.15 mm |
| 100 mm | 84.47 mm | 82.47 mm | 85.44 mm |

### Validation Summary

- Mean Absolute Error (MAE): approximately 1.64 mm.
- Maximum absolute error: approximately 2.97 mm.
- Total readings: 8.
- All measurements are within approximately 3 mm of the theoretical predictions.

These results represent preliminary agreement between the 2D model and image-based measurements.

Manual centreline selection, homography calibration, camera alignment, and physical geometry assumptions may contribute to the observed error.

The measurements represent apparent mirror-plane positions, not direct 3D flask coordinates.

---

## 5. Software and Technologies

- Python
- OpenCV
- NumPy
- Perspective transformation and homography
- Image remapping
- Geometric optics and reflection modelling
- Barcode detection and decoding tools

---

## 6. Repository Contents

The repository contains experimental implementations and supporting documentation from the development phases.

| File | Purpose |
|---|---|
| `test1phase2.py` | Cylindrical image flattening |
| `test2phase1.py` | Mirror-image inversion and flattening |
| `two_mirror_calibration_v3_1.py` | Earlier mirror calibration |
| `two_mirror_calibration_v4_1.py` | Improved mirror calibration |
| `reflection_model_2d.py` | Earlier reflection geometry model |
| `reflection_model_2d_hinge_reference.py` | Hinge-referenced 2D reflection model |
| `phase4_batch_measure.py` | Batch photographic measurement and validation |
| `New_Barcode_Detection_Approach_Report.pdf` | Earlier technical report |

Historical scripts are retained to preserve the development process. They should not automatically be treated as production-ready implementations.

---

## 7. Current Limitations

1. Centreline measurements currently require manual clicking.
2. The reflection model assumes idealized symmetry.
3. Camera lateral offset is assumed to be zero.
4. The model uses an average mirror gap and does not fully account for the vertical wedge.
5. Lens distortion and detailed 3D reflection effects are not yet modelled.
6. Homography calibration assumes the camera and mirror setup remain fixed.
7. Barcode recognition performance still requires further validation.
8. Measurement accuracy has not yet been established using traceable physical references.

---

## 8. Future Development

- Implement automatic flask-centreline detection using OpenCV.
- Quantify measurement variation caused by manual clicking.
- Improve camera and mirror calibration.
- Extend the 2D reflection model to 3D geometry.
- Account for mirror-gap variation with height.
- Improve reflected-image alignment and blending.
- Integrate geometric localization with barcode detection.
- Evaluate barcode reading accuracy under different lighting and viewing conditions.
- Develop a reproducible end-to-end sample accessioning prototype.

---

## 9. Project Handover and Reproducibility

For successful continuation of the project, maintain the following alongside the source code:

- Original experimental photographs.
- Top-view photographs of the mirror arrangement.
- Calibration parameters and homography matrices.
- Rectified mirror images.
- Measured-versus-predicted CSV and JSON files.
- Python environment and dependency versions.
- Documentation of changes to mathematical assumptions.

Do not overwrite earlier experimental versions without preserving them.

---

## 10. Important Notice

This repository contains research and experimental software intended to investigate computer vision for laboratory sample barcode localization and recognition.

It is not a validated laboratory accessioning or patient-identification system. Any future operational implementation must undergo appropriate technical validation, quality assurance, and human verification before being used in laboratory workflows.

---

**Project:** Laboratory Sample Accessioning – Cylindrical Barcode Computer Vision  
**Organization:** computer-vision-automation  
**Status:** Ongoing research and development