# Plants_Tomato_2025

This repository contains tomato images, polygon annotations for the `locule` and `pedicel` classes, preprocessing scripts, and example images for paper presentation.

## Repository contents

```text
.
|-- README.md
|-- locule/          # Images and polygon annotations for the locule class
|-- pedicel/         # Images and polygon annotations for the pedicel class
|-- pre/             # Preprocessing and visualization scripts
`-- paper_examples/  # Example images for paper presentation
```

| Directory | Description |
| --- | --- |
| [locule/](locule/) | JPG images and corresponding JSON polygon annotations for the `locule` class |
| [pedicel/](pedicel/) | JPG images and corresponding JSON polygon annotations for the `pedicel` class |
| [pre/](pre/) | Scripts for image extraction, resizing, renaming, annotation maintenance, augmentation, and visualization |
| [paper_examples/](paper_examples/) | Example images for paper presentation |

## Original-image naming convention

Original images using the date-and-batch naming convention follow this format:

```text
<month>_<day>_image_<batch>_<view>.<extension>
```

For example, `7_28_image_1_2.png` means:

| Component | Meaning |
| --- | --- |
| `7_28` | Acquisition date: July 28 |
| `image` | Image identifier in this naming convention |
| `1` | Tomato batch 1 |
| `2` | Capture/view index within that batch; images of the same batch were captured at different positions or heights |

The batch identifier links images of the same tomato batch. The view index identifies a capture within that batch, and the date prefix records the month and day. Preserve these components when organizing images to retain their acquisition context.

## Annotated images and augmentation

Images and their JSON annotation files share the same filename stem. For example:

```text
locule/3_4_image_1_3_0.jpg
locule/3_4_image_1_3_0.json
```

The JSON files contain `shapes` entries with a `label`, polygon `points`, and `shape_type: "polygon"`, together with `imagePath`, `imageWidth`, and `imageHeight`. Polygon vertices use image-space `(x, y)` coordinates. The annotated images use a resolution of **512 × 512 pixels**.

| Directory | Annotation label |
| --- | --- |
| `locule/` | `locule` |
| `pedicel/` | `pedicel` |

### Augmentation suffixes

| Suffix | Processing operation |
| --- | --- |
| `_resized` | Scale the image and polygon coordinates, then center the image on a black 512 × 512 canvas; the script uses a scale factor of 0.5 |
| `_rotated_90`, `_rotated_180`, `_rotated_270` | Rotate the image and transform polygon coordinates by the indicated angle in degrees |
| `_brighter` | Multiply pixel intensities by 1.5, clipping values to the valid range |
| `_darker` | Multiply pixel intensities by 0.7, clipping values to the valid range |

Suffixes can be combined, as in `_resized_rotated_90_brighter`, to describe successive processing operations.

## Paper example images

The [paper_examples/](paper_examples/) directory contains images for paper presentation. Files named `tomato_src_<id>` provide consistent identifiers for referencing these examples. Original filenames are also retained for traceability.

## Preprocessing scripts

The scripts in [pre/](pre/) cover image preprocessing, annotation handling, augmentation, and visualization. Configure the paths and parameters in the selected script for your local environment before running it.

| Script | Purpose |
| --- | --- |
| [1_ex_mask.py](pre/1_ex_mask.py) | Use HSV thresholds and contour operations to extract tomato regions and place them on a black background |
| [2_2_3_cut.py](pre/2_2_3_cut.py) | Scale images, center them on a 512 × 512 black canvas, and update polygon coordinates |
| [3_rename.py](pre/3_rename.py) | Rename files using an example rule that replaces `color` with `7_28` for filenames starting with `color` |
| [4_image_path_update.py](pre/4_image_path_update.py) | Update JSON `imagePath` to the same-stem JPG filename and refresh embedded `imageData` when the image is found |
| [5_label_info_verify.py](pre/5_label_info_verify.py) | Overlay polygons on an image for visual inspection |
| [6_rotation_data_90_180_270.py](pre/6_rotation_data_90_180_270.py) | Generate rotated JPG images and corresponding transformed polygon annotations |
| [7_brightness.py](pre/7_brightness.py) | Generate brighter and darker images and update their JSON image references and embedded data |
| [verify.py](pre/verify.py) | Display an image alongside its segmentation-label image for visual inspection |

### Dependencies and local use

Install the packages used by the scripts:

```bash
python -m pip install numpy opencv-python matplotlib Pillow
```

Set the input, output, and example file paths in the selected script, review its processing parameters, and run the script individually. Several scripts execute their processing code directly when run or imported.

Use source-data copies for operations that rename files or rewrite JSON. When renaming image and annotation files, keep the JSON `imagePath` references consistent. Use the visualization scripts to inspect images and their annotations.
