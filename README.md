# 👁️ AEGIS

[Documentation](https://siddhesh-thakur.github.io/aegis/) | [License](https://github.com/siddhesh-thakur/aegis?tab=License-1-ov-file)

AEGIS is a toolkit for large-scale whole-slide image processing.
This project was developed by Siddhesh Thakur.

> [!NOTE]
> Contributions are welcome! Please report any issues. You may also contribute by opening a pull request.

### Key Features:

<img align="right" src="_readme/aegis_crop.jpg" width="250px" />

- **Tissue Segmentation**: Extract tissue from background (H&E, IHC, etc.).
- **Patch Extraction**: Extract tissue patches of any size and magnification.
- **Patch Feature Extraction**: Extract patch embeddings from 20+ foundation models, including [UNI](https://www.nature.com/articles/s41591-024-02857-3), [Virchow](https://www.nature.com/articles/s41591-024-03141-0), [H-Optimus-0](https://github.com/bioptimus/releases/tree/main/models/h-optimus/v0) and more...
- **Slide Feature Extraction**: Extract slide embeddings from 5+ slide foundation models, including [Threads](https://arxiv.org/abs/2501.16652) (coming soon!), [Titan](https://arxiv.org/abs/2411.19666), and [GigaPath](https://www.nature.com/articles/s41586-024-07441-w).

### 🔨 1. **Installation**:

#### Local Installation
- Create an environment: `conda create -n "aegis" python=3.10`, and activate it `conda activate aegis`.
- Cloning: `git clone https://github.com/siddhesh-thakur/aegis.git && cd aegis`.
- Local installation: `pip install -e .`.

Additional packages may be required to load some pretrained models. Follow error messages for instructions.

#### Docker Installation
- Build the Docker image: `docker build -t aegis .`
- Run the Docker container: `docker run -it --gpus all -v /path/to/your/data:/data aegis /bin/bash`
This will mount your local data directory into the container at `/data`.

### 🔨 2. **Running AEGIS**:

**Already familiar with WSI processing?** Perform segmentation, patching, and UNI feature extraction from a directory of WSIs with:

```
aegis --task all --wsi_dir ./wsis --job_dir ./aegis_processed --patch_encoder uni_v1 --mag 20 --patch_size 256
```

**Feeling cautious?**

Run this command to perform all processing steps for a **single** slide:
```
aegis --slide_path ./wsis/xxxx.svs --job_dir ./aegis_processed --patch_encoder uni_v1 --mag 20 --patch_size 256
```

**Or follow step-by-step instructions:**

**Step 1: Tissue Segmentation:** Segments tissue vs. background from a dir of WSIs
 - **Command**:
   ```bash
   aegis --task seg --wsi_dir ./wsis --job_dir ./aegis_processed --gpu 0 --segmenter hest
   ```
   - `--task seg`: Specifies that you want to do tissue segmentation.
   - `--wsi_dir ./wsis`: Path to dir with your WSIs.
   - `--job_dir ./aegis_processed`: Output dir for processed results.
   - `--gpu 0`: Uses GPU with index 0.
   - `--segmenter`: Segmentation model. Defaults to `hest`. Switch to `grandqc` for fast H&E segmentation. Add the option `--remove_artifacts` for additional artifact clean up.
 - **Outputs**:
   - WSI thumbnails in `./aegis_processed/thumbnails`.
   - WSI thumbnails with tissue contours in `./aegis_processed/contours`.
   - GeoJSON files containing tissue contours in `./aegis_processed/contours_geojson`. These can be opened in [QuPath](https://qupath.github.io/) for editing/quality control, if necessary.

 **Step 2: Tissue Patching:** Extracts patches from segmented tissue regions at a specific magnification.
 - **Command**:
   ```bash
   aegis --task coords --wsi_dir ./wsis --job_dir ./aegis_processed --mag 20 --patch_size 256 --overlap 0
   ```
   - `--task coords`: Specifies that you want to do patching.
   - `--wsi_dir wsis`: Path to the dir with your WSIs.
   - `--job_dir ./aegis_processed`: Output dir for processed results.
   - `--mag 20`: Extracts patches at 20x magnification.
   - `--patch_size 256`: Each patch is 256x256 pixels.
   - `--overlap 0`: Patches overlap by 0 pixels (**always** an absolute number in pixels, e.g., `--overlap 128` for 50% overlap for 256x256 patches.
 - **Outputs**:
   - Patch coordinates as h5 files in `./aegis_processed/20x_256px/patches`.
   - WSI thumbnails annotated with patch borders in `./aegis_processed/20x_256px/visualization`.

 **Step 3a: Patch Feature Extraction:** Extracts features from tissue patches using a specified encoder
 - **Command**:
   ```bash
   aegis --task feat --wsi_dir ./wsis --job_dir ./aegis_processed --patch_encoder uni_v1 --mag 20 --patch_size 256
   ```
   - `--task feat`: Specifies that you want to do feature extraction.
   - `--wsi_dir wsis`: Path to the dir with your WSIs.
   - `--job_dir ./aegis_processed`: Output dir for processed results.
   - `--patch_encoder uni_v1`: Uses the `UNI` patch encoder. See below for list of supported models.
   - `--mag 20`: Features are extracted from patches at 20x magnification.
   - `--patch_size 256`: Patches are 256x256 pixels in size.
 - **Outputs**:
   - Features are saved as h5 files in `./aegis_processed/20x_256px/features_uni_v1`. (Shape: `(n_patches, feature_dim)`)

AEGIS supports 21 patch encoders, loaded via a patch [`encoder_factory`](https://github.com/siddhesh-thakur/aegis/blob/main/aegis/patch_encoder_models/load.py#L14). Models requiring specific installations will return error messages with additional instructions. Gated models on HuggingFace require access requests.

| Patch Encoder         | Embedding Dim | Args                                                             | Link |
|-----------------------|---------------:|------------------------------------------------------------------|------|
| **UNI**               | 1024           | `--patch_encoder uni_v1 --patch_size 256 --mag 20`               | [MahmoodLab/UNI](https://huggingface.co/MahmoodLab/UNI) |
| **UNI2-h**             | 1536           | `--patch_encoder uni_v2 --patch_size 256 --mag 20`               | [MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) |
| **CONCH**             | 512            | `--patch_encoder conch_v1 --patch_size 512 --mag 20`             | [MahmoodLab/CONCH](https://huggingface.co/MahmoodLab/CONCH) |
| **CONCHv1.5**         | 768            | `--patch_encoder conch_v15 --patch_size 512 --mag 20`            | [MahmoodLab/conchv1_5](https://huggingface.co/MahmoodLab/conchv1_5) |
| **Virchow**           | 2560           | `--patch_encoder virchow --patch_size 224 --mag 20`              | [paige-ai/Virchow](https://huggingface.co/paige-ai/Virchow) |
| **Virchow2**          | 2560           | `--patch_encoder virchow2 --patch_size 224 --mag 20`             | [paige-ai/Virchow2](https://huggingface.co/paige-ai/Virchow2) |
| **Phikon**            | 768            | `--patch_encoder phikon --patch_size 224 --mag 20`               | [owkin/phikon](https://huggingface.co/owkin/phikon) |
| **Phikon-v2**         | 1024           | `--patch_encoder phikon_v2 --patch_size 224 --mag 20`            | [owkin/phikon-v2](https://huggingface.co/owkin/phikon-v2/) |
| **Prov-Gigapath**     | 1536           | `--patch_encoder gigapath --patch_size 256 --mag 20`             | [prov-gigapath](https://huggingface.co/prov-gigapath/prov-gigapath) |
| **H-Optimus-0**       | 1536           | `--patch_encoder hoptimus0 --patch_size 224 --mag 20`            | [bioptimus/H-optimus-0](https://huggingface.co/bioptimus/H-optimus-0) |
| **H-Optimus-1**       | 1536           | `--patch_encoder hoptimus1 --patch_size 224 --mag 20`            | [bioptimus/H-optimus-1](https://huggingface.co/bioptimus/H-optimus-1) |
| **MUSK**              | 1024           | `--patch_encoder musk --patch_size 384 --mag 20`                 | [xiangjx/musk](https://huggingface.co/xiangjx/musk) |
| **Midnight-12k**      | 3072           | `--patch_encoder midnight12k --patch_size 224 --mag 20`          | [kaiko-ai/midnight](https://huggingface.co/kaiko-ai/midnight) |
| **Kaiko**             | 384/768/1024   | `--patch_encoder {kaiko-vits8, kaiko-vits16, kaiko-vitb8, kaiko-vitb16, kaiko-vitl14} --patch_size 256 --mag 20` | [1aurent/kaikoai-models-66636c99d8e1e34bc6dcf795](https://huggingface.co/collections/1aurent/kaikoai-models-66636c99d8e1e34bc6dcf795) |
| **Lunit**             | 384            | `--patch_encoder lunit-vits8 --patch_size 224 --mag 20`          | [1aurent/vit_small_patch8_224.lunit_dino](https://huggingface.co/1aurent/vit_small_patch8_224.lunit_dino) |
| **Hibou**             | 1024           | `--patch_encoder hibou_l --patch_size 224 --mag 20`              | [histai/hibou-L](https://huggingface.co/histai/hibou-L) |
| **CTransPath-CHIEF**  | 768            | `--patch_encoder ctranspath --patch_size 256 --mag 10`           | — |
| **ResNet50**          | 1024           | `--patch_encoder resnet50 --patch_size 256 --mag 20`             | — |

**Step 3b: Slide Feature Extraction:** Extracts slide embeddings using a slide encoder. Will also automatically extract the right patch embeddings.
 - **Command**:
   ```bash
   aegis --task feat --wsi_dir ./wsis --job_dir ./aegis_processed --slide_encoder titan --mag 20 --patch_size 512
   ```
   - `--task feat`: Specifies that you want to do feature extraction.
   - `--wsi_dir wsis`: Path to the dir containing WSIs.
   - `--job_dir ./aegis_processed`: Output dir for processed results.
   - `--slide_encoder titan`: Uses the `Titan` slide encoder. See below for supported models.
   - `--mag 20`: Features are extracted from patches at 20x magnification.
   - `--patch_size 512`: Patches are 512x512 pixels in size.
 - **Outputs**:
   - Features are saved as h5 files in `./aegis_processed/20x_256px/slide_features_titan`. (Shape: `(feature_dim)`)

AEGIS supports 5 slide encoders, loaded via a slide-level [`encoder_factory`](https://github.com/siddhesh-thakur/aegis/blob/main/aegis/slide_encoder_models/load.py#L14). Models requiring specific installations will return error messages with additional instructions. Gated models on HuggingFace require access requests.

| Slide Encoder | Patch Encoder | Args | Link |
|---------------|----------------|------|------|
| **Threads** | conch_v15 | `--slide_encoder threads --patch_size 512 --mag 20` | *(Coming Soon!)* |
| **Titan** | conch_v15 | `--slide_encoder titan --patch_size 512 --mag 20` | [MahmoodLab/TITAN](https://huggingface.co/MahmoodLab/TITAN) |
| **PRISM** | virchow | `--slide_encoder prism --patch_size 224 --mag 20` | [paige-ai/Prism](https://huggingface.co/paige-ai/Prism) |
| **CHIEF** | ctranspath | `--slide_encoder chief --patch_size 256 --mag 10` | [CHIEF](https://github.com/hms-dbmi/CHIEF) |
| **GigaPath** | gigapath | `--slide_encoder gigapath --patch_size 256 --mag 20` | [prov-gigapath](https://huggingface.co/prov-gigapath/prov-gigapath) |
| **Madeleine** | conch_v1 | `--slide_encoder madeleine --patch_size 256 --mag 10` | [MahmoodLab/madeleine](https://huggingface.co/MahmoodLab/madeleine) |
| **Feather** | conch_v15 | `--slide_encoder feather --patch_size 512 --mag 20` | [MahmoodLab/FEATHER](https://huggingface.co/MahmoodLab/abmil.base.conch_v15.pc108-24k) |

> [!NOTE]
> If your task includes multiple slides per patient, you can generate patient-level embeddings by: (1) processing each slide independently and taking their average slide embedding (late fusion) or (2) pooling all patches together and processing that as a single "pseudo-slide" (early fusion).

Please see our [tutorials](https://github.com/siddhesh-thakur/aegis/tree/main/tutorials) for more support.

### 🙋 FAQ
- **Q**: How do I extract patch embeddings from legacy patch coordinates extracted with [CLAM](https://github.com/mahmoodlab/CLAM)?
   - **A**:
      ```bash
      aegis --task feat --wsi_dir ..wsis --job_dir legacy_dir --patch_encoder uni_v1 --mag 20 --patch_size 256 --coords_dir extracted_mag20x_patch256_fp/
      ```
- **Q**: How do I keep patches corresponding to holes in the tissue?
   - **A**: In `aegis`, this behavior is default. Set `--remove_holes` to exclude patches on top of holes.

- **Q**: I see weird messages when building models using timm. What is happening?
   - **A**: Make sure `timm==0.9.16` is installed. `timm==1.X.X` creates issues with most models.

- **Q**: How can I use `aegis` in other repos with minimal work?
  - **A**: Make sure `aegis` is installed using `pip install -e .`. Then, the `aegis` command will be available in your path.

- **Q**: I am not satisfied with the tissue vs background segmentation. What can I do?
   - **A**: AEGIS uses GeoJSON to store and load segmentations. This format is natively supported by [QuPath](https://qupath.github.io/). You can load the AEGIS segmentation into QuPath, modify it using QuPath's annotation tools, and save the updated segmentation back to GeoJSON.
   - **A**: You can try another segmentation model by specifying `segmenter --grandqc`.

- **Q**: I want to process a custom list of WSIs. Can I do it? Also, most of my WSIs don't have the micron per pixel (mpp) stored. Can I pass it?
   - **A**: Yes using the `--custom_list_of_wsis` argument. Provide a list of WSI names in a CSV (with slide extension, `wsi`). Optionally, provide the mpp (field `mpp`)

 - **Q**: Do I need to install any additional packages to use AEGIS?
   - **A**: Most pretrained models require additional dependencies (e.g., the CTransPath patch encoder requires `pip install timm_ctp`). When you load a model using AEGIS, it will tell you what dependencies are missing and how to install them.

## License and Terms of Use

ⓒ Siddhesh Thakur. This repository is released under the [CC-BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/deed.en) license and may only be used for non-commercial, academic research purposes with proper attribution. Any commercial use, sale, or other monetization of this repository is prohibited and requires prior approval. By downloading any pretrained encoder, you agree to follow the model's respective license.

## Acknowledgements

The project was built on top of amazing repositories such as [Timm](https://github.com/huggingface/pytorch-image-models/), [HuggingFace](https://huggingface.co/docs/datasets/en/index), and open-source contributions from the community. We thank the authors and developers for their contribution.

## Issues

- The preferred mode of communication is via GitHub issues.
- If GitHub issues are inappropriate, email Siddhesh Thakur.
- Immediate response to minor issues may not be available.

## How to cite

If you find our work useful in your research or if you use parts of this code, please consider citing our work.

(Citation placeholder)
