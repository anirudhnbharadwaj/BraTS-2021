import os
import json
import time
from glob import glob
import shutil
import nibabel as nib
import numpy as np
import logging
from multiprocessing import Pool, cpu_count
from monai.transforms import (
    Compose, LoadImaged, ScaleIntensityRanged, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d, ToTensord, EnsureChannelFirstd,
    RandAffined, RandAdjustContrastd, NormalizeIntensityd, SpatialPadd
)

def load_nifty(directory, example_id, suffix):
    """Load a NIfTI file."""
    return nib.load(os.path.join(directory, f"{example_id}_{suffix}.nii.gz"))

def load_channels(directory, example_id):
    """Load all modality channels for a given example."""
    return [load_nifty(directory, example_id, suffix) for suffix in ["flair", "t1", "t1ce", "t2"]]

def get_data(nifty, dtype="int16"):
    """Extract data from a NIfTI object with specified dtype."""
    if dtype == "int16":
        data = np.abs(nifty.get_fdata().astype(np.int16))
        data[data == -32768] = 0
        return data
    return nifty.get_fdata().astype(np.uint8)

def prepare_nifty(d):
    """Preprocess a single BraTS sample: combine modalities into a 4-channel image and adjust segmentation labels."""
    example_id = d.split(os.sep)[-1]
    flair, t1, t1ce, t2 = load_channels(d, example_id)
    affine, header = flair.affine, flair.header
    vol = np.stack([get_data(flair), get_data(t1), get_data(t1ce), get_data(t2)], axis=-1)
    vol = nib.Nifti1Image(vol, affine, header=header)
    nib.save(vol, os.path.join(d, f"{example_id}.nii.gz"))

    seg_path = os.path.join(d, f"{example_id}_seg.nii.gz")
    if os.path.exists(seg_path):
        seg = load_nifty(d, example_id, "seg")
        original_labels = np.unique(seg.get_fdata())
        logging.info(f"Sample {example_id} - Original Segmentation Labels: {original_labels}")

        affine, header = seg.affine, seg.header
        vol = get_data(seg, "uint8")
        vol[vol == 4] = 3  # Map label 4 to 3
        seg = nib.Nifti1Image(vol, affine, header=header)
        nib.save(seg, seg_path)

        updated_labels = np.unique(vol)
        logging.info(f"Sample {example_id} - Updated Segmentation Labels: {updated_labels}")

def prepare_dirs(data, train):
    """Organize the processed files into images and labels directories."""
    img_path = os.path.join(data, "images")
    lbl_path = os.path.join(data, "labels")
    os.makedirs(img_path, exist_ok=True)
    if train:
        os.makedirs(lbl_path, exist_ok=True)

    dirs = glob(os.path.join(data, "BraTS*"))
    for d in dirs:
        if "_" in d.split(os.sep)[-1]:
            files = glob(os.path.join(d, "*.nii.gz"))
            for f in files:
                if "flair" in f or "t1" in f or "t1ce" in f or "t2" in f:
                    continue
                if "_seg" in f:
                    shutil.move(f, lbl_path)
                else:
                    shutil.move(f, img_path)
            shutil.rmtree(d)

def prepare_dataset_json(data, train):
    """Create a dataset.json file describing the dataset."""
    images = glob(os.path.join(data, "images", "*"))
    labels = glob(os.path.join(data, "labels", "*"))
    images = sorted([os.path.join("images", os.path.basename(img)).replace(os.sep, "/") for img in images])
    labels = sorted([os.path.join("labels", os.path.basename(lbl)).replace(os.sep, "/") for lbl in labels])

    modality = {"0": "FLAIR", "1": "T1", "2": "T1CE", "3": "T2"}
    labels_dict = {"0": "background", "1": "non-enhancing tumor", "2": "edema", "3": "enhancing tumor"}
    if train:
        key = "training"
        data_pairs = [{"image": img, "label": lbl} for (img, lbl) in zip(images, labels)]
    else:
        key = "test"
        data_pairs = [{"image": img} for img in images]

    dataset = {
        "labels": labels_dict,
        "modality": modality,
        key: data_pairs,
    }

    with open(os.path.join(data, "dataset.json"), "w") as outfile:
        json.dump(dataset, outfile)

def prepare_dataset(data, train):
    """Preprocess the entire dataset."""
    logging.info(f"Preparing BraTS2021 dataset from: {data}")
    start = time.time()
    for d in sorted(glob(os.path.join(data, "BraTS*"))):
        prepare_nifty(d)
    prepare_dirs(data, train)
    prepare_dataset_json(data, train)
    end = time.time()
    logging.info(f"Preparing time: {(end - start):.2f}s")

def get_data_list(data_dir):
    """Load the dataset.json file and prepare a list of data dictionaries."""
    with open(os.path.join(data_dir, "dataset.json"), "r") as f:
        dataset = json.load(f)

    data_list = []
    for item in dataset["training"]:
        image_path = os.path.join(data_dir, item["image"]).replace(os.sep, "/")
        seg_path = os.path.join(data_dir, item["label"]).replace(os.sep, "/")
        data_dict = {
            "image": image_path,
            "seg": seg_path
        }
        data_list.append(data_dict)
    return data_list

def print_nii_info(data_list):
    """Print crucial information from .nii.gz files."""
    for data_dict in data_list[:5]:
        logging.info(f"Sample: {os.path.basename(data_dict['image']).split('.nii.gz')[0]}")
        nii = nib.load(data_dict["image"])
        data = nii.get_fdata()
        logging.info(f"Image: Shape={nii.shape}, Dtype={nii.get_data_dtype()}, "
              f"Min={np.min(data)}, Max={np.max(data)}")
        seg = nib.load(data_dict["seg"])
        seg_data = seg.get_fdata()
        logging.info(f"Seg: Shape={seg.shape}, Dtype={seg.get_data_dtype()}, "
              f"Min={np.min(seg_data)}, Max={np.max(seg_data)}")
        unique_labels = np.unique(seg_data)
        logging.info(f"Segmentation Labels: {unique_labels}")
        logging.info("---")

def compute_sample_class_counts(sample):
    """Compute class counts for a single sample."""
    seg_path = sample["seg"]
    seg = nib.load(seg_path).get_fdata()
    class_counts = np.zeros(4)
    for label in range(4):
        class_counts[label] = np.sum(seg == label)
    return class_counts

def compute_class_distribution(data_list):
    """Compute the class distribution across the dataset in parallel."""
    if not data_list:
        logging.error("data_list is empty in compute_class_distribution.")
        return np.ones(4) / 4  # Return uniform distribution as fallback
    
    start_time = time.time()
    num_processes = min(cpu_count(), len(data_list))  # Use available CPUs or fewer if data_list is small
    logging.info(f"Computing class distribution with {num_processes} parallel processes.")
    
    with Pool(processes=num_processes) as pool:
        results = pool.map(compute_sample_class_counts, data_list)
    
    # Aggregate results
    class_counts = np.sum(results, axis=0)
    total_voxels = np.sum(class_counts)
    class_frequencies = class_counts / total_voxels if total_voxels > 0 else np.ones_like(class_counts)
    
    logging.info(f"Class distribution - Background: {class_frequencies[0]:.4f}, "
                 f"NCR/NET: {class_frequencies[1]:.4f}, Edema: {class_frequencies[2]:.4f}, "
                 f"ET: {class_frequencies[3]:.4f}")
    logging.info(f"Class distribution computation took {(time.time() - start_time):.2f}s")
    return class_frequencies

def compute_class_weights(class_frequencies, epsilon=1e-6):
    """Compute class weights based on inverse frequency."""
    inverse_freq = 1.0 / (class_frequencies + epsilon)
    weights = inverse_freq / np.sum(inverse_freq)
    logging.info(f"Computed class weights - Background: {weights[0]:.4f}, "
          f"NCR/NET: {weights[1]:.4f}, Edema: {weights[2]:.4f}, "
          f"ET: {weights[3]:.4f}")
    return weights

def get_transforms(config, train=True):
    """Define data transforms for training and validation."""
    keys = ["image", "seg"]
    spatial_size = [128, 128, 144]  # Adjusted depth to 144 (divisible by 16)
    config["spatial_size"] = spatial_size

    base_transforms = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        SpatialPadd(keys=keys, spatial_size=spatial_size, mode="constant", constant_values=0),
        NormalizeIntensityd(keys=["image"], channel_wise=True, nonzero=True),
    ]
    
    if train:
        general_rotation_prob = config.get("general_rotation_prob", 0.3)
        general_flip_prob = config.get("general_flip_prob", 0.3)
        general_contrast_prob = config.get("general_contrast_prob", 0.2)
        general_contrast_gamma = config.get("general_contrast_gamma", [0.9, 1.1])
        
        train_transforms = base_transforms + [
            RandCropByPosNegLabeld(
                keys=keys,
                label_key="seg",
                spatial_size=spatial_size,
                pos=3,
                neg=1,
                num_samples=2,
                image_key="image",
                image_threshold=0,
                allow_smaller=False,
            ),
            RandRotate90d(keys=keys, prob=general_rotation_prob, max_k=3),
            RandFlipd(keys=keys, prob=general_flip_prob, spatial_axis=0),
            RandFlipd(keys=keys, prob=general_flip_prob, spatial_axis=1),
            RandFlipd(keys=keys, prob=general_flip_prob, spatial_axis=2),
            RandAdjustContrastd(keys=["image"], prob=general_contrast_prob, gamma=general_contrast_gamma),
            ToTensord(keys=keys),
        ]
        return Compose(train_transforms)
    else:
        val_transforms = base_transforms + [
            RandCropByPosNegLabeld(
                keys=keys,
                label_key="seg",
                spatial_size=spatial_size,
                pos=3,
                neg=1,
                num_samples=1,
                image_key="image",
                image_threshold=0,
                allow_smaller=False,
            ),
            ToTensord(keys=keys),
        ]
        return Compose(val_transforms)

def prepare_data(data_dir, config):
    """Prepare the dataset and return a list of data dictionaries."""
    if not os.path.exists(os.path.join(data_dir, "dataset.json")):
        logging.info(f"Preparing BraTS2021 dataset from: {data_dir}")
        prepare_dataset(data_dir, train=True)
    else:
        logging.info(f"BraTS2021 dataset already prepared at: {data_dir}")
    
    data_list = get_data_list(data_dir)
    print_nii_info(data_list)
    return data_list