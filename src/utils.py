"""
This file contains the utility functions used in this project
"""

import os
import random
import math
from pathlib import Path
import numpy as np
from numpy.typing import ArrayLike # for type-hint
import pandas as pd
import seaborn as sns
from seaborn.axisgrid import JointGrid # for type-hint
import matplotlib.pyplot as plt
from matplotlib.axes import Axes # for type-hint
from typing import Optional, List, Dict, Any, Tuple
from config import settings
import kagglehub
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
# Image.MAX_IMAGE_PIXELS = None # Suppress warnings for large images if necessary
import logging
logger = logging.getLogger(__name__)

#----------------------------------------------------------------------------------------
def copy_single_file(args):
    """
    helper function used in ThreadPoolExecutor-map to copy single file
    Args:
        args: a tuple contains source and destination path for the file to be copied
    Return:
        None
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        src_file, dst_file = args
        shutil.copy2(src_file, dst_file) # all info/permission/metadata
        #shutil.copy(src_file, dst_file) # no metadata but permission
        #shutil.copyfile(src_file, dst_file) # just file data (should be fine)
    except Exception as e:
        logger.error(f"Error in copy_single_file: {e}")
        raise

#----------------------------------------------------------------------------------------
def fast_copy_tree(src, dst, max_workers=16):
    """
    fast copy of huge number of files, in src folder to dst folder
    Args:
        src: source folder
        dst: destination folder
        max_workers: max number of workers/thread use for copy by ThreadPoolExecutor
    Return:
        None
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
        """
    try:
        if not os.path.exists(dst):
            os.makedirs(dst)

        tasks = []
        for root, dirs, files in os.walk(src):
            # Recreate directory structure at destination
            rel_path = os.path.relpath(root, src)
            dest_dir = os.path.join(dst, rel_path)
            os.makedirs(dest_dir, exist_ok=True)

            for file in files:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(dest_dir, file)
                tasks.append((src_file, dst_file))

        # Copy files concurrently using threads
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(copy_single_file, tasks)
    except Exception as e:
        logger.error(f"Error in fast_copy_tree: {e}")
        raise

#----------------------------------------------------------------------------------------
def download_dataset_from_kagglehub(dataset_id:str = settings.DATASET_ID,
                                    dataset_path:str = settings.DATASET_PATH,
                                    dl_dir:Path = settings.DATA_DIR):
    """
    download dataset from kagglehub into standard kagglehub-cache
    then copy to local dl_dir, if not downloaded yet
    Args:
        dataset_id: dataset id
        dataset_path: dataset file-path or a representative when we have multiple files
        dl_dir: destination download directory
    Returns:
        None.
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    if dataset_path.exists():
        print(f"dataset is already downloaded in {dl_dir}")
        return
    try:
        cache_path = kagglehub.dataset_download(dataset_id)
        print(f'dataset downloaded in kagglehub-cache-path = {cache_path}')

        #shutil.copytree(cache_path, dl_dir, dirs_exist_ok=True) # extremely slow for 65000+ files
        print('copy file to data folder: WAIT ...')
        fast_copy_tree(cache_path, dl_dir, max_workers=8)
        print(f"Download and copy completed! Files saved in {dl_dir} folder.")

    except Exception as e:
        logger.error(f"Error in download_dataset_from_kagglehub: {e}")
        raise


#----------------------------------------------------------------------------------------
def check_image(file_path):
    """
    Check a single image width/height, is-RGB, and if-corrupted
    Args:
        file_path: image path to be checked
    Return:
        file_path, width, height, mode, is_rgb, error (if error is not None, other info set to None except file_path)
    """
    try:
        with Image.open(file_path) as img:
            width, height = img.size
            mode = img.mode
            is_rgb = (mode == 'RGB')
            return file_path, width, height, mode, is_rgb, None
    except Exception as e:
        return file_path, None, None, None, False, str(e)

#----------------------------------------------------------------------------------------
def scan_images(folder_path, max_workers=8):
    """
    scan all images in folder_path, to check their width/height, if they are RGB, and if they are corrupted
    Args:
        folder_path: folder path contains images to be scaned/checked
        max_workers: max number of workers/thread use for copy by ThreadPoolExecutor
    Returns:
        dataframe contains width/height/aspect-ratio, non_rgb_file, and corrupted_files.
    Raises:
        Exception: Re-raises any exception caught during processing after logging.
    """
    try:
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

        # Gather paths first
        file_paths = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(image_extensions):
                    file_paths.append(os.path.join(root, file))

        print(f"Found {len(file_paths)} images. Starting scan...")

        widths = []
        heights = []
        non_rgb_files = []
        corrupted_files = []

        # Run thread pool
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(check_image, path): path for path in file_paths}

            for i, future in enumerate(as_completed(futures)):
                path, w, h, mode, is_rgb, error = future.result()

                if error:
                    corrupted_files.append((path, error))
                else:
                    widths.append(w)
                    heights.append(h)
                    if not is_rgb:
                        non_rgb_files.append((path, mode))

                # Print progress every 10000 files
                if (i + 1) % 10000 == 0:
                    print(f"Processed {i + 1} / {len(file_paths)} files...")

        print(f"Number of Corrupted or unreadable images: {len(corrupted_files)}")
        print(f"Number of Non-RGB images: {len(non_rgb_files)}")
        df = pd.DataFrame()
        df['width'] = widths
        df['height']= heights
        df['aspect_ratio']=df['width']/df['height']

        # Convert lists to NumPy arrays for easy plotting/histogram analysis
        return df, non_rgb_files, corrupted_files
    except Exception as e:
        logger.error(f"Error in scan_images: {e}")
        raise

#----------------------------------------------------------------------------------------
def show_sample_images_in_grid(image_dir: Path, target_size: tuple = (64, 64), figsize: tuple = (12, 6),
                               num_images: int = 16, images_per_row: int = 4, padding: int = 8):
    """
    Randomly selects images from image_dir, resizes them, and plots them in a grid.
    Args:
        image_dir: Path to the directory containing images.
        target_size: (width, height) to resize images.
        figsize: Matplotlib figure size (width, height).
        num_images: Total number of images to plot.
        images_per_row: Number of images per row in the grid.
        padding: Pixel padding between images in the grid.
    """
    try:
        image_dir = Path(image_dir)
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_paths = [p for p in image_dir.rglob("*") if p.suffix.lower() in valid_extensions]

        if not image_paths:
            raise FileNotFoundError(f"No valid image files found in {image_dir}")

        sample_paths = random.sample(image_paths, min(num_images, len(image_paths)))

        # Calculate grid dimensions
        n_images = len(sample_paths)
        rows = (n_images + images_per_row - 1) // images_per_row

        img_w, img_h = target_size
        grid_h = rows * img_h + (rows + 1) * padding
        grid_w = images_per_row * img_w + (images_per_row + 1) * padding

        # Create white or black canvas for padding (255 or 0 = uint8 white)
        grid = np.full((grid_h, grid_w, 3), 0, dtype=np.uint8) # we used black

        for idx, path in enumerate(sample_paths):
            r = idx // images_per_row
            c = idx % images_per_row

            with Image.open(path) as img:
                img = img.convert("RGB").resize(target_size)
                img_np = np.asarray(img)

            y_start = padding + r * (img_h + padding)
            x_start = padding + c * (img_w + padding)

            grid[y_start:y_start + img_h, x_start:x_start + img_w] = img_np

        plt.figure(figsize=figsize)
        plt.imshow(grid)
        plt.axis("off")
        plt.tight_layout()
    except Exception as e:
        logger.error(f"Error in show_sample_images_in_grid: {e}")
        raise