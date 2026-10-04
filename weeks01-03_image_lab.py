import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def inspect_image(image_path: str) -> dict:
    """Task 1: Load image and measure data properties."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    height, width, channels = img.shape
    pixel_count = height * width
    estimated_bytes = pixel_count * channels * 1  # 8-bit per channel
    color_order = "BGR"  # OpenCV loads images in BGR format

    return {
        "width": width,
        "height": height,
        "channels": channels,
        "shape": [height, width, channels],
        "pixel_count": pixel_count,
        "estimated_bytes": estimated_bytes,
        "color_order": color_order
    }

def create_pixel_views(image_path: str, output_dir: str) -> dict:
    """Task 2: Channel, grayscale, and downsampled views."""
    os.makedirs(output_dir, exist_ok=True)
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    b, g, r = cv2.split(img_bgr)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    h, w = img_bgr.shape[:2]
    new_w, new_h = w // 2, h // 2
    downsampled = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes[0, 0].imshow(r, cmap='gray')
    axes[0, 0].set_title("Red Channel")
    
    axes[0, 1].imshow(g, cmap='gray')
    axes[0, 1].set_title("Green Channel")
    
    axes[0, 2].imshow(b, cmap='gray')
    axes[0, 2].set_title("Blue Channel")
    
    axes[1, 0].imshow(gray, cmap='gray')
    axes[1, 0].set_title("Grayscale")
    
    axes[1, 1].imshow(downsampled)
    axes[1, 1].set_title(f"Downsampled ({new_w}x{new_h})")
    
    axes[1, 2].imshow(img_rgb)
    axes[1, 2].set_title(f"Original RGB ({w}x{h})")

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()
    output_path = os.path.join(output_dir, "pixel_views.png")
    plt.savefig(output_path)
    plt.close()

    return {
        "original_size": [w, h],
        "downsampled_size": [new_w, new_h],
        "output_path": output_path
    }

def create_adjustments(
    image_path: str,
    output_dir: str,
    brightness_delta: int = 40,
    contrast_factor: float = 1.5,
    threshold: int = 127,
) -> dict:
    """Task 3: Brightness, contrast, and thresholding."""
    os.makedirs(output_dir, exist_ok=True)
    img_bgr = cv2.imread(image_path)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Brightness adjustment with clipping
    bright = np.clip(gray.astype(np.int16) + brightness_delta, 0, 255).astype(np.uint8)

    # Contrast adjustment with clipping
    contrast = np.clip(gray.astype(np.float32) * contrast_factor, 0, 255).astype(np.uint8)

    # Thresholding
    _, thresh_img = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title("Original Grayscale")

    axes[0, 1].imshow(bright, cmap='gray')
    axes[0, 1].set_title(f"Brighter (Delta={brightness_delta})")

    axes[1, 0].imshow(contrast, cmap='gray')
    axes[1, 0].set_title(f"High Contrast (Factor={contrast_factor})")

    axes[1, 1].imshow(thresh_img, cmap='gray')
    axes[1, 1].set_title(f"Thresholded (Val={threshold})")

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()
    output_path = os.path.join(output_dir, "adjustments.png")
    plt.savefig(output_path)
    plt.close()

    return {
        "brightness_delta": brightness_delta,
        "contrast_factor": contrast_factor,
        "threshold": threshold,
        "output_path": output_path
    }

def create_blur_and_edges(
    image_path: str,
    output_dir: str,
    kernel_size: int = 5,
) -> dict:
    """Task 4: Blur and Sobel edge detection."""
    if kernel_size % 2 == 0 or kernel_size <= 0:
        raise ValueError("kernel_size must be a positive odd integer.")

    os.makedirs(output_dir, exist_ok=True)
    img_bgr = cv2.imread(image_path)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Mean Blur
    blurred = cv2.blur(gray, (kernel_size, kernel_size))

    # Sobel on Original Grayscale
    sobelx_orig = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely_orig = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_orig = np.clip(cv2.magnitude(sobelx_orig, sobely_orig), 0, 255).astype(np.uint8)

    # Sobel on Blurred Grayscale
    sobelx_blur = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely_blur = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_blur = np.clip(cv2.magnitude(sobelx_blur, sobely_blur), 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title("Original Grayscale")

    axes[0, 1].imshow(blurred, cmap='gray')
    axes[0, 1].set_title(f"Mean Blur ({kernel_size}x{kernel_size})")

    axes[1, 0].imshow(sobel_orig, cmap='gray')
    axes[1, 0].set_title("Sobel Edges (Original)")

    axes[1, 1].imshow(sobel_blur, cmap='gray')
    axes[1, 1].set_title("Sobel Edges (Blurred)")

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()
    output_path = os.path.join(output_dir, "blur_and_edges.png")
    plt.savefig(output_path)
    plt.close()

    return {
        "kernel_size": kernel_size,
        "output_path": output_path
    }

def run_lab(image_path: str, output_dir: str) -> dict:
    """Task 5: Run all tasks."""
    return {
        "task1": inspect_image(image_path),
        "task2": create_pixel_views(image_path, output_dir),
        "task3": create_adjustments(image_path, output_dir),
        "task4": create_blur_and_edges(image_path, output_dir)
    }

def main() -> None:
    image_path = "images/original.jpg"
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    results = run_lab(image_path, output_dir)
    print("Execution complete. Properties:")
    print(results["task1"])

if __name__ == "__main__":
    main()