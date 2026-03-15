"""
Image Preprocessing Module

Provides image quality validation, enhancement, and standardization
for fall hazard detection analysis.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import base64


class ImageQuality(Enum):
    """Image quality classification for hazard detection suitability."""
    EXCELLENT = "excellent"  # Ideal for analysis
    GOOD = "good"           # Suitable with minor issues
    ACCEPTABLE = "acceptable"  # Usable but may affect accuracy
    POOR = "poor"           # May significantly impact analysis
    REJECTED = "rejected"   # Not suitable for analysis


@dataclass
class ImageMetadata:
    """
    Metadata extracted from image analysis.

    Attributes:
        width: Image width in pixels
        height: Image height in pixels
        channels: Number of color channels
        format: File format extension
        file_size_kb: File size in kilobytes
        brightness: Average brightness (0-1)
        contrast: Contrast measure (0-1)
        blur_score: Blur detection score (higher = sharper)
        quality: Overall quality classification
        issues: List of detected quality issues
    """
    width: int
    height: int
    channels: int
    format: str
    file_size_kb: float
    brightness: float
    contrast: float
    blur_score: float
    quality: ImageQuality
    issues: List[str]


@dataclass
class ProcessingResult:
    """
    Result of image processing operation.

    Attributes:
        success: Whether processing succeeded
        image: Processed image array (if successful)
        metadata: Image metadata
        original_path: Path to original image
        error: Error message (if failed)
    """
    success: bool
    image: Optional[np.ndarray]
    metadata: Optional[ImageMetadata]
    original_path: str
    error: Optional[str] = None


class ImageProcessor:
    """
    Image preprocessing for fall hazard detection.

    Capabilities:
    - Quality validation (brightness, contrast, blur)
    - Resolution standardization
    - Brightness/contrast enhancement
    - Noise reduction
    - Format conversion

    Usage:
        processor = ImageProcessor(target_size=(1024, 1024))
        result = processor.process("path/to/image.jpg")
        if result.success:
            processed_image = result.image
            quality = result.metadata.quality
    """

    # Quality thresholds
    MIN_SIZE = (256, 256)
    MAX_SIZE = (4096, 4096)
    MIN_BRIGHTNESS = 0.15
    MAX_BRIGHTNESS = 0.95
    MIN_CONTRAST = 0.15
    MIN_BLUR_SCORE = 50
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}

    def __init__(
        self,
        target_size: Tuple[int, int] = (1024, 1024),
        maintain_aspect_ratio: bool = True,
        auto_enhance: bool = True
    ):
        """
        Initialize the image processor.

        Args:
            target_size: Target dimensions (width, height)
            maintain_aspect_ratio: Keep original aspect ratio when resizing
            auto_enhance: Automatically enhance poor quality images
        """
        self.target_size = target_size
        self.maintain_aspect_ratio = maintain_aspect_ratio
        self.auto_enhance = auto_enhance

    def process(
        self,
        image_path: str,
        validate: bool = True,
        enhance: bool = None,
        resize: bool = True
    ) -> ProcessingResult:
        """
        Process an image for hazard detection.

        Args:
            image_path: Path to the image file
            validate: Run quality validation
            enhance: Apply enhancement (None = use auto_enhance setting)
            resize: Resize to target dimensions

        Returns:
            ProcessingResult with processed image and metadata
        """
        path = Path(image_path)

        # Validate file exists
        if not path.exists():
            return ProcessingResult(
                success=False,
                image=None,
                metadata=None,
                original_path=str(path),
                error=f"File not found: {image_path}"
            )

        # Validate format
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            return ProcessingResult(
                success=False,
                image=None,
                metadata=None,
                original_path=str(path),
                error=f"Unsupported format: {path.suffix}"
            )

        try:
            # Load image
            img = cv2.imread(str(path))
            if img is None:
                return ProcessingResult(
                    success=False,
                    image=None,
                    metadata=None,
                    original_path=str(path),
                    error="Failed to load image"
                )

            # Extract metadata and assess quality
            metadata = self._analyze_image(img, path)

            # Check if image should be rejected
            if validate and metadata.quality == ImageQuality.REJECTED:
                return ProcessingResult(
                    success=False,
                    image=None,
                    metadata=metadata,
                    original_path=str(path),
                    error=f"Image quality too poor: {', '.join(metadata.issues)}"
                )

            # Apply enhancements if needed
            should_enhance = enhance if enhance is not None else self.auto_enhance
            if should_enhance and metadata.quality in (ImageQuality.POOR, ImageQuality.ACCEPTABLE):
                img = self._enhance_image(img, metadata)

            # Resize if needed
            if resize:
                img = self._resize_image(img)

            return ProcessingResult(
                success=True,
                image=img,
                metadata=metadata,
                original_path=str(path)
            )

        except Exception as e:
            return ProcessingResult(
                success=False,
                image=None,
                metadata=None,
                original_path=str(path),
                error=str(e)
            )

    def process_batch(
        self,
        image_paths: List[str],
        skip_failed: bool = True
    ) -> List[ProcessingResult]:
        """
        Process multiple images.

        Args:
            image_paths: List of image file paths
            skip_failed: Continue processing if some images fail

        Returns:
            List of ProcessingResult objects
        """
        results = []
        for path in image_paths:
            result = self.process(path)
            if result.success or not skip_failed:
                results.append(result)
        return results

    def validate_only(self, image_path: str) -> ImageMetadata:
        """
        Validate image quality without processing.

        Args:
            image_path: Path to the image file

        Returns:
            ImageMetadata with quality assessment
        """
        path = Path(image_path)
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        return self._analyze_image(img, path)

    def _analyze_image(self, img: np.ndarray, path: Path) -> ImageMetadata:
        """
        Analyze image and extract quality metrics.

        Args:
            img: Image array (BGR format)
            path: Original file path

        Returns:
            ImageMetadata with all quality metrics
        """
        height, width = img.shape[:2]
        channels = img.shape[2] if len(img.shape) > 2 else 1

        # Convert to grayscale for analysis
        if channels > 1:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Calculate brightness (normalized mean)
        brightness = np.mean(gray) / 255.0

        # Calculate contrast (normalized std dev)
        contrast = np.std(gray) / 128.0

        # Calculate blur score (Laplacian variance)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Get file size
        file_size_kb = path.stat().st_size / 1024

        # Assess quality and identify issues
        issues = []
        quality = self._assess_quality(
            width, height, brightness, contrast, blur_score, issues
        )

        return ImageMetadata(
            width=width,
            height=height,
            channels=channels,
            format=path.suffix.lower(),
            file_size_kb=round(file_size_kb, 2),
            brightness=round(brightness, 3),
            contrast=round(contrast, 3),
            blur_score=round(blur_score, 2),
            quality=quality,
            issues=issues
        )

    def _assess_quality(
        self,
        width: int,
        height: int,
        brightness: float,
        contrast: float,
        blur_score: float,
        issues: List[str]
    ) -> ImageQuality:
        """
        Assess overall image quality for hazard detection.

        Args:
            width, height: Image dimensions
            brightness: Normalized brightness (0-1)
            contrast: Normalized contrast
            blur_score: Blur detection score
            issues: List to append detected issues

        Returns:
            ImageQuality classification
        """
        severity = 0  # Accumulate issue severity

        # Check resolution
        if width < self.MIN_SIZE[0] or height < self.MIN_SIZE[1]:
            issues.append(f"Resolution too low ({width}x{height})")
            severity += 3

        # Check brightness
        if brightness < self.MIN_BRIGHTNESS:
            issues.append(f"Too dark (brightness: {brightness:.2f})")
            severity += 2
        elif brightness > self.MAX_BRIGHTNESS:
            issues.append(f"Too bright/overexposed (brightness: {brightness:.2f})")
            severity += 2
        elif brightness < 0.25 or brightness > 0.85:
            issues.append(f"Suboptimal brightness ({brightness:.2f})")
            severity += 1

        # Check contrast
        if contrast < self.MIN_CONTRAST:
            issues.append(f"Low contrast ({contrast:.2f})")
            severity += 2
        elif contrast < 0.25:
            issues.append(f"Suboptimal contrast ({contrast:.2f})")
            severity += 1

        # Check blur
        if blur_score < self.MIN_BLUR_SCORE:
            issues.append(f"Image is blurry (score: {blur_score:.0f})")
            severity += 2
        elif blur_score < 100:
            issues.append(f"Slight blur detected (score: {blur_score:.0f})")
            severity += 1

        # Classify quality
        if severity == 0:
            return ImageQuality.EXCELLENT
        elif severity <= 1:
            return ImageQuality.GOOD
        elif severity <= 3:
            return ImageQuality.ACCEPTABLE
        elif severity <= 5:
            return ImageQuality.POOR
        else:
            return ImageQuality.REJECTED

    def _resize_image(self, img: np.ndarray) -> np.ndarray:
        """
        Resize image to target size.

        Args:
            img: Input image array

        Returns:
            Resized image array
        """
        height, width = img.shape[:2]
        target_w, target_h = self.target_size

        # Skip if already within acceptable range
        if width <= target_w and height <= target_h:
            return img

        if self.maintain_aspect_ratio:
            # Calculate scaling factor
            scale = min(target_w / width, target_h / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
        else:
            new_width, new_height = target_w, target_h

        # Use appropriate interpolation
        if new_width < width:
            # Downscaling - use INTER_AREA
            interpolation = cv2.INTER_AREA
        else:
            # Upscaling - use INTER_CUBIC
            interpolation = cv2.INTER_CUBIC

        return cv2.resize(img, (new_width, new_height), interpolation=interpolation)

    def _enhance_image(
        self,
        img: np.ndarray,
        metadata: ImageMetadata
    ) -> np.ndarray:
        """
        Enhance image quality for better hazard detection.

        Args:
            img: Input image array
            metadata: Image metadata with quality metrics

        Returns:
            Enhanced image array
        """
        enhanced = img.copy()

        # Enhance brightness if too dark
        if metadata.brightness < 0.35:
            factor = 0.35 / max(metadata.brightness, 0.1)
            factor = min(factor, 1.5)  # Cap enhancement
            enhanced = self._adjust_brightness(enhanced, factor)

        # Enhance contrast if too low
        if metadata.contrast < 0.35:
            enhanced = self._apply_clahe(enhanced)

        # Apply mild denoising
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)

        return enhanced

    def _adjust_brightness(
        self,
        img: np.ndarray,
        factor: float
    ) -> np.ndarray:
        """Adjust image brightness using HSV conversion."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _apply_clahe(self, img: np.ndarray) -> np.ndarray:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def to_base64(self, img: np.ndarray, format: str = ".jpg") -> str:
        """
        Convert image array to base64 string.

        Args:
            img: Image array
            format: Output format (.jpg, .png)

        Returns:
            Base64 encoded string
        """
        _, buffer = cv2.imencode(format, img)
        return base64.b64encode(buffer).decode('utf-8')

    def save_processed(
        self,
        img: np.ndarray,
        output_path: str,
        quality: int = 95
    ) -> bool:
        """
        Save processed image to file.

        Args:
            img: Image array to save
            output_path: Output file path
            quality: JPEG quality (1-100)

        Returns:
            True if successful
        """
        try:
            params = []
            if output_path.lower().endswith(('.jpg', '.jpeg')):
                params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            elif output_path.lower().endswith('.png'):
                params = [cv2.IMWRITE_PNG_COMPRESSION, 9 - (quality // 11)]

            cv2.imwrite(output_path, img, params)
            return True
        except Exception:
            return False


class BatchProcessor:
    """
    Batch processing utility for multiple images.

    Usage:
        batch = BatchProcessor(input_dir="data/raw", output_dir="data/processed")
        results = batch.process_all()
        batch.generate_report(results)
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        processor: Optional[ImageProcessor] = None
    ):
        """
        Initialize batch processor.

        Args:
            input_dir: Directory containing input images
            output_dir: Directory for processed images
            processor: ImageProcessor instance (creates default if None)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.processor = processor or ImageProcessor()

        # Create output directory if needed
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_all(
        self,
        save_processed: bool = True,
        extensions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Process all images in input directory.

        Args:
            save_processed: Save processed images to output directory
            extensions: File extensions to process (None = all supported)

        Returns:
            Summary dictionary with results
        """
        extensions = extensions or ['.jpg', '.jpeg', '.png', '.webp']

        # Find all images
        image_paths = []
        for ext in extensions:
            image_paths.extend(self.input_dir.glob(f"*{ext}"))
            image_paths.extend(self.input_dir.glob(f"*{ext.upper()}"))

        results = {
            "total": len(image_paths),
            "processed": 0,
            "failed": 0,
            "quality_distribution": {q.value: 0 for q in ImageQuality},
            "details": []
        }

        for img_path in image_paths:
            result = self.processor.process(str(img_path))

            detail = {
                "filename": img_path.name,
                "success": result.success,
                "error": result.error
            }

            if result.success and result.metadata:
                detail["quality"] = result.metadata.quality.value
                detail["brightness"] = result.metadata.brightness
                detail["contrast"] = result.metadata.contrast
                detail["issues"] = result.metadata.issues

                results["processed"] += 1
                results["quality_distribution"][result.metadata.quality.value] += 1

                # Save processed image
                if save_processed and result.image is not None:
                    output_path = self.output_dir / img_path.name
                    self.processor.save_processed(result.image, str(output_path))
            else:
                results["failed"] += 1

            results["details"].append(detail)

        return results

    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a text report from batch processing results.

        Args:
            results: Results dictionary from process_all()

        Returns:
            Formatted report string
        """
        lines = [
            "=" * 50,
            "IMAGE PREPROCESSING BATCH REPORT",
            "=" * 50,
            f"\nTotal Images: {results['total']}",
            f"Successfully Processed: {results['processed']}",
            f"Failed: {results['failed']}",
            "\nQuality Distribution:",
        ]

        for quality, count in results['quality_distribution'].items():
            if count > 0:
                lines.append(f"  {quality.upper()}: {count}")

        if results['failed'] > 0:
            lines.append("\nFailed Images:")
            for detail in results['details']:
                if not detail['success']:
                    lines.append(f"  - {detail['filename']}: {detail['error']}")

        lines.append("\n" + "=" * 50)

        return "\n".join(lines)
