"""
AquaChain Photo Verification Module
Step 1: OpenCV-based authenticity verification for coastal ecosystem photos
"""

import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Union
import hashlib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PhotoVerifier:
    """
    Main class for verifying field agent photo submissions
    Detects fraud, manipulation, and quality issues
    """

    def __init__(self):
        # Verification thresholds
        self.MIN_PHOTOS_REQUIRED = 3
        self.AUTHENTICITY_THRESHOLD = 0.7
        self.BLUR_THRESHOLD = 100
        self.DARKNESS_THRESHOLD = 50
        self.MIN_RESOLUTION = (800, 600)

        # Initialize OpenCV components - FIXED: corrected filename
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # Color ranges for vegetation detection (HSV)
        self.vegetation_lower = np.array([35, 40, 40])
        self.vegetation_upper = np.array([85, 255, 255])

        # Water color ranges (HSV)
        self.water_lower = np.array([90, 50, 50])
        self.water_upper = np.array([130, 255, 255])

        self.verification_results = []
        self.seen_hashes = set()  # Initialize to prevent AttributeError

    def verify_photo_batch(self, image_paths: List[str]) -> Dict:
        """
        Main entry point for verifying a batch of photos

        Args:
            image_paths: List of paths to images to verify

        Returns:
            Dictionary containing verification results
        """
        logger.info(f"Starting verification of {len(image_paths)} photos...")

        # Check minimum photos requirement
        if len(image_paths) < self.MIN_PHOTOS_REQUIRED:
            return {
                'status': 'REJECTED',
                'reason': f'Insufficient photos. Minimum {self.MIN_PHOTOS_REQUIRED} required.',
                'photos_analyzed': len(image_paths),
                'verification_score': 0.0
            }

        verified_photos = []
        rejected_photos = []

        for idx, image_path in enumerate(image_paths):
            logger.info(f"Processing photo {idx + 1}/{len(image_paths)}: {os.path.basename(image_path)}")

            if not os.path.exists(image_path):
                rejected_photos.append({
                    'path': image_path,
                    'filename': os.path.basename(image_path),
                    'reason': 'File not found'
                })
                continue

            # Run comprehensive verification
            verification_result = self._verify_single_photo(image_path)

            if verification_result['is_valid']:
                verified_photos.append(verification_result)
            else:
                rejected_photos.append(verification_result)

        # Calculate overall verification score
        total_score = self._calculate_batch_score(verified_photos, rejected_photos)

        # Determine final status
        status = 'APPROVED' if total_score >= self.AUTHENTICITY_THRESHOLD else 'REJECTED'

        return {
            'status': status,
            'verification_score': total_score,
            'total_photos': len(image_paths),
            'verified_photos': len(verified_photos),
            'rejected_photos': len(rejected_photos),
            'detailed_results': {
                'verified': verified_photos,
                'rejected': rejected_photos
            },
            'timestamp': datetime.now().isoformat()
        }

    def _verify_single_photo(self, image_path: str) -> Dict:
        """
        Comprehensive verification of a single photo
        """
        result = {
            'path': image_path,
            'filename': os.path.basename(image_path),
            'is_valid': False,
            'checks': {},
            'scores': {},
            'metadata': {}
        }

        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                result['reason'] = 'Cannot read image file'
                return result

            # Extract metadata
            metadata = self._extract_metadata(image_path)
            result['metadata'] = metadata

            # Run all verification checks
            checks = {
                'screen_detection': self._detect_screen_photo(image),
                'manipulation_detection': self._detect_manipulation(image, image_path),
                'content_validation': self._validate_ecosystem_content(image),
                'quality_check': self._check_image_quality(image),
                'metadata_check': self._verify_metadata(metadata),
                'duplicate_check': self._check_duplicate(image)
            }

            result['checks'] = checks

            # Calculate individual scores
            scores = {
                'screen_score': 1.0 - checks['screen_detection']['confidence'],
                'manipulation_score': 1.0 - checks['manipulation_detection']['confidence'],
                'content_score': checks['content_validation']['ecosystem_confidence'],
                'quality_score': checks['quality_check']['quality_score'],
                'metadata_score': checks['metadata_check']['validity_score']
            }

            result['scores'] = scores

            # Calculate overall authenticity score
            weights = {
                'screen_score': 0.25,
                'manipulation_score': 0.25,
                'content_score': 0.20,
                'quality_score': 0.15,
                'metadata_score': 0.15
            }

            authenticity_score = sum(
                scores[key] * weights[key] for key in weights
            )

            result['authenticity_score'] = authenticity_score
            result['is_valid'] = authenticity_score >= self.AUTHENTICITY_THRESHOLD

            if not result['is_valid']:
                # Find the main reason for rejection
                min_score_key = min(scores, key=scores.get)
                result['reason'] = f'Failed {min_score_key.replace("_score", "")} check'

        except Exception as e:
            logger.error(f"Error verifying {image_path}: {e}")
            result['reason'] = f'Verification error: {str(e)}'

        return result

    def _detect_screen_photo(self, image: np.ndarray) -> Dict:
        """
        Detect if the photo is taken of a screen/monitor
        Uses FFT analysis and moiré pattern detection
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # FFT Analysis for screen patterns
        fft = np.fft.fft2(gray)
        fft_magnitude = np.abs(fft)

        # Look for regular frequency patterns (screen refresh rates)
        height, width = gray.shape

        # Check for suspicious regular patterns
        peak_threshold = np.percentile(fft_magnitude, 99.9)
        regular_peaks = np.sum(fft_magnitude > peak_threshold)
        screen_pattern_score = regular_peaks / (height * width)

        # Edge detection for screen bezels
        edges = cv2.Canny(gray, 50, 150)

        # Look for rectangular patterns (screen edges)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rectangular_contours = 0

        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:  # Rectangle has 4 vertices
                rectangular_contours += 1

        # Reflection detection (screens often have reflections)
        reflection_score = self._detect_reflections(image)

        # Calculate final screen detection confidence
        is_screen = (
                screen_pattern_score > 0.001 or
                rectangular_contours > 2 or
                reflection_score > 0.3
        )

        confidence = max(
            screen_pattern_score * 100,
            min(rectangular_contours * 0.2, 1.0),
            reflection_score
        )

        return {
            'is_screen': bool(is_screen),  # Ensure boolean
            'confidence': float(min(confidence, 1.0)),  # Ensure float
            'pattern_score': float(screen_pattern_score),
            'rectangular_edges': int(rectangular_contours),
            'reflection_score': float(reflection_score)
        }

    def _detect_reflections(self, image: np.ndarray) -> float:
        """
        Detect reflections commonly found in screen photos
        """
        # Convert to LAB color space for better light detection
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]

        # Find bright spots that could be reflections
        _, bright_spots = cv2.threshold(l_channel, 220, 255, cv2.THRESH_BINARY)

        # Calculate percentage of bright spots
        reflection_percentage = np.sum(bright_spots > 0) / (image.shape[0] * image.shape[1])

        return min(reflection_percentage * 10, 1.0)

    def _detect_manipulation(self, image: np.ndarray, image_path: str) -> Dict:
        """
        Detect image manipulation using Error Level Analysis (ELA)
        and noise pattern analysis
        """
        # Error Level Analysis
        temp_path = "temp_ela_check.jpg"
        cv2.imwrite(temp_path, image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        compressed = cv2.imread(temp_path)
        os.remove(temp_path)

        # Calculate difference between original and recompressed
        ela_diff = cv2.absdiff(image, compressed)
        ela_score = np.mean(ela_diff)

        # Noise consistency check
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        noise_levels = []

        # Sample different regions for noise analysis
        h, w = gray.shape
        region_size = 50

        for i in range(0, h - region_size, 100):
            for j in range(0, w - region_size, 100):
                region = gray[i:i + region_size, j:j + region_size]
                noise_levels.append(np.std(region))

        # Check noise consistency
        if noise_levels:
            noise_consistency = 1.0 - (np.std(noise_levels) / (np.mean(noise_levels) + 1e-6))
        else:
            noise_consistency = 0.5

        # Check for copy-paste artifacts
        copy_paste_score = self._detect_copy_paste(image)

        # Calculate manipulation confidence
        is_manipulated = (
                ela_score > 30 or
                noise_consistency < 0.6 or
                copy_paste_score > 0.3
        )

        confidence = max(
            min(ela_score / 50, 1.0),
            1.0 - noise_consistency,
            copy_paste_score
        )

        return {
            'is_manipulated': bool(is_manipulated),
            'confidence': float(min(confidence, 1.0)),
            'ela_score': float(ela_score),
            'noise_consistency': float(noise_consistency),
            'copy_paste_score': float(copy_paste_score)
        }

    def _detect_copy_paste(self, image: np.ndarray) -> float:
        """
        Detect copy-paste artifacts using template matching
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Sample small patches and look for duplicates
        patch_size = 32
        max_matches = 0

        for _ in range(5):  # Sample 5 random patches
            # Random patch location
            y = np.random.randint(0, max(1, h - patch_size))
            x = np.random.randint(0, max(1, w - patch_size))

            template = gray[y:y + patch_size, x:x + patch_size]

            # Search for this template in the image
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)

            # Count strong matches (excluding the original location)
            threshold = 0.95
            loc = np.where(result >= threshold)
            matches = len(loc[0])

            if matches > 1:  # More than just the original location
                max_matches = max(max_matches, matches - 1)

        return min(max_matches * 0.2, 1.0)

    def _validate_ecosystem_content(self, image: np.ndarray) -> Dict:
        """
        Validate that the image contains coastal ecosystem elements
        (vegetation, water, natural landscapes)
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Detect vegetation (greens)
        vegetation_mask = cv2.inRange(hsv, self.vegetation_lower, self.vegetation_upper)
        vegetation_percentage = np.sum(vegetation_mask > 0) / (image.shape[0] * image.shape[1])

        # Detect water (blues)
        water_mask = cv2.inRange(hsv, self.water_lower, self.water_upper)
        water_percentage = np.sum(water_mask > 0) / (image.shape[0] * image.shape[1])

        # Detect sand/soil (browns and tans)
        sand_lower = np.array([10, 30, 30])
        sand_upper = np.array([25, 255, 200])
        sand_mask = cv2.inRange(hsv, sand_lower, sand_upper)
        sand_percentage = np.sum(sand_mask > 0) / (image.shape[0] * image.shape[1])

        # Check for human faces (shouldn't be the main subject)
        faces = self.face_cascade.detectMultiScale(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            scaleFactor=1.1,
            minNeighbors=5
        )
        has_faces = len(faces) > 0

        # Check for text/documents (shouldn't be present)
        has_text = self._detect_text_regions(image)

        # Check for indoor elements
        is_indoor = self._detect_indoor_elements(image)

        # Calculate ecosystem confidence
        ecosystem_score = (
                vegetation_percentage * 0.4 +
                water_percentage * 0.3 +
                sand_percentage * 0.2 +
                (0.1 if not has_faces else 0) +
                (0.0 if has_text else 0.1) +
                (0.0 if is_indoor else 0.1)
        )

        is_ecosystem = (
                (vegetation_percentage > 0.1 or water_percentage > 0.1) and
                not is_indoor and
                not has_text
        )

        return {
            'is_ecosystem': bool(is_ecosystem),
            'ecosystem_confidence': float(min(ecosystem_score * 2, 1.0)),
            'vegetation_percentage': float(vegetation_percentage),
            'water_percentage': float(water_percentage),
            'sand_percentage': float(sand_percentage),
            'has_faces': bool(has_faces),
            'has_text': bool(has_text),
            'is_indoor': bool(is_indoor)
        }

    def _detect_text_regions(self, image: np.ndarray) -> bool:
        """
        Detect if image contains significant text regions
        (documents, screens with text, etc.)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Use morphological operations to find text-like regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        morph = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

        _, thresh = cv2.threshold(morph, 30, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Count horizontal rectangular regions (likely text)
        text_regions = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / (h + 1e-6)
            if 2 < aspect_ratio < 20 and h > 10:  # Horizontal rectangles
                text_regions += 1

        return text_regions > 5

    def _detect_indoor_elements(self, image: np.ndarray) -> bool:
        """
        Detect indoor elements like walls, furniture, screens
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect straight lines (walls, furniture edges)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

        if lines is not None:
            # Count vertical and horizontal lines
            vertical_lines = 0
            horizontal_lines = 0

            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

                if angle < 10 or angle > 170:  # Horizontal
                    horizontal_lines += 1
                elif 80 < angle < 100:  # Vertical
                    vertical_lines += 1

            # Many perpendicular lines suggest indoor environment
            if vertical_lines > 5 and horizontal_lines > 5:
                return True

        # Check for uniform colored regions (walls)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]

        # Low saturation large areas suggest walls
        low_sat_percentage = np.sum(saturation < 30) / (image.shape[0] * image.shape[1])

        return low_sat_percentage > 0.4

    def _check_image_quality(self, image: np.ndarray) -> Dict:
        """
        Check image quality: blur, darkness, resolution
        """
        h, w = image.shape[:2]

        # Check resolution
        resolution_ok = w >= self.MIN_RESOLUTION[0] and h >= self.MIN_RESOLUTION[1]
        resolution_score = min(
            (w * h) / (self.MIN_RESOLUTION[0] * self.MIN_RESOLUTION[1]),
            1.0
        )

        # Check blur using Laplacian variance
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_blurry = laplacian_var < self.BLUR_THRESHOLD
        blur_score = min(laplacian_var / self.BLUR_THRESHOLD, 1.0)

        # Check darkness
        mean_brightness = np.mean(gray)
        is_dark = mean_brightness < self.DARKNESS_THRESHOLD
        brightness_score = min(mean_brightness / 127, 1.0)

        # Check contrast
        contrast = np.std(gray)
        contrast_score = min(contrast / 50, 1.0)

        # Overall quality score
        quality_score = (
                resolution_score * 0.2 +
                blur_score * 0.3 +
                brightness_score * 0.25 +
                contrast_score * 0.25
        )

        return {
            'quality_score': float(quality_score),
            'resolution': (int(w), int(h)),
            'resolution_ok': bool(resolution_ok),
            'is_blurry': bool(is_blurry),
            'blur_score': float(blur_score),
            'is_dark': bool(is_dark),
            'brightness_score': float(brightness_score),
            'contrast_score': float(contrast_score)
        }

    def _extract_metadata(self, image_path: str) -> Dict:
        """
        Extract EXIF metadata from image
        """
        metadata = {
            'has_exif': False,
            'camera_make': None,
            'camera_model': None,
            'datetime': None,
            'gps_latitude': None,
            'gps_longitude': None,
            'file_size': os.path.getsize(image_path)
        }

        try:
            image = Image.open(image_path)
            exifdata = image.getexif()

            if exifdata:
                metadata['has_exif'] = True

                # Extract standard EXIF data
                for tag_id in exifdata:
                    tag = TAGS.get(tag_id, tag_id)
                    data = exifdata.get(tag_id)

                    if tag == 'Make':
                        metadata['camera_make'] = str(data)
                    elif tag == 'Model':
                        metadata['camera_model'] = str(data)
                    elif tag == 'DateTime':
                        metadata['datetime'] = str(data)

                # Extract GPS data if available - FIXED
                if 34853 in exifdata:  # GPSInfo tag
                    try:
                        gps_info = exifdata[34853]
                        # Check if gps_info is a dictionary
                        if isinstance(gps_info, dict):
                            metadata['gps_latitude'] = self._extract_gps_coordinate(gps_info.get(2), gps_info.get(1))
                            metadata['gps_longitude'] = self._extract_gps_coordinate(gps_info.get(4), gps_info.get(3))
                        else:
                            logger.debug(f"GPSInfo is not a dictionary for {image_path}: {type(gps_info)}")
                    except Exception as e:
                        logger.debug(f"Error extracting GPS data: {e}")

        except (IOError, OSError, ValueError) as e:
            logger.warning(f"Could not extract metadata from {image_path}: {e}")

        return metadata

    def _extract_gps_coordinate(self, coord_data: Union[Tuple, None], ref: Union[str, None]) -> Union[float, None]:
        """
        Convert GPS coordinates from EXIF format to decimal

        Args:
            coord_data: GPS coordinate data tuple
            ref: Reference direction (N/S/E/W)

        Returns:
            Decimal coordinate or None
        """
        if not coord_data or not ref:
            return None

        try:
            degrees = float(coord_data[0])
            minutes = float(coord_data[1])
            seconds = float(coord_data[2])

            decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)

            if ref in ['S', 'W']:
                decimal = -decimal

            return decimal
        except (TypeError, IndexError, ValueError):
            return None

    def _verify_metadata(self, metadata: Dict) -> Dict:
        """
        Verify metadata validity and consistency
        """
        validity_score = 0.0
        checks = {
            'has_exif': bool(metadata['has_exif']),
            'has_camera_info': bool(metadata['camera_make'] or metadata['camera_model']),
            'has_timestamp': bool(metadata['datetime']),
            'has_gps': bool(metadata['gps_latitude'] and metadata['gps_longitude']),
            'timestamp_recent': False
        }

        # Check if timestamp is recent (within 24 hours)
        if metadata['datetime']:
            try:
                photo_time = datetime.strptime(metadata['datetime'], '%Y:%m:%d %H:%M:%S')
                time_diff = abs((datetime.now() - photo_time).total_seconds())
                checks['timestamp_recent'] = time_diff < 86400  # 24 hours
            except (ValueError, TypeError):
                pass

        # Calculate validity score
        weights = {
            'has_exif': 0.3,
            'has_camera_info': 0.2,
            'has_timestamp': 0.2,
            'has_gps': 0.2,
            'timestamp_recent': 0.1
        }

        validity_score = sum(
            checks[key] * weights[key] for key in weights
        )

        return {
            'validity_score': float(validity_score),
            'checks': checks
        }

    def _check_duplicate(self, image: np.ndarray) -> Dict:
        """
        Check if image is a duplicate using perceptual hashing
        """
        # Simple perceptual hash
        resized = cv2.resize(image, (8, 8))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Calculate hash
        avg = np.mean(gray)
        hash_binary = (gray > avg).flatten()
        image_hash = ''.join(['1' if b else '0' for b in hash_binary])

        # Check against previously seen hashes
        is_duplicate = image_hash in self.seen_hashes

        # Store hash
        self.seen_hashes.add(image_hash)

        return {
            'is_duplicate': bool(is_duplicate),
            'hash': str(image_hash[:16])  # Store only first 16 chars for display
        }

    def _calculate_batch_score(self, verified_photos: List[Dict], rejected_photos: List[Dict]) -> float:
        """
        Calculate overall score for the batch of photos
        """
        if not verified_photos:
            return 0.0

        # Average score of verified photos
        avg_score = sum(p['authenticity_score'] for p in verified_photos) / len(verified_photos)

        # Penalty for rejected photos
        rejection_penalty = len(rejected_photos) * 0.1

        # Bonus for having multiple verified photos
        diversity_bonus = min(len(verified_photos) * 0.05, 0.2)

        final_score = max(0, min(1, avg_score - rejection_penalty + diversity_bonus))

        return final_score

    def generate_verification_report(self, verification_result: Dict) -> str:
        """
        Generate a detailed HTML report of the verification
        """
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>AquaChain Photo Verification Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .header {{ background: #2196F3; color: white; padding: 20px; border-radius: 5px; }}
                .status {{ font-size: 24px; font-weight: bold; margin: 20px 0; }}
                .approved {{ color: #4CAF50; }}
                .rejected {{ color: #F44336; }}
                .section {{ background: white; padding: 15px; margin: 10px 0; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .photo-result {{ margin: 10px 0; padding: 10px; border-left: 3px solid #2196F3; }}
                .score {{ font-weight: bold; color: #2196F3; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background: #f0f0f0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>AquaChain Photo Verification Report</h1>
                <p>Generated: {timestamp}</p>
            </div>

            <div class="status {status_class}">
                Status: {status}
            </div>

            <div class="section">
                <h2>Summary</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total Photos</td><td>{total_photos}</td></tr>
                    <tr><td>Verified Photos</td><td>{verified_photos}</td></tr>
                    <tr><td>Rejected Photos</td><td>{rejected_photos}</td></tr>
                    <tr><td>Verification Score</td><td class="score">{verification_score:.2%}</td></tr>
                </table>
            </div>

            <div class="section">
                <h2>Detailed Results</h2>
                {detailed_results}
            </div>
        </body>
        </html>
        """

        # Build detailed results HTML
        detailed_html = ""

        if 'detailed_results' in verification_result:
            # Verified photos
            if verification_result['detailed_results']['verified']:
                detailed_html += "<h3>✅ Verified Photos</h3>"
                for photo in verification_result['detailed_results']['verified']:
                    detailed_html += f"""
                    <div class="photo-result">
                        <strong>{photo['filename']}</strong><br>
                        Authenticity Score: <span class="score">{photo['authenticity_score']:.2%}</span><br>
                        Content: {photo['checks']['content_validation']['ecosystem_confidence']:.2%} ecosystem confidence<br>
                        Quality: {photo['checks']['quality_check']['quality_score']:.2%}<br>
                    </div>
                    """

            # Rejected photos
            if verification_result['detailed_results']['rejected']:
                detailed_html += "<h3>❌ Rejected Photos</h3>"
                for photo in verification_result['detailed_results']['rejected']:
                    detailed_html += f"""
                    <div class="photo-result">
                        <strong>{photo['filename']}</strong><br>
                        Reason: {photo.get('reason', 'Failed verification')}<br>
                    </div>
                    """

        # Fill in the template
        html = html_template.format(
            timestamp=verification_result.get('timestamp', datetime.now().isoformat()),
            status=verification_result['status'],
            status_class='approved' if verification_result['status'] == 'APPROVED' else 'rejected',
            total_photos=verification_result.get('total_photos', 0),
            verified_photos=verification_result.get('verified_photos', 0),
            rejected_photos=verification_result.get('rejected_photos', 0),
            verification_score=verification_result.get('verification_score', 0),
            detailed_results=detailed_html
        )

        return html


if __name__ == "__main__":
    # Define the path to your test images folder
    test_image_dir = 'test_images'

    # Check if directory exists
    if not os.path.exists(test_image_dir):
        print(f"Creating directory: {test_image_dir}")
        os.makedirs(test_image_dir)
        print("Please add some test images to the 'test_images' directory.")
        exit()

    # Get a list of all image paths in the directory
    image_paths = [os.path.join(test_image_dir, f) for f in os.listdir(test_image_dir) if
                   f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_paths:
        print("No images found in the 'test_images' directory. Please add some.")
    else:
        # Create an instance of the verifier
        verifier = PhotoVerifier()

        # Run the verification on the batch of photos
        results = verifier.verify_photo_batch(image_paths)

        # Print the results in a readable format
        print("\n" + "=" * 80)
        print(" " * 20 + "🌊 AQUACHAIN PHOTO VERIFICATION REPORT 🌊")
        print("=" * 80)

        # Overall Summary
        print("\n📊 OVERALL SUMMARY")
        print("-" * 80)
        print(f"  Status: {results['status']}")
        print(f"  Total Photos Analyzed: {results['total_photos']}")
        print(f"  Photos Verified: {results['verified_photos']}")
        print(f"  Photos Rejected: {results['rejected_photos']}")
        print(f"  Overall Verification Score: {results['verification_score']:.2%}")

        # Detailed Analysis for Each Photo
        print("\n" + "=" * 80)
        print(" " * 25 + "DETAILED PHOTO ANALYSIS")
        print("=" * 80)

        all_photos = []
        if 'detailed_results' in results:
            all_photos.extend([(photo, True) for photo in results['detailed_results'].get('verified', [])])
            all_photos.extend([(photo, False) for photo in results['detailed_results'].get('rejected', [])])

        for idx, (photo, is_verified) in enumerate(all_photos, 1):
            status_icon = "✅" if is_verified else "❌"
            status_text = "VERIFIED" if is_verified else "REJECTED"

            print(f"\n{'-' * 80}")
            print(f"PHOTO {idx}/{len(all_photos)}: {photo['filename']}")
            print(f"STATUS: {status_icon} {status_text}")
            if 'authenticity_score' in photo:
                print(f"AUTHENTICITY SCORE: {photo['authenticity_score']:.2%}")
            if not is_verified and 'reason' in photo:
                print(f"REJECTION REASON: {photo['reason']}")
            print(f"{'-' * 80}")

            if 'checks' in photo:
                # 1. Screen Detection Check
                print("\n1️⃣  SCREEN DETECTION CHECK")
                print("   " + "-" * 35)
                screen_check = photo['checks'].get('screen_detection', {})
                print(f"   Is Screen Photo: {screen_check.get('is_screen', 'N/A')}")
                print(f"   Detection Confidence: {screen_check.get('confidence', 0):.2%}")
                print(f"   Pattern Score: {screen_check.get('pattern_score', 0):.4f}")
                print(f"   Rectangular Edges Found: {screen_check.get('rectangular_edges', 0)}")
                print(f"   Reflection Score: {screen_check.get('reflection_score', 0):.2%}")

                # 2. Manipulation Detection Check
                print("\n2️⃣  MANIPULATION DETECTION CHECK")
                print("   " + "-" * 35)
                manip_check = photo['checks'].get('manipulation_detection', {})
                print(f"   Is Manipulated: {manip_check.get('is_manipulated', 'N/A')}")
                print(f"   Detection Confidence: {manip_check.get('confidence', 0):.2%}")
                print(f"   ELA Score: {manip_check.get('ela_score', 0):.2f}")
                print(f"   Noise Consistency: {manip_check.get('noise_consistency', 0):.2%}")
                print(f"   Copy-Paste Score: {manip_check.get('copy_paste_score', 0):.2%}")

                # 3. Content Validation Check
                print("\n3️⃣  ECOSYSTEM CONTENT VALIDATION")
                print("   " + "-" * 35)
                content_check = photo['checks'].get('content_validation', {})
                print(f"   Is Ecosystem: {content_check.get('is_ecosystem', 'N/A')}")
                print(f"   Ecosystem Confidence: {content_check.get('ecosystem_confidence', 0):.2%}")
                print(f"   Vegetation Detected: {content_check.get('vegetation_percentage', 0):.2%}")
                print(f"   Water Detected: {content_check.get('water_percentage', 0):.2%}")
                print(f"   Sand/Soil Detected: {content_check.get('sand_percentage', 0):.2%}")
                print(f"   Human Faces Present: {content_check.get('has_faces', 'N/A')}")
                print(f"   Text Detected: {content_check.get('has_text', 'N/A')}")
                print(f"   Indoor Elements: {content_check.get('is_indoor', 'N/A')}")

                # 4. Image Quality Check
                print("\n4️⃣  IMAGE QUALITY CHECK")
                print("   " + "-" * 35)
                quality_check = photo['checks'].get('quality_check', {})
                print(f"   Overall Quality Score: {quality_check.get('quality_score', 0):.2%}")
                resolution = quality_check.get('resolution', (0, 0))
                print(f"   Resolution: {resolution[0]}x{resolution[1]} pixels")
                print(f"   Resolution Adequate: {quality_check.get('resolution_ok', 'N/A')}")
                print(f"   Is Blurry: {quality_check.get('is_blurry', 'N/A')}")
                print(f"   Blur Score: {quality_check.get('blur_score', 0):.2%}")
                print(f"   Is Too Dark: {quality_check.get('is_dark', 'N/A')}")
                print(f"   Brightness Score: {quality_check.get('brightness_score', 0):.2%}")
                print(f"   Contrast Score: {quality_check.get('contrast_score', 0):.2%}")

                # 5. Metadata Check
                print("\n5️⃣  METADATA VERIFICATION")
                print("   " + "-" * 35)
                metadata_check = photo['checks'].get('metadata_check', {})
                print(f"   Validity Score: {metadata_check.get('validity_score', 0):.2%}")
                if 'checks' in metadata_check:
                    meta_checks = metadata_check['checks']
                    print(f"   Has EXIF Data: {meta_checks.get('has_exif', 'N/A')}")
                    print(f"   Has Camera Info: {meta_checks.get('has_camera_info', 'N/A')}")
                    print(f"   Has Timestamp: {meta_checks.get('has_timestamp', 'N/A')}")
                    print(f"   Has GPS Data: {meta_checks.get('has_gps', 'N/A')}")
                    print(f"   Timestamp Recent: {meta_checks.get('timestamp_recent', 'N/A')}")

                # 6. Duplicate Check
                print("\n6️⃣  DUPLICATE CHECK")
                print("   " + "-" * 35)
                dup_check = photo['checks'].get('duplicate_check', {})
                print(f"   Is Duplicate: {dup_check.get('is_duplicate', 'N/A')}")
                print(f"   Perceptual Hash: {dup_check.get('hash', 'N/A')}")

                # Score Summary for this photo
                if 'scores' in photo:
                    print("\n📊 INDIVIDUAL SCORES BREAKDOWN")
                    print("   " + "-" * 35)
                    scores = photo['scores']
                    print(f"   Screen Score: {scores.get('screen_score', 0):.2%} (weight: 25%)")
                    print(f"   Manipulation Score: {scores.get('manipulation_score', 0):.2%} (weight: 25%)")
                    print(f"   Content Score: {scores.get('content_score', 0):.2%} (weight: 20%)")
                    print(f"   Quality Score: {scores.get('quality_score', 0):.2%} (weight: 15%)")
                    print(f"   Metadata Score: {scores.get('metadata_score', 0):.2%} (weight: 15%)")
                    print(f"   " + "-" * 35)
                    if 'authenticity_score' in photo:
                        print(f"   FINAL AUTHENTICITY SCORE: {photo['authenticity_score']:.2%}")
                        print(f"   THRESHOLD REQUIRED: {verifier.AUTHENTICITY_THRESHOLD:.2%}")
                        print(
                            f"   PASS/FAIL: {'PASS ✅' if photo['authenticity_score'] >= verifier.AUTHENTICITY_THRESHOLD else 'FAIL ❌'}")

        print("\n" + "=" * 80)
        print(" " * 30 + "END OF DETAILED ANALYSIS")
        print("=" * 80)

        # Generate and save HTML report
        print("\n📄 Generating HTML report...")
        html_report = verifier.generate_verification_report(results)
        report_filename = f"verification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

        # Create reports directory if it doesn't exist
        if not os.path.exists("reports"):
            os.makedirs("reports")

        report_path = os.path.join("reports", report_filename)
        # Fix encoding issue by specifying UTF-8 encoding
        with open(report_path, "w", encoding='utf-8') as f:
            f.write(html_report)
        print(f"✅ Report saved to: {report_path}")

        # Save JSON results
        json_filename = f"verification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        json_path = os.path.join("reports", json_filename)

        # Ensure all values are JSON serializable - also use UTF-8 encoding
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"💾 JSON results saved to: {json_path}")

        print("\n" + "=" * 80)
        print(" " * 25 + "✅ VERIFICATION COMPLETE!")
        print("=" * 80)