"""
Verification service that integrates PhotoVerifier with your existing FastAPI backend
"""

import os
import sys
import asyncio
import tempfile
import requests
from typing import List, Dict, Any
import logging
from datetime import datetime

# Add the verification directory to Python path
sys.path.append(os.path.dirname(__file__))

from photo_verification import PhotoVerifier

logger = logging.getLogger(__name__)


class AquaChainVerificationService:
    """
    Service to integrate OpenCV verification with your existing app
    """

    def __init__(self):
        self.verifier = PhotoVerifier()
        self.temp_dir = tempfile.mkdtemp()

    async def verify_report_photos(self,
                                   report_id: str,
                                   cloudinary_urls: List[str],
                                   project_name: str,
                                   gps: str) -> Dict[str, Any]:
        """
        Download photos from Cloudinary and run verification

        Args:
            report_id: Your existing report ID
            cloudinary_urls: List of Cloudinary URLs from your existing system
            project_name: Project name from the report
            gps: GPS coordinates from the report

        Returns:
            Verification result dictionary
        """
        try:
            logger.info(f"Starting verification for report {report_id}")

            # Download images from Cloudinary to temp directory
            temp_image_paths = []
            for i, url in enumerate(cloudinary_urls):
                temp_path = os.path.join(self.temp_dir, f"{report_id}_photo_{i}.jpg")

                # Download image
                response = requests.get(url)
                response.raise_for_status()

                with open(temp_path, 'wb') as f:
                    f.write(response.content)

                temp_image_paths.append(temp_path)

            # Run your OpenCV verification
            verification_result = self.verifier.verify_photo_batch(temp_image_paths)

            # Add project metadata
            verification_result.update({
                'report_id': report_id,
                'project_name': project_name,
                'gps_coordinates': gps,
                'verification_timestamp': datetime.now().isoformat(),
                'photo_urls': cloudinary_urls
            })

            # Clean up temp files
            for path in temp_image_paths:
                if os.path.exists(path):
                    os.remove(path)

            logger.info(f"Verification completed for report {report_id}: {verification_result['status']}")

            return verification_result

        except Exception as e:
            logger.error(f"Verification failed for report {report_id}: {e}")
            return {
                'status': 'ERROR',
                'error': str(e),
                'report_id': report_id,
                'verification_timestamp': datetime.now().isoformat()
            }

    def cleanup(self):
        """Clean up temporary directory"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)