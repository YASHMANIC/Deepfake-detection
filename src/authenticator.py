import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
import magic
import hashlib
import os

class MediaAuthenticator:
    def __init__(self):
        self.mime = magic.Magic(mime=True)

    def _check_metadata(self, image_path):
        """Analyze image metadata for signs of manipulation"""
        try:
            with Image.open(image_path) as img:
                exif = img._getexif()
                if not exif:
                    return {'metadata_present': False, 'suspicious': True}
                
                # Convert EXIF data to readable format
                exif_data = {}
                for tag_id in exif:
                    tag = TAGS.get(tag_id, tag_id)
                    data = exif.get(tag_id)
                    exif_data[tag] = data
                
                return {
                    'metadata_present': True,
                    'suspicious': False,
                    'details': exif_data
                }
        except Exception as e:
            print(f"Error checking metadata: {str(e)}")
            return {'metadata_present': False, 'suspicious': True}

    def _check_compression(self, image_path):
        """Check for multiple compression artifacts"""
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            dct = cv2.dct(np.float32(img))
            compression_score = np.mean(np.abs(dct))
            return {'compression_score': compression_score}
        except Exception as e:
            print(f"Error checking compression: {str(e)}")
            return {'compression_score': 0}

    def _check_noise_patterns(self, image_path):
        """Analyze noise patterns for inconsistencies"""
        try:
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            noise = cv2.Laplacian(gray, cv2.CV_64F).var()
            return {'noise_level': noise}
        except Exception as e:
            print(f"Error checking noise patterns: {str(e)}")
            return {'noise_level': 0}

    def _analyze_noise_patterns(self, image_path):
        """Analyze image noise patterns for inconsistencies"""
        try:
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply noise analysis
            noise_map = cv2.fastNlMeansDenoising(gray)
            noise = cv2.absdiff(gray, noise_map)
            
            # Calculate noise statistics and convert to Python native types
            mean_noise = float(np.mean(noise))
            std_noise = float(np.std(noise))
            
            return {
                'mean_noise': mean_noise,
                'std_noise': std_noise,
                'suspicious': bool(std_noise > 30)  # Convert to Python bool
            }
        except Exception as e:
            return {'error': str(e)}

    def _calculate_hash(self, file_path):
        """Calculate file hash for integrity checking"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def authenticate(self, media_path):
        """
        Authenticate a media file by analyzing various aspects.
        Returns a dictionary with authentication results.
        """
        try:
            if not os.path.exists(media_path):
                raise FileNotFoundError(f"Media file not found: {media_path}")

            # Get file type
            mime_type = self.mime.from_file(media_path)
            
            # Perform various checks
            metadata_result = self._check_metadata(media_path)
            compression_result = self._check_compression(media_path)
            noise_result = self._check_noise_patterns(media_path)
            
            # Calculate authenticity score based on various factors
            score = 0.0
            total_factors = 3
            
            # Factor 1: Metadata presence and validity
            if metadata_result.get('metadata_present', False) and not metadata_result.get('suspicious', True):
                score += 1.0
            
            # Factor 2: Compression analysis
            compression_score = compression_result.get('compression_score', 0)
            if compression_score > 0:
                score += min(compression_score / 1000, 1.0)
            
            # Factor 3: Noise pattern analysis
            noise_level = noise_result.get('noise_level', 0)
            if noise_level > 0:
                score += min(noise_level / 100, 1.0)
            
            # Calculate final score
            final_score = score / total_factors
            
            return {
                'authenticity_score': final_score,
                'is_authentic': final_score > 0.6,
                'mime_type': mime_type,
                'metadata_analysis': metadata_result,
                'compression_analysis': compression_result,
                'noise_analysis': noise_result
            }
            
        except Exception as e:
            print(f"Error in media authentication: {str(e)}")
            return {
                'authenticity_score': 0.0,
                'is_authentic': False,
                'error': str(e)
            }

    def verify(self, media_path):
        """Verify the authenticity of the media file"""
        results = {
            'file_type': self.mime.from_file(media_path),
            'file_hash': self._calculate_hash(media_path),
            'metadata_analysis': self._check_metadata(media_path),
            'noise_analysis': self._analyze_noise_patterns(media_path)
        }
        
        # Determine overall authenticity score
        suspicious_factors = [
            results['metadata_analysis'].get('suspicious', False),
            results['noise_analysis'].get('suspicious', False)
        ]
        
        authenticity_score = float(1.0 - (sum(suspicious_factors) / len(suspicious_factors)))
        results['authenticity_score'] = authenticity_score
        results['is_authentic'] = bool(authenticity_score > 0.7)
        
        return results
