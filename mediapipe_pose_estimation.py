import cv2
import mediapipe as mp
import numpy as np
import os 
import time
import json
import math
from pathlib import Path
import argparse
from typing import Tuple, Optional
import logging

from utils.camera import load_cam_infos
from utils.image import undistort_image

DEFAULT_CONF = 0.35

# Enhancement configuration
class EnhancementConfig:
    def __init__(self):
        self.max_alpha = 2.0
        self.alpha_step = 0.3
        self.min_beta = -75
        self.max_beta = 76
        self.beta_step = 25
        # Gamma correction parameters
        self.gamma_values = [0.5, 0.7, 0.8, 1.2, 1.5, 2.0]  # Range of gamma values to try
        self.use_adaptive_gamma = True  # Use exposure analysis to suggest gamma

def setup_logging(output_path):
    """Setup logging configuration"""
    log_folder = os.path.join(output_path, 'log')
    logfile = os.path.join(log_folder, 'output.log')
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)
    
def hand_confidence(pose_landmarks, hand="left"):
    if hand == "left":
        indices = [
            mp.solutions.pose.PoseLandmark.LEFT_WRIST,
            mp.solutions.pose.PoseLandmark.LEFT_PINKY,
            mp.solutions.pose.PoseLandmark.LEFT_INDEX,
            mp.solutions.pose.PoseLandmark.LEFT_THUMB,
        ]
    else:  # right hand
        indices = [
            mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
            mp.solutions.pose.PoseLandmark.RIGHT_PINKY,
            mp.solutions.pose.PoseLandmark.RIGHT_INDEX,
            mp.solutions.pose.PoseLandmark.RIGHT_THUMB,
        ]

    visibilities = [pose_landmarks.landmark[i].visibility for i in indices]
    return visibilities


def infer_full_hand_visibility(pose_landmarks, hand="left"):
    """
    Infer approximate visibility for all 21 hand keypoints
    based on the 4 hand-related landmarks from MediaPipe Pose.
    """
    # Select indices for the hand
    if hand == "left":
        wrist = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST].visibility
        thumb = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_THUMB].visibility
        index = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_INDEX].visibility
        pinky = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_PINKY].visibility
    else:
        wrist = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST].visibility
        thumb = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_THUMB].visibility
        index = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_INDEX].visibility
        pinky = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_PINKY].visibility

    middle = 0.7 * index + 0.3 * pinky
    ring = 0.3 * index + 0.7 * pinky

    # Map to 21 hand keypoints (MediaPipe Hands indexing)
    visibility = [
        wrist,         # 0: wrist
        thumb, thumb, thumb, thumb,   # 1-4: thumb joints
        index, index, index, index,   # 5-8: index joints
        middle, middle, middle, middle, # 9-12: middle joints
        ring, ring, ring, ring,       # 13-16: ring joints
        pinky, pinky, pinky, pinky    # 17-20: pinky joints
    ]

    return visibility


# Example: Adding CLAHE enhancement
def enhance_with_clahe(image, clip_limit=2.0, grid_size=(8,8)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# Example: Multi-scale detection
def detect_multi_scale(holistic, image, scales=[0.8, 1.0, 1.2]):
    best_results = None
    max_hands = 0
    
    for scale in scales:
        if scale != 1.0:
            h, w = image.shape[:2]
            resized = cv2.resize(image, (int(w*scale), int(h*scale)))
            results = holistic.process(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
            # Scale landmarks back to original size
        else:
            results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        hands_count = sum([
            1 if results.left_hand_landmarks else 0,
            1 if results.right_hand_landmarks else 0
        ])
        
        if hands_count > max_hands:
            max_hands = hands_count
            best_results = results
    
    return best_results


def apply_gamma_correction(image, gamma):
    """
    Apply gamma correction to an image.
    
    Args:
        image: Input image (BGR format)
        gamma: Gamma value
               - gamma < 1.0: Brightens image (good for dark/underexposed images)
               - gamma > 1.0: Darkens image (good for bright/overexposed images)
               - gamma = 1.0: No change
    
    Returns:
        Gamma-corrected image
    """
    # Build a lookup table mapping pixel values [0, 255] to their adjusted gamma values
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    
    # Apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def analyze_image_exposure(image):
    """
    Analyze image exposure to determine if gamma correction is needed.
    
    Returns:
        - exposure_level: 'dark', 'normal', 'bright'
        - suggested_gamma: recommended gamma value
    """
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    # Calculate mean brightness
    mean_brightness = np.mean(gray)
    
    # Calculate percentage of pixels in dark (0-85), mid (85-170), bright (170-255) ranges
    total_pixels = gray.shape[0] * gray.shape[1]
    dark_pixels = np.sum(hist[0:85]) / total_pixels
    bright_pixels = np.sum(hist[170:256]) / total_pixels
    
    # Determine exposure level and suggest gamma
    if mean_brightness < 80 or dark_pixels > 0.4:
        return 'dark', 0.7  # Brighten with gamma < 1
    elif mean_brightness > 175 or bright_pixels > 0.4:
        return 'bright', 1.3  # Darken with gamma > 1
    else:
        return 'normal', 1.0  # No correction needed


def detect_hands_with_enhancement(holistic, image: np.ndarray, cam_idx: int, config: EnhancementConfig) -> Tuple[any, int]:
    """Detect hands with brightness/contrast enhancement, gamma correction, and histogram equalization"""
    # Try original image first
    rgb_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb_image)
    
    # Count hands detected
    hands_detected = 0
    if results.left_hand_landmarks:
        hands_detected += 1
    if results.right_hand_landmarks:
        hands_detected += 1
    
    if hands_detected == 2:
        return results, 0, None
    
    params = None
    enhancement_attempts = 0
    best_results = results
    best_hands = hands_detected
    
    # Try gamma correction first (often more effective than linear adjustments)
    if config.use_adaptive_gamma:
        exposure_level, suggested_gamma = analyze_image_exposure(image)
        gamma_values = [suggested_gamma] + [g for g in config.gamma_values if g != suggested_gamma]
    else:
        gamma_values = config.gamma_values
    
    for gamma in gamma_values:
        enhancement_attempts += 1
        gamma_corrected = apply_gamma_correction(image.copy(), gamma)
        gamma_rgb = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2RGB)
        gamma_rgb = enhance_with_clahe(gamma_rgb)
        gamma_results = holistic.process(gamma_rgb)
        
        # Count hands detected with gamma correction
        gamma_hands = 0
        if gamma_results.left_hand_landmarks:
            gamma_hands += 1
        if gamma_results.right_hand_landmarks:
            gamma_hands += 1
        
        # If we found both hands with gamma correction, return immediately
        if gamma_hands == 2:
            return gamma_results, enhancement_attempts, ('gamma', gamma)
        
        # Keep track of best result so far
        if gamma_hands > best_hands:
            best_results = gamma_results
            best_hands = gamma_hands
            params = ('gamma', gamma)
    
    # If gamma correction didn't achieve perfect results, try traditional brightness/contrast
    for alpha in np.arange(1.0, config.max_alpha, config.alpha_step):
        for beta in range(config.min_beta, config.max_beta, config.beta_step):
            enhancement_attempts += 1
            enhanced = np.clip(image.copy().astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
            enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            enhanced_rgb = enhance_with_clahe(enhanced_rgb)
            enhanced_results = holistic.process(enhanced_rgb)
            
            # Count hands detected in enhanced image
            enhanced_hands = 0
            if enhanced_results.left_hand_landmarks:
                enhanced_hands += 1
            if enhanced_results.right_hand_landmarks:
                enhanced_hands += 1
            
            # If we found both hands with enhancement, return enhanced results
            if enhanced_hands == 2:
                return enhanced_results, enhancement_attempts, ('linear', alpha, beta)
            
            # If enhanced result is better than current best, keep it
            if enhanced_hands > best_hands:
                best_results = enhanced_results
                best_hands = enhanced_hands
                params = ('linear', alpha, beta)
    
    return best_results, enhancement_attempts, params


def parse_args():
    parser = argparse.ArgumentParser(description='MediaPipe Holistic (Pose + Hands) detection')
    parser.add_argument('--save_images', default=False, action='store_true', help='Save rendered images with keypoints')
    parser.add_argument('--conf', type=float, default=DEFAULT_CONF, help='Detection confidence threshold')
    parser.add_argument('--cam_idx', type=int, default=5, help='Camera index')
    parser.add_argument('--fast', default=False, help='Use fast mode')
    return parser.parse_args()


def main():
    args = parse_args()
    # Initialize MediaPipe Holistic (includes pose, face, and hands)
    conf = args.conf
    cam_idx = args.cam_idx
    save = bool(args.save_images)
    ORBBEC = True if cam_idx < 5 else False
    dataset = "20250519_Testing"
    input_base_path = f"./data/input/{dataset}/camera0{cam_idx}/"
    output_base_path = input_base_path.replace('input', 'output').replace('tony', 'tony/mediapipe_pose').replace(f'camera0{cam_idx}/', f'{conf:2f}/camera0{cam_idx}/images')
    keypoints_base_path = output_base_path.replace('images', f'keypoint_{"fast" if args.fast else "slow"}')

    # Create output directories
    for dir_path in [output_base_path + '/success', output_base_path + '/partial', output_base_path + '/failure']:
        os.makedirs(dir_path, exist_ok=True)

    for dir_path in [keypoints_base_path + '/success', keypoints_base_path + '/partial', keypoints_base_path + '/failure']:
        os.makedirs(dir_path, exist_ok=True)
    
    logger = setup_logging(output_base_path.replace('images', 'logs'))
    logger.info(f'Running pose estimation for {dataset} on camera0{cam_idx} with confidence {conf} on {"fast" if args.fast else "slow"} mode...')

    # Initialize MediaPipe Holistic
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=conf,
        min_tracking_confidence=0.5
    )

    # Initialize enhancement configuration
    enhancement_config = EnhancementConfig()

    CALIB_DIR = f'./data/input/{dataset}'
    cam_infos = load_cam_infos(Path(CALIB_DIR), orbbec=ORBBEC)
    cam_params = cam_infos[f'camera0{cam_idx}']

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    files = sorted(os.listdir(input_base_path))
    stats = {
        "success": 0,  # both hands detected
        "partial": 0,  # one hand detected
        "failure": 0,  # no hands detected
        "pose_detected": 0,  # pose detected
        "enhanced_success": 0,  # success achieved through enhancement
        "total_enhancement_attempts": 0,  # total enhancement attempts
        "params": [],  # parameters for enhancement
        "start_time": time.time()
    }
    
    for idx, file in enumerate(files):
        if idx % 500 == 0:
            logger.info(f"Processed {idx}/{len(files)}...")
        image_path = os.path.join(input_base_path, file)
        image = cv2.imread(image_path)
        if image is None:
            logger.info(f"Failed to load image: {image_path}")
            continue

        if ORBBEC:
            image = undistort_image(image, cam_params, "color")
        else:
            image = cv2.undistort(
                image, 
                cam_params['intrinsics'], 
                np.array([cam_params['radial_params'][0]] + [cam_params['radial_params'][1]] + list(cam_params['tangential_params'][:2]) + [cam_params['radial_params'][2]] + [0, 0, 0])
            )

        if not args.fast:
            # Use enhanced detection with retry logic
            results, enhancement_attempts, params = detect_hands_with_enhancement(holistic, image, cam_idx, enhancement_config)
            stats["total_enhancement_attempts"] += enhancement_attempts
            if params is not None:
                stats["params"].append(params)
        else:
            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb_image)
            enhancement_attempts = 0

        
        # Initialize data structure for keypoints
        data = {
            "people": [{
                "hand_left_keypoints_2d": [],
                "hand_right_keypoints_2d": [],
                "pose_keypoints_2d": []
            }]
        }
        
        image_for_drawing = image.copy()
        hands_detected = 0
        
        # Process pose landmarks
        if results.pose_landmarks:
            stats["pose_detected"] += 1
            # Draw pose landmarks
            mp_drawing.draw_landmarks(
                image_for_drawing,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Extract pose keypoints (33 landmarks for pose)
            pose_keypoints = []
            for landmark in results.pose_landmarks.landmark:
                pose_keypoints.extend([
                    float(landmark.x * image.shape[1]), 
                    float(landmark.y * image.shape[0]), 
                    float(landmark.visibility)
                ])
            data["people"][0]["pose_keypoints_2d"] = pose_keypoints
        
        # Process left hand landmarks
        if results.left_hand_landmarks:
            hands_detected += 1
            # Draw left hand landmarks
            mp_drawing.draw_landmarks(
                image_for_drawing,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Extract left hand keypoints (21 landmarks) with confidence scores
            left_hand_keypoints = []
            for landmark in results.left_hand_landmarks.landmark:
                left_hand_keypoints.extend([
                    float(landmark.x * image.shape[1]), 
                    float(landmark.y * image.shape[0]), 
                    float(landmark.z * image.shape[1])
                ])
    
            data["people"][0]["hand_left_keypoints_2d"] = left_hand_keypoints
            data["people"][0]["hand_left_conf"] = hand_confidence(results.pose_landmarks, "left")
        
        # Process right hand landmarks
        if results.right_hand_landmarks:
            hands_detected += 1
            # Draw right hand landmarks
            mp_drawing.draw_landmarks(
                image_for_drawing,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # Extract right hand keypoints (21 landmarks) with confidence scores
            right_hand_keypoints = []
            for landmark in results.right_hand_landmarks.landmark:
                right_hand_keypoints.extend([
                    float(landmark.x * image.shape[1]), 
                    float(landmark.y * image.shape[0]), 
                    float(landmark.z * image.shape[1])
                ])
            data["people"][0]["hand_right_keypoints_2d"] = right_hand_keypoints
            data["people"][0]["hand_right_conf"] = hand_confidence(results.pose_landmarks, "right")
        
        # Determine success level based on hands detected
        if hands_detected == 2:
            stats["success"] += 1
            if enhancement_attempts > 0:
                stats["enhanced_success"] += 1
            category = "success"
        elif hands_detected == 1:
            stats["partial"] += 1
            category = "partial"
        else:
            stats["failure"] += 1
            category = "failure"
        
        # Save the annotated image and keypoints
        output_image_path = os.path.join(output_base_path, category, file)
        keypoints_file = os.path.join(keypoints_base_path, category, file.replace('.jpg', '.json'))
        
        if save:
            cv2.imwrite(output_image_path, image_for_drawing)
        
        with open(keypoints_file, 'w') as f:
            json.dump(data, f, indent=4)

    # Release resources
    holistic.close()

    # Calculate and logger.info statistics
    stats["end_time"] = time.time()
    stats["total_time"] = stats["end_time"] - stats["start_time"]
    stats["total_images"] = len(files)
    stats["success_rate"] = stats["success"] / stats["total_images"] if stats["total_images"] > 0 else 0
    stats["partial_rate"] = stats["partial"] / stats["total_images"] if stats["total_images"] > 0 else 0
    stats["pose_rate"] = stats["pose_detected"] / stats["total_images"] if stats["total_images"] > 0 else 0
    stats["enhanced_success_rate"] = stats["enhanced_success"] / stats["success"] if stats["success"] > 0 else 0

    logger.info(f"Processing complete!")
    logger.info(f"Time taken: {stats['total_time']:.2f} seconds for {stats['total_images']} images")
    logger.info(f"Pose detected: {stats['pose_detected']} images ({stats['pose_rate']*100:.1f}%)")
    logger.info(f"Success (both hands): {stats['success']} images ({stats['success_rate']*100:.1f}%)")
    logger.info(f"Enhanced success: {stats['enhanced_success']} images ({stats['enhanced_success_rate']*100:.1f}% of successes)")
    logger.info(f"Partial (one hand): {stats['partial']} images ({stats['partial_rate']*100:.1f}%)")
    logger.info(f"Failed (no hands): {stats['failure']} images")
    logger.info(f"Total enhancement attempts: {stats['total_enhancement_attempts']}")
    logger.info(f"Average enhancement attempts per image: {stats['total_enhancement_attempts']/stats['total_images']:.2f}")
    logger.info(f"Enhancement parameters: {stats['params']}")
    
    # Separate gamma and linear parameters for analysis
    gamma_params = [p for p in stats['params'] if len(p) >= 2 and p[0] == 'gamma']
    linear_params = [p for p in stats['params'] if len(p) >= 3 and p[0] == 'linear']
    
    if gamma_params:
        gamma_values = [p[1] for p in gamma_params]
        logger.info(f"Gamma corrections used: {len(gamma_params)} times")
        logger.info(f"Average gamma value: {np.mean(gamma_values):.2f}")
    
    if linear_params:
        alpha_values = [p[1] for p in linear_params]
        beta_values = [p[2] for p in linear_params]
        logger.info(f"Linear adjustments used: {len(linear_params)} times")
        logger.info(f"Average linear parameters: alpha: {np.mean(alpha_values):.2f}, beta: {np.mean(beta_values):.2f}")
    
    if not gamma_params and not linear_params and stats['params']:
        logger.info("No enhancement parameters could be analyzed (unexpected format)")

if __name__ == "__main__":
    main()
