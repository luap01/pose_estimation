import numpy as np
import json
import os
import cv2
import argparse

def json_load(p):
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d

def save_file(data, path):
    with open(path, "w") as json_file:
        json.dump(data, json_file, indent=4)

def build_arr(data, hand_side, cam=None):
    kps = data[f"hand_{hand_side}_keypoints_2d"]
    conf = data[f"hand_{hand_side}_conf"]
    coords = []
    confs = []
    # LEFT_WRIST,
    # LEFT_PINKY,
    # LEFT_INDEX,
    # LEFT_THUMB,
    wrist = conf[0]
    pinky = conf[1]
    index = conf[2]
    thumb = conf[3]
    middle = 0.7 * index + 0.3 * pinky
    ring = 0.3 * index + 0.7 * pinky
    confs = [
        wrist,                             # 0: wrist
        thumb, thumb, thumb, thumb,        # 1-4: thumb joints
        index, index, index, index,        # 5-8: index joints
        middle, middle, middle, middle,    # 9-12: middle joints
        ring, ring, ring, ring,            # 13-16: ring joints
        pinky, pinky, pinky, pinky         # 17-20: pinky joints
    ]
    for idx in range(0, len(kps), 3):
        coords.append(kps[idx])
        coords.append(kps[idx+1])

    if cam == "camera03":
        confs = [0] * 21
    return [coords, confs]


def extract_hand_pose_coords(pose_keypoints_2d, hand_side: str):
    """
    Build a coords list from pose_keypoints_2d for a given hand.

    We use the 4 hand-related pose landmarks per hand (wrist, pinky, index, thumb):
    - Left: 15 (wrist), 17 (pinky), 19 (index), 21 (thumb)
    - Right: 16 (wrist), 18 (pinky), 20 (index), 22 (thumb)

    Input pose_keypoints_2d is expected to be a flat list of length 33*3:
    [x0, y0, c0, x1, y1, c1, ..., x32, y32, c32].

    Returns a flat [x, y, ...] list for 21 hand keypoints (42 numbers total).
    Only these slots are filled from pose, remaining are zeros:
      - 0: wrist
      - 4: thumb tip
      - 8: index tip
      - 20: pinky tip
    """
    # Initialize 21 keypoints (x,y) with zeros
    coords = [0.0] * (21 * 2)

    if not pose_keypoints_2d or len(pose_keypoints_2d) < 33 * 3:
        return [coords, [0.0] * (len(coords) // 2)]

    if hand_side == "left":
        wrist_idx, pinky_idx, index_idx, thumb_idx = 15, 17, 19, 21
    else:
        wrist_idx, pinky_idx, index_idx, thumb_idx = 16, 18, 20, 22

    wrist_x, wrist_y = pose_keypoints_2d[3*wrist_idx], pose_keypoints_2d[3*wrist_idx+1]
    pinky_x, pinky_y = pose_keypoints_2d[3*pinky_idx], pose_keypoints_2d[3*pinky_idx+1]
    index_x, index_y = pose_keypoints_2d[3*index_idx], pose_keypoints_2d[3*index_idx+1]
    thumb_x, thumb_y = pose_keypoints_2d[3*thumb_idx], pose_keypoints_2d[3*thumb_idx+1] 

    x_max = max(max(max(wrist_x, pinky_x), index_x), thumb_x)
    x_min = min(min(min(wrist_x, pinky_x), index_x), thumb_x)
    y_max = max(max(max(wrist_y, pinky_y), index_y), thumb_y)
    y_min = min(min(min(wrist_y, pinky_y), index_y), thumb_y)

    threshold_val = 120
    if x_max - x_min < threshold_val:
        diff = threshold_val - (x_max - x_min)
        x_max += (diff * 0.5)
        x_min -= (diff * 0.5)
    if y_max - y_min < threshold_val:
        diff = threshold_val - (y_max - y_min)
        y_max += (diff * 0.5)
        y_min -= (diff * 0.5)

    # x_max += 50
    # x_min += 50

    # Keep legacy return structure (unused downstream)
    kps = [0.0] * 42
    for idx in range(0, 42, 2):
        kps[idx] = float(x_max)
        kps[idx + 1] = float(y_max)
    kps[2] = float(x_min)
    kps[3] = float(y_min)

    coords = kps

    return [coords, [0.0] * (len(coords) // 2)]

def has_valid_hand_keypoints(data, hand_side):
    """Check if hand has valid keypoints (not empty/zero)"""
    kps = data.get(f"hand_{hand_side}_keypoints_2d", [])
    if not kps or len(kps) == 0:
        return False
    # Check if keypoints are not all zeros
    return any(kp != 0 for kp in kps)

def parse_args():
    parser = argparse.ArgumentParser(description='Convert MediaPipe keypoints to custom format')
    parser.add_argument('--conf', type=float, default=0.35, help='Confidence threshold level')
    parser.add_argument('--categories', nargs='+', default=['success'], 
                       help='Categories to process (success, partial, failure)')
    parser.add_argument('--dataset', default="20250519_Testing")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build paths relative to the script location or use provided paths
    base_path = os.path.join(script_dir, "data", "output", args.dataset, f"{args.conf:.6f}")
    
    tar_base_path = os.path.join(script_dir, "results", args.dataset)
    
    print(f"Processing confidence level: {args.conf:.6f}")
    print(f"Categories: {args.categories}")
    print(f"Input path: {base_path}")
    print(f"Output path: {tar_base_path}")
    print("-" * 60)
    
    if not os.path.exists(base_path):
        print(f"Error: Base path {base_path} does not exist!")
        exit(1)
    
    directories = os.listdir(base_path)
    total_processed = 0
    total_errors = 0
    
    for camera_name in directories:
        if camera_name == "calib":
            continue
            
        print(f"Processing {camera_name}...")
        camera_processed = 0
        camera_errors = 0
        
        for category in args.categories:
            category_path = os.path.join(base_path, camera_name, "keypoint_slow", category)
            
            if not os.path.exists(category_path):
                print(f"  Warning: {category_path} does not exist, skipping...")
                continue
            
            files = os.listdir(category_path)
            files = [f for f in files if f.endswith(".json")]
            print(f"  Processing {len(files)} files from {category} category...")
            
            for idx, file in enumerate(files):
                try:
                    data = json_load(os.path.join(category_path, file))
                    files_saved = 0

                    if len(data["people"]) > 0:
                        person_data = data["people"][0]
                        
                        # Check which hands are detected
                        has_left = has_valid_hand_keypoints(person_data, "left")
                        has_right = has_valid_hand_keypoints(person_data, "right")
                        
                        if category == "partial":
                            # For success, save both hands (even if one might be empty)
                            
                            res_left = build_arr(person_data, "left") if has_left else extract_hand_pose_coords(person_data['pose_keypoints_2d'], "left")
                            res_right = build_arr(person_data, "right") if has_right else extract_hand_pose_coords(person_data['pose_keypoints_2d'], "right")

                            if camera_name == "camera06":
                                tmp = res_left
                                res_left = res_right
                                res_right = tmp
                            
                        # elif category == "partial":
                        #     # For partial, only save detected hands
                        #     if has_left:
                        #         res_left = build_arr(person_data, "left")
                        #         res_right = extract_hand_pose_coords(person_data['pose_keypoints_2d'], "right")
                                
                        #     if has_right:
                        #         res_right = build_arr(person_data, "right")
                        #         res_left = extract_hand_pose_coords(person_data['pose_keypoints_2d'], "left")
                            
                        # elif category == "failure":
                        #     res_right = extract_hand_pose_coords(person_data['pose_keypoints_2d'], "right")
                        #     res_left = extract_hand_pose_coords(person_data['pose_keypoints_2d'], "left")

                            
                            tar_pth_left = os.path.join(tar_base_path, category, "left", camera_name, file)
                            tar_pth_right = os.path.join(tar_base_path, category, "right", camera_name, file)

                            os.makedirs(os.path.dirname(tar_pth_left), exist_ok=True)
                            os.makedirs(os.path.dirname(tar_pth_right), exist_ok=True)
                            
                            if res_right:
                                save_file(res_right, tar_pth_right)
                                files_saved += 1
                            if res_left:
                                save_file(res_left, tar_pth_left)
                                files_saved += 1
                            
                    else:
                        pass
                        # For partial with no people, skip entirely
                    
                    camera_processed += files_saved
                    total_processed += files_saved

                except Exception as e:
                    print(f"    Error processing {camera_name}/{category}/{file}: {e}")
                    camera_errors += 1
                    total_errors += 1
        
        print(f"  {camera_name}: {camera_processed} output files created, {camera_errors} errors")
    
    print("-" * 60)
    print(f"Total output files created: {total_processed}")
    print(f"Total errors: {total_errors}")
    print(f"Output saved to: {tar_base_path}")
    print(f"Directory structure: {tar_base_path}/{{left,right}}/{{camera_name}}/{{file.json}}")