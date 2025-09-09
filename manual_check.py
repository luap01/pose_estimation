import os
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import cv2
import math
import statistics

# ---------- Drawing topology ----------
HAND_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]

OVERLAP_RATIO = 0.06   # relative to median edge length
OVERLAP_MIN_PX = 6     # at least this many pixels for safety
EPS = 1e-8

# ---------- Basic I/O ----------
def load_json(json_path: Path) -> dict:
    with open(json_path, "r") as f:
        return json.load(f)

def save_json(json_path: Path, data: dict, make_backup: bool = True) -> None:
    # One-time backup to <name>.bak.json
    if make_backup:
        bak = json_path.with_suffix(json_path.suffix + ".bak")
        if not bak.exists():
            try:
                with open(bak, "w") as f:
                    json.dump(load_json(json_path), f, indent=2)
            except Exception:
                pass
    tmp = json_path.with_suffix(json_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, json_path)

# ---------- Hand utilities ----------
def _all_zero_keypoints(flat: List[float]) -> bool:
    if not isinstance(flat, list) or len(flat) < 63:
        return True
    # consider all 63 numbers (x,y,z/score)*21; treat all-zeros as no data
    return all(abs(v) < EPS for v in flat[:63])

def extract_points(person: dict, side: str) -> List[Tuple[float, float]]:
    """
    Extract 21 (x, y) points from the export format.

    New format: absolute image coordinates in hand_{side}_keypoints_2d (length >= 63).
    No shift is present or used.
    """
    key = f"hand_{side}_keypoints_2d"
    flat = person.get(key, [])
    num_kps = 63
    if not isinstance(flat, list) or len(flat) < num_kps or _all_zero_keypoints(flat):
        return []
    return [(float(flat[i]), float(flat[i + 1])) for i in range(0, num_kps, 3)][:21]

def has_hand_data(person: dict, side: str) -> bool:
    """True if a hand has real (non-zero) keypoints."""
    key = f"hand_{side}_keypoints_2d"
    if key not in person:
        return False
    flat = person[key]
    return isinstance(flat, list) and len(flat) >= 63 and not _all_zero_keypoints(flat)

def get_hand_conf_raw(person: dict, side: str):
    """
    Return confidence exactly as stored:
      - list of floats (preferred new format),
      - scalar number (legacy),
      - or None if missing.
    """
    return person.get(f"hand_{side}_conf")

def _conf_to_scalar_for_compare(conf) -> float:
    """
    Convert confidence (list or scalar) to a scalar ONLY for comparison decisions.
    - If list/tuple: use average.
    - If scalar: use as float.
    - Else: 0.0
    This does NOT change what's stored; lists stay lists in JSON.
    """
    if isinstance(conf, (int, float)):
        return float(conf)
    if isinstance(conf, (list, tuple)) and conf:
        nums = [float(x) for x in conf if isinstance(x, (int, float))]
        if nums:
            return sum(nums) / len(nums)
    return 0.0

def zero_hand(person: dict, side: str) -> None:
    """Zero out a hand completely (no shift fields involved)."""
    person[f"hand_{side}_keypoints_2d"] = [0.0] * 63
    conf = person.get(f"hand_{side}_conf")
    if isinstance(conf, list):
        person[f"hand_{side}_conf"] = [0.0] * len(conf)
    else:
        person[f"hand_{side}_conf"] = 0.0

# ---------- Geometry for overlap detection ----------
def _median_edge_length(points: List[Tuple[float, float]]) -> float:
    if not points:
        return 1.0
    lengths = []
    for a, b in HAND_CONNECTIONS:
        if a < len(points) and b < len(points):
            x1, y1 = points[a]
            x2, y2 = points[b]
            lengths.append(math.hypot(x2 - x1, y2 - y1))
    return max(1.0, statistics.median(lengths) if lengths else 1.0)

def _median_pairwise_distance(pA: List[Tuple[float, float]], pB: List[Tuple[float, float]]) -> float:
    if not pA or not pB or len(pA) != len(pB):
        return float("inf")
    dists = [math.hypot(ax - bx, ay - by) for (ax, ay), (bx, by) in zip(pA, pB)]
    return statistics.median(dists) if dists else float("inf")

def are_hands_overlapping(
    left_points: List[Tuple[float, float]],
    right_points: List[Tuple[float, float]]
) -> Tuple[bool, float, float]:
    """
    Returns (overlap, median_pixel_distance, threshold).
    Works directly on absolute coords (no shift).
    """
    if not left_points or not right_points:
        return (False, float("inf"), 0.0)

    median_len = max(_median_edge_length(left_points), _median_edge_length(right_points))
    threshold = max(OVERLAP_MIN_PX, OVERLAP_RATIO * median_len)
    mdist = _median_pairwise_distance(left_points, right_points)
    return (mdist <= threshold, mdist, threshold)

# ---------- Drawing ----------
def draw_hand(img, points: List[Tuple[float, float]], dot_color, line_color, dot_size, line_size):
    if not points:
        return
    # draw points
    for (x, y) in points:
        cv2.circle(img, (int(round(x)), int(round(y))), dot_size, dot_color, -1)
    # draw connections
    for a, b in HAND_CONNECTIONS:
        if a < len(points) and b < len(points):
            x1, y1 = int(round(points[a][0])), int(round(points[a][1]))
            x2, y2 = int(round(points[b][0])), int(round(points[b][1]))
            cv2.line(img, (x1, y1), (x2, y2), line_color, line_size)

def overlay_text(img, lines, origin=(10, 22)):
    x, y = origin
    for line in lines:
        cv2.putText(img, line, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(img, line, (x, y),   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
        y += 22

# ---------- Swap/Move ----------
def swap_hands_in_person(person: dict) -> bool:
    """
    Swap left/right for keys:
      - hand_{side}_keypoints_2d
      - hand_{side}_conf  (list preserved)
    Return True if any swap occurred.
    """
    has_left = has_hand_data(person, "left")
    has_right = has_hand_data(person, "right")
    if not (has_left or has_right):
        return False

    left_keys = {
        "kps": "hand_left_keypoints_2d",
        "conf": "hand_left_conf",
    }
    right_keys = {
        "kps": "hand_right_keypoints_2d",
        "conf": "hand_right_conf",
    }

    tmp = {}
    for k in left_keys:
        tmp[f"L_{k}"] = person.get(left_keys[k])
    for k in right_keys:
        tmp[f"R_{k}"] = person.get(right_keys[k])

    # Remove existing keys to avoid stale values
    for key in left_keys.values():
        person.pop(key, None)
    for key in right_keys.values():
        person.pop(key, None)

    # Right -> Left
    for k, lk in left_keys.items():
        rv = tmp.get(f"R_{k}")
        if rv is not None:
            person[lk] = rv
    # Left -> Right
    for k, rk in right_keys.items():
        lv = tmp.get(f"L_{k}")
        if lv is not None:
            person[rk] = lv

    return True

def move_hand_to_other_side(person: dict) -> Tuple[bool, str]:
    """
    If exactly one hand is present, move it to the other side (keypoints + conf only).
    """
    has_left = has_hand_data(person, "left")
    has_right = has_hand_data(person, "right")

    if (has_left and has_right) or (not has_left and not has_right):
        return False, "Move requires exactly one detected hand."

    if has_left and not has_right:
        for key_type in ["keypoints_2d", "conf"]:
            lkey = f"hand_left_{key_type}"
            rkey = f"hand_right_{key_type}"
            if lkey in person:
                person[rkey] = person.pop(lkey)
        return True, "Moved LEFT → RIGHT."
    else:
        for key_type in ["keypoints_2d", "conf"]:
            lkey = f"hand_left_{key_type}"
            rkey = f"hand_right_{key_type}"
            if rkey in person:
                person[lkey] = person.pop(rkey)
        return True, "Moved RIGHT → LEFT."

# ---------- Navigation helpers ----------
def collect_cameras(dataset_out_dir: Path) -> List[Path]:
    return sorted([d for d in dataset_out_dir.iterdir() if d.is_dir() and d.name.startswith("camera")])

def collect_all_frames(dataset_out_dir: Path, dataset_in_dir: Path) -> List[Dict[str, Any]]:
    """
    Flattens all camera/json frames into a single linear list:
    [
      {
        "cam_dir": Path,
        "cam_name": str,
        "json_path": Path,
        "img_path": Path
      }, ...
    ]
    """
    frames = []
    cameras = collect_cameras(dataset_out_dir)
    for cam_dir in cameras:
        cam_name = cam_dir.name
        json_dir = cam_dir / "keypoints" / "success"
        if not json_dir.exists():
            print(f"Skipping {cam_name}: no json at {json_dir}")
            continue
        img_in_dir = dataset_in_dir / cam_name
        if not img_in_dir.exists():
            print(f"Skipping {cam_name}: images dir missing at {img_in_dir}")
            continue
        json_files = sorted([p for p in json_dir.glob("*.json")])
        if not json_files:
            print(f"Skipping {cam_name}: no JSON frames in {json_dir}")
            continue
        for jf in json_files:
            if int(jf.stem) < 2220:
                continue
            img_path = img_in_dir / f"{jf.stem}.jpg"
            frames.append({
                "cam_dir": cam_dir,
                "cam_name": cam_name,
                "json_path": jf,
                "img_path": img_path,
                "cam_index_1based": cam_name[-1],
                "frame_index_1based": jf.stem
            })
    return frames

def keycode(k: int) -> str:
    """
    Normalize relevant keys.
    Uses cv2.waitKeyEx extended codes for arrows (cross-platform).
    """
    # Printable keys (lowercased)
    if k in (27,): return "ESC"
    if k in (ord('q'), ord('Q')): return "q"
    if k in (ord('d'), ord('D')): return "next"
    if k in (ord('a'), ord('A')): return "prev"
    if k in (ord('s'), ord('S')): return "swap"
    if k in (ord('m'), ord('M')): return "move"
    if k in (ord('r'), ord('R')): return "reload"
    if k in (ord('l'), ord('L')): return "mark_left"
    if k in (ord('p'), ord('P')): return "mark_right"
    if k in (ord('b'), ord('B')): return "mark_both"
    if k in (13, 32): return "next"  # Enter or Space

    # Backspace (platform dependent)
    if k in (8, 65288): return "prev"

    # Arrow keys (extended)
    # Left: 2424832 (Windows), 81 (Linux/macOS)
    # Right: 2555904 (Windows), 83 (Linux/macOS)
    if k in (123, 2424832): return "prev"   # Left arrow
    if k in (124, 2555904): return "next"   # Right arrow

    return ""

# ---------- Viewer ----------
def interactive_viewer(
    output_root: Path,
    dataset: str,
    variant: str,
    input_root: Path,
    save_overlay: bool = False,
    marks_file: Optional[Path] = None
) -> None:
    """
    Interactive visualization with global navigation & swapping.
    If save_overlay=True, writes the drawn image next to JSON for visual audit (optional).
    """
    dataset_out_dir = output_root / dataset / variant
    dataset_in_dir = input_root / dataset

    if not dataset_out_dir.exists():
        print(f"Output directory not found: {dataset_out_dir}")
        return
    if not dataset_in_dir.exists():
        print(f"Input images directory not found: {dataset_in_dir}")
        return

    frames = collect_all_frames(dataset_out_dir, dataset_in_dir)
    if not frames:
        print(f"No frames found under {dataset_out_dir}")
        return
    
    # default marks file
    if marks_file is None:
        marks_file = dataset_out_dir / "marks.txt"

    cv2.namedWindow("hand_viewer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("hand_viewer", 1400, 900)

    cache: Dict[str, dict] = {}      # json_path -> parsed dict
    saved_flags: Dict[str, bool] = {}  # json_path -> saved modification happened at least once
    last_mark_msg: str = ""

    i = 0
    while 0 <= i < len(frames):
        entry = frames[i]
        cam_dir: Path = entry["cam_dir"]
        cam_name: str = entry["cam_name"]
        jf: Path = entry["json_path"]
        img_path: Path = entry["img_path"]
        cam_idx_1 = entry["cam_index_1based"]
        frame_idx_1 = entry["frame_index_1based"]

        frame_key = str(jf)

        # Load data (cache)
        try:
            if frame_key not in cache:
                cache[frame_key] = load_json(jf)
            data = cache[frame_key]
        except Exception as e:
            print(f"  Failed to read {jf}: {e}")
            i = min(i + 1, len(frames) - 1)
            continue

        people = data.get("people", [])
        if not people:
            data["people"] = [{}]
            people = data["people"]
        person = people[0]

        # Auto-fix: if both hands overlap too closely, zero out the weaker one and save
        left_points  = extract_points(person, "left")
        right_points = extract_points(person, "right")

        auto_msg = ""
        if left_points and right_points:
            overlap, mdist, threshold = are_hands_overlapping(left_points, right_points)
            if overlap:
                # Choose which side to keep:
                left_conf_raw  = get_hand_conf_raw(person, "left")
                right_conf_raw = get_hand_conf_raw(person, "right")
                left_conf_s  = _conf_to_scalar_for_compare(left_conf_raw)
                right_conf_s = _conf_to_scalar_for_compare(right_conf_raw)

                # Fallback by hand size if confs tie
                left_size  = _median_edge_length(left_points)
                right_size = _median_edge_length(right_points)

                if (right_conf_s > left_conf_s) or (abs(right_conf_s - left_conf_s) < 1e-6 and right_size >= left_size):
                    zero_hand(person, "left")
                    kept, dropped = "RIGHT", "LEFT"
                else:
                    zero_hand(person, "right")
                    kept, dropped = "LEFT", "RIGHT"

                try:
                    save_json(jf, data, make_backup=True)
                    cache[frame_key] = load_json(jf)  # refresh
                    saved_flags[frame_key] = True
                    auto_msg = f"Auto-fix: overlapped hands (median Δ={mdist:.1f}px ≤ {threshold:.1f}px). Kept {kept}, zeroed {dropped}."
                    # Re-extract after save
                    person = cache[frame_key].get("people", [{}])[0]
                    left_points  = extract_points(person, "left")
                    right_points = extract_points(person, "right")
                except Exception as e:
                    auto_msg = f"Auto-fix failed to save: {e}"

        # Prepare image
        img = cv2.imread(str(img_path)) if img_path.exists() else None
        if img is None:
            # Create placeholder canvas to still allow swapping JSON without image
            placeholder = cv2.UMat(720, 1280, cv2.CV_8UC3).get()
            placeholder[:] = 255
            img = placeholder.copy()

        # Draw after any auto-fix
        draw_hand(img, left_points, (0, 0, 255), (0, 255, 0), 5, 1)     # right: red dots, green lines
        draw_hand(img, right_points,  (255, 0, 0), (0, 255, 255), 3, 2)   # left:  blue dots, yellow lines

        # Hand status
        has_left = bool(left_points)
        has_right = bool(right_points)
        if has_left and has_right:
            hand_status = "Both hands detected"
        elif has_left:
            hand_status = "Only LEFT detected – press 's' to move to RIGHT"
        elif has_right:
            hand_status = "Only RIGHT detected – press 's' to move to LEFT"
        else:
            hand_status = "No hands detected"

        status = "MODIFIED (saved)" if saved_flags.get(frame_key, False) else "Loaded"
        if auto_msg:
            status += " | " + auto_msg

        lines = [
            f"Camera: {cam_name}   Frame: {jf.stem}   [{i+1}/{len(frames)}]",
            "Controls: ←/a/backspace prev | →/d/space/enter next | s swap/move | m move | r reload | q/ESC quit",
            "Marks: l=left, p=right, b=both  (appends to marks file)",
            f"JSON path: {jf.name}",
            f"Marks file: {marks_file}",
            f"Status: {status}",
            f"Hands: {hand_status}",
            "Legend: Right=red+green, Left=blue+yellow",
        ]

        if last_mark_msg:
            lines.append(f"Last mark: {last_mark_msg}")

        overlay_text(img, lines, origin=(10, 26))
        cv2.imshow("hand_viewer", img)

        # Optionally save an overlay image for auditing
        if save_overlay:
            audit_dir = cam_dir / "overlay_preview"
            audit_dir.mkdir(parents=True, exist_ok=True)
            out_path = audit_dir / f"{jf.stem}.jpg"
            try:
                cv2.imwrite(str(out_path), img)
            except Exception:
                pass

        # --- input loop ---
        k = cv2.waitKeyEx(0)  # extended codes
        action = keycode(k)

        if action in ("q", "ESC"):
            cv2.destroyAllWindows()
            return

        elif action == "swap":
            # If only one hand -> move it to empty side; else swap
            if not people:
                data["people"] = [{}]
                person = data["people"][0]
                cache[frame_key] = data

            only_one = (has_left != has_right)
            changed = False
            msg = ""
            if only_one:
                changed, msg = move_hand_to_other_side(person)
            else:
                changed = swap_hands_in_person(person)
                msg = "Swapped LEFT <-> RIGHT" if changed else "Nothing to swap"

            print(f"  {msg}")
            if changed:
                try:
                    save_json(jf, data, make_backup=True)
                    saved_flags[frame_key] = True
                    cache[frame_key] = load_json(jf)
                except Exception as e:
                    print(f"  Failed to save swap/move to {jf}: {e}")

        elif action == "move":
            if not people:
                print(f"  No people data to move in {jf.stem}")
            else:
                changed, msg = move_hand_to_other_side(person)
                print(f"  {msg}")
                if changed:
                    try:
                        save_json(jf, data, make_backup=True)
                        saved_flags[frame_key] = True
                        cache[frame_key] = load_json(jf)
                    except Exception as e:
                        print(f"  Failed to save move to {jf}: {e}")

        elif action == "reload":
            try:
                cache[frame_key] = load_json(jf)
                saved_flags.pop(frame_key, None)  # clear saved flag on reload
                print(f"  Reloaded {jf.stem} from disk")
            except Exception as e:
                print(f"  Failed to reload {jf}: {e}")

        elif action in ("mark_left", "mark_right", "mark_both"):
            remark = "left" if action == "mark_left" else ("right" if action == "mark_right" else "both")
            cam_num = infer_cam_number(cam_name, cam_idx_1)
            frame_num = infer_frame_number(jf.stem, frame_idx_1)
            try:
                last_mark_msg = append_mark_line(marks_file, cam_num, frame_num, remark)
                print(f"  Marked: {last_mark_msg}")
            except Exception as e:
                last_mark_msg = f"Failed to write mark: {e}"
                print(f"  {last_mark_msg}")

        elif action == "prev":
            i = max(0, i - 1)

        else:  # default to next for any other key or "next"
            i = min(i + 1, len(frames) - 1)

    cv2.destroyAllWindows()


# ---------- Marks helpers ----------
def _parse_number_from_string(s: str):
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits == "":
        return None
    try:
        return int(digits.lstrip("0") or "0")
    except Exception:
        return None

def infer_cam_number(cam_name: str, cam_index_1based: int) -> int:
    n = _parse_number_from_string(cam_name)
    return n if n is not None else cam_index_1based

def infer_frame_number(stem: str, frame_index_1based: int) -> int:
    n = _parse_number_from_string(stem)
    return n if n is not None else frame_index_1based

def append_mark_line(marks_path: Path, cam_num: int, frame_num: int, remark: str) -> str:
    marks_path.parent.mkdir(parents=True, exist_ok=True)
    line = f"{remark} camera {cam_num} frame {frame_num}\n"
    with open(marks_path, "a", encoding="utf-8") as f:
        f.write(line)
    return line.strip()

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="Interactive viewer for mediapipe keypoints (no shift) with swap & navigation")
    parser.add_argument(
        "--dataset",
        default="20250519_Testing",
        help="Dataset name under data roots (e.g., 20250206 or 20250206_Testing)",
    )
    parser.add_argument(
        "--variant",
        default="0.350000",
        help="Output variant folder name under dataset (default: mediapipe_0.35)",
    )
    parser.add_argument(
        "--output-root",
        default="data/output",
        help="Root directory where JSON outputs live (default: data/output)",
    )
    parser.add_argument(
        "--input-root",
        default="data/input",
        help="Root directory where input images live (default: data/input)",
    )
    parser.add_argument(
        "--save-overlay",
        action="store_true",
        help="If set, stores a drawn preview image per frame under each camera/overlay_preview/",
    )

    args = parser.parse_args()

    interactive_viewer(
        output_root=Path(args.output_root),
        dataset=args.dataset,
        variant=args.variant,
        input_root=Path(args.input_root),
        save_overlay=args.save_overlay,
    )

if __name__ == "__main__":
    main()
