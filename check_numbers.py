
import os
import json



count = {
    "left": {
        "camera01": 0,
        "camera02": 0,
        "camera03": 0,
        "camera04": 0,
        "camera05": 0,
        "camera06": 0,
    },
    "right": {
        "camera01": 0,
        "camera02": 0,
        "camera03": 0,
        "camera04": 0,
        "camera05": 0,
        "camera06": 0,
    }
}

base_path = "results/20250519_Testing/partial"
for hand in ['left', 'right']:
    for cam_idx in [1, 2, 3, 4, 5, 6]:
        cam = f"camera0{cam_idx}"
        path = f"{base_path}/{hand}/{cam}"
        files = os.listdir(path)
        for file in files:
            with open(f"{path}/{file}", "r") as f:
                kps, conf = json.load(f)
            
            if any(c > 0 for c in conf):
                count[hand][cam] += 1


print(count)