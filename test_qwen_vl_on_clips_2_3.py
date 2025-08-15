#!/usr/bin/env python3
# =======================================================================
# Predict VRU questions with Qwen-VL
# Robust version – skips missing files, handles decode + API errors
# =======================================================================
import os, json, time, pathlib, base64
from typing import Dict, List

import cv2
import dashscope

# ----------------------------- CONFIG ----------------------------------
JSON_ANNOT   = "cliped_test_pilot.json"        # Q/A definition file
RESULT_FILE  = "cliped_vru_predictions.json"   # output
SAMPLE_FPS   = 2                               # fps for frame sampling
MAX_FRAMES   = 16                              # max frames per clip
QWEN_MODEL   = "qwen2.5-vl-72b-instruct"

dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
# -----------------------------------------------------------------------


# ---------------------- video reading helpers --------------------------
def _print_cv_build_info_once():
    if getattr(_print_cv_build_info_once, "_printed", False):
        return
    print("─ OpenCV build info (FFMPEG line) ─")
    for ln in cv2.getBuildInformation().splitlines():
        if "FFMPEG:" in ln:
            print("  ", ln.strip())
            break
    print("────────────────────────────────────")
    _print_cv_build_info_once._printed = True


def _extract_with_opencv(path: pathlib.Path,
                         target_fps: int,
                         max_frames: int) -> List[str]:
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError("OpenCV cannot open the file")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step       = max(int(native_fps // target_fps), 1)

    frames, idx = [], 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                b64 = base64.b64encode(buf).decode("utf8")
                frames.append(f"data:image/jpeg;base64,{b64}")
                if len(frames) >= max_frames:
                    break
        idx += 1
    cap.release()
    return frames


def _extract_with_imageio(path: pathlib.Path,
                          target_fps: int,
                          max_frames: int) -> List[str]:
    import imageio.v3 as iio
    frames = []
    for idx, frame in enumerate(iio.imiter(path, plugin="pyav", fps=target_fps)):
        ok, buf = cv2.imencode(".jpg", frame[..., ::-1])
        if ok:
            b64 = base64.b64encode(buf).decode("utf8")
            frames.append(f"data:image/jpeg;base64,{b64}")
            if idx + 1 >= max_frames:
                break
    return frames


def extract_frames_b64(path: pathlib.Path,
                       target_fps: int = SAMPLE_FPS,
                       max_frames:  int = MAX_FRAMES) -> List[str]:
    _print_cv_build_info_once()

    # try OpenCV first
    try:
        frames = _extract_with_opencv(path, target_fps, max_frames)
        if frames:
            return frames
        raise RuntimeError("OpenCV decoded 0 frames")
    except Exception as e:
        print(f"   ⚠️  OpenCV failed ({e}). Falling back to imageio …")

    # fallback imageio
    frames = _extract_with_imageio(path, target_fps, max_frames)
    if not frames:
        raise RuntimeError("imageio decoded 0 frames")
    return frames
# -----------------------------------------------------------------------


# ---------------------- Qwen-VL helpers --------------------------------
def build_question_blocks(meta: Dict) -> List[Dict]:
    order = [
        "weather and light", "location", "road type",
        "accident type", "accident reason", "prevention method"
    ]
    blocks = []
    for key in order:
        q  = meta[key]["question"]
        op = meta[key]["options"]
        blocks.append(
            {"text": f"{q} {op} Answer with a single letter (A, B, C or D) and nothing else."}
        )
    return blocks


def query_qwen(frames_b64: List[str], q_blocks: List[Dict]) -> List[str]:
    messages = [{
        "role": "user",
        "content": [{"video": frames_b64, "fps": SAMPLE_FPS}, *q_blocks]
    }]

    rsp = dashscope.MultiModalConversation.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model=QWEN_MODEL,
        messages=messages,
        seed=42
    )

    # ★ NEW: make sure the response body is valid
    if rsp is None or rsp.get("output") is None:
        raise RuntimeError(f"Invalid response: {rsp}")

    outputs = rsp["output"]["choices"][0]["message"]["content"]
    return [blk["text"] for blk in outputs if "text" in blk]
# -----------------------------------------------------------------------


# ------------------------------ main -----------------------------------
def main() -> None:
    with open(JSON_ANNOT, encoding="utf8") as f:
        meta_json: Dict[str, Dict] = json.load(f)

    predictions: Dict[str, List[str] | str] = {}

    for vid_path, qa_meta in meta_json.items():
        print(f"▶ Processing {vid_path} …", flush=True)

        abs_path = pathlib.Path(vid_path).expanduser().resolve()

        # ---------- check if the file actually exists ------------------ ★ NEW
        if not abs_path.is_file():
            print("   ✗ File missing → skipping.")
            predictions[vid_path] = "MISSING_FILE"
            continue
        # ----------------------------------------------------------------

        # 1) frame extraction
        try:
            frames_b64 = extract_frames_b64(abs_path)
        except Exception as e:
            print(f"   ✗ Cannot extract frames: {e}")
            predictions[vid_path] = f"FRAME_ERROR: {e}"
            continue

        # 2) ask the model
        try:
            answers = query_qwen(frames_b64, build_question_blocks(qa_meta))
            predictions[vid_path] = answers
            print("   ✓ done.")
        except Exception as e:
            print(f"   ⚠️  API error: {e}")
            predictions[vid_path] = f"API_ERROR: {e}"
            time.sleep(5)          # polite back-off

    # 3) save results
    with open(RESULT_FILE, "w", encoding="utf8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f"\nAll finished ⇒ {RESULT_FILE}")


if __name__ == "__main__":
    main()