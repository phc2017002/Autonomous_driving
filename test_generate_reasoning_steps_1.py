#!/usr/bin/env python3
"""
analyse_vru_to_json_numbered.py
———————————————
• Handles ALL six features (weather & light, location, road type,
  accident type, accident reason, prevention method).
• Produces numbered reasoning steps.
• Dumps everything to a JSON file.
"""
import os, json, cv2, base64, argparse, re, tempfile, shutil 
import dashscope
from dashscope import MultiModalConversation

dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
API_KEY = os.getenv("DASHSCOPE_API_KEY")
assert API_KEY, "export DASHSCOPE_API_KEY first!"


def _flush(out_path: str, data: dict):
    """Atomically dump `data` to `out_path`."""
    tmp = tempfile.NamedTemporaryFile(  # write to a tmp-file first
        delete=False, suffix=".tmp", dir=os.path.dirname(out_path))
    with open(tmp.name, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)
    shutil.move(tmp.name, out_path)  

# ------------ helper: load / convert metadata ---------------------
def load_video_config(jpath):
    """
    Converts the user-supplied JSON into a dict
      { video :
          {feature_key : prompt_string, …}
      }
    """
    raw = json.load(open(jpath, "r", encoding="utf-8"))
    mapping = {
        "weather and light": "weather_and_light",
        "location":          "location",
        "road type":         "road_type",
        "accident type":     "accident_type",
        "accident reason":   "accident_reason",
        "prevention method": "prevention_method",
    }

    cfg = {}
    for vid, sections in raw.items():
        feats = {}
        for jkey, pkey in mapping.items():
            if jkey in sections:
                q  = sections[jkey]["question"]
                op = sections[jkey]["options"]
                q = q.replace("Choose the correct option",
                              "Choose ONLY one option")
                # Ask the model to OUTPUT numbered reasoning
                prompt = (f"{q}  {op} "
                          "First think step-by-step with numbered points "
                          "(1., 2., 3., …). Then in your final answer give "
                          "ONLY the chosen option letter.")
                feats[pkey] = prompt
        cfg[vid] = feats
    return cfg

# ------------ helper: numbering for reasoning ---------------------
def number_reasoning(text: str) -> str:
    # split by newline or full stop
    bits = re.split(r'[\n\.]', text)
    bits = [b.strip() for b in bits if b.strip()]
    return "\n".join(f"{i+1}. {b}" for i, b in enumerate(bits))

def to_b64(img):
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf).decode()

def img_item(img, caption=""): return {"image_base64":to_b64(img), "text":caption}

# ------------ helper: extract every frame -------------------------
def extract_all_frames(path):
    """
    Extract every frame from the video.
    
    Parameters
    ----------
    path : str  – path to input video.

    Returns
    -------
    frames : list  – list of all frames (as numpy arrays).
    fps    : float – frames per second.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)

    cap.release()
    return frames, fps

# ------------ helper: call QVQ-Max -------------------------------
def stream_qvq(prompt, frames):
    content = [img_item(fr, "Context") for fr in frames[:-1]]
    content.append(img_item(frames[-1], "Accident moment"))
    content.append({"text": prompt})

    resp = MultiModalConversation.call(
        api_key=API_KEY,
        model="qvq-max-latest",
        messages=[{"role": "user", "content": content}],
        stream=True,
    )

    reasoning, answer = "", ""
    is_ans = False
    print("=" * 20 + "Reasoning Process" + "=" * 20)

    for chunk in resp:
        # ---- 1️⃣  Skip keep-alive / error / empty chunks ------------
        if chunk is None or getattr(chunk, "output", None) is None:
            # If the SDK attaches error details, you can surface them:
            if getattr(chunk, "code", None):
                print(f"\n[STREAM-ERROR] {chunk.code}: {chunk.message}\n")
            continue

        msg = chunk.output.choices[0].message

        # ---- 2️⃣  Accumulate reasoning vs. final answer -------------
        if msg.get("reasoning_content") is not None and msg.content == []:
            reasoning += msg.reasoning_content
            print(msg.reasoning_content, end="", flush=True)

        elif msg.content:
            if not is_ans:
                print("\n" + "=" * 20 + "Complete Response" + "=" * 20)
                is_ans = True
            txt = msg.content[0]["text"]
            answer += txt
            print(txt, end="", flush=True)

    print("\n" + "=" * 57 + "\n")
    reasoning_numbered = number_reasoning(reasoning)
    return f"<think>\n{reasoning_numbered}\n</think>", answer.strip()

def extract_option(ans):                           # A / B / C / D
    m=re.search(r"\b([A-D])\b", ans)
    return m.group(1) if m else ans

def extract_frame(ans):                            # integer
    m=re.search(r"\b(\d+)\b", ans)
    return int(m.group(1)) if m else None

# ------------ main ------------------------------------------------
def main(meta_json, out_json):
    cfg      = load_video_config(meta_json)

    # resume if a partial file already exists
    results  = {}
    if os.path.isfile(out_json):
        try:
            results = json.load(open(out_json, "r", encoding="utf-8"))
            print(f"Loaded existing {out_json} – continuing …")
        except Exception as e:
            print(f"Could not read existing file ({e}), starting fresh.")

    for vid, feats in cfg.items():
        print(f"\n##########  {vid}  ##########")
        frames, fps = extract_all_frames(vid)

        vid_res = results.get(vid, {})

        order = ["weather_and_light","location","road_type",
                 "accident_type","accident_reason","prevention_method"]

        # ----------- regular six Q&As --------------------------------
        for key in order:
            if key in vid_res:                       # already processed
                print(f"-- {key.replace('_',' ').title()} (cached) --")
                continue

            print(f"-- {key.replace('_',' ').title()} --")
            reasoning, ans = stream_qvq(feats[key], frames)

            # ➊  ONLY save the single-letter answer; drop reasoning
            vid_res[key] = {
                "answer": extract_option(ans)
            }

            results[vid] = vid_res
            _flush(out_json, results)                # atomic write

        # ------------- prevention frame ------------------------------
        if "prevention_frame" not in vid_res:
            print("-- Prevention Frame --")
            idxs=list(range(len(frames)))
            pf_prompt=(f"We numbered the frames 0-{idxs[-1]} "
                       "(0 = earliest, last = accident). "
                       "Which single frame (≈1-2 s before impact) provides "
                       "the best opportunity to avoid the collision? "
                       f"Choose ONLY one: {', '.join(map(str,idxs))}. "
                       "Explain your reasoning in numbered steps first.")
            r_pf, ans_pf = stream_qvq(pf_prompt, frames)
            sel = extract_frame(ans_pf)
            time_sec = (sel / fps) if sel is not None and fps > 0 else None

            vid_res["prevention_frame"] = {          # ➋ keep reasoning
                "Reasoning_steps": r_pf,
                "frame_for_prevention": sel,
                "time_of_the_frame": time_sec
            }

            results[vid] = vid_res
            _flush(out_json, results)

    print(f"\n✅ All done! Latest JSON is in: {out_json}")

# -----------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="input metadata JSON")
    ap.add_argument("--out",  default="results.json",
                    help="output results JSON")
    args = ap.parse_args()
    main(args.meta, args.out)