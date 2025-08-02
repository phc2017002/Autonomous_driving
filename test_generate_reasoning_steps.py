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

# ------------ helper: sample context frames -----------------------
def sample_frames(path, lookback=10, step=1, acc_ts=None):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Video FPS:", fps, "Total frames:", total)
    dur   = total / fps
    print("Video duration:", dur, "seconds")
    if acc_ts is None: acc_ts = max(0, dur-1)
    start = max(0, acc_ts - lookback)
    ts = [min(dur, start+i*step) for i in range(int(lookback/step)+1)]
    frames=[]
    for t in ts:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t*fps))
        ok, fr = cap.read()
        if ok: frames.append(fr)
    cap.release()
    return frames, fps, start, step

# ------------ helper: call QVQ-Max -------------------------------
# ------------ helper: call QVQ-Max (robust version) ----------------
def stream_qvq(prompt, frames):
    """
    Sends `prompt` + context `frames` to QVQ-Max and returns
    (numbered_reasoning, answer_text).

    The function is now **stream-safe**: it ignores / logs any chunks that
    don’t contain the fields we expect, so we never hit the old
    AttributeError: 'NoneType' object has no attribute 'choices'.
    """
    # -------- assemble multimodal request --------------------------
    content = [img_item(fr, "Context") for fr in frames[:-1]]
    content.append(img_item(frames[-1], "Accident moment"))
    content.append({"text": prompt})

    resp = MultiModalConversation.call(
        api_key=API_KEY,
        model="qvq-max-latest",
        messages=[{"role": "user", "content": content}],
        stream=True,
    )

    reasoning, answer, is_ans = "", "", False
    print("=" * 20 + "Reasoning Process" + "=" * 20)

    # -------- consume streaming chunks safely ----------------------
    for chunk in resp:
        # ➊ Skip chunks with no usable output
        out = getattr(chunk, "output", None)
        if out is None:
            # Could be an error / done / usage chunk → just continue
            continue

        choices = getattr(out, "choices", None)
        if not choices:
            continue

        # ➋ A chunk can carry either .message (full) or .delta (partial)
        choice = choices[0]
        msg = getattr(choice, "message", None) or getattr(choice, "delta", None)
        if msg is None:
            continue

        # ---------- pull out reasoning / final answer --------------
        # msg.* can be either dict-style or attr-style → handle both
        reasoning_part = (
            msg.get("reasoning_content") if isinstance(msg, dict)
            else getattr(msg, "reasoning_content", None)
        )
        content_part = (
            msg.get("content") if isinstance(msg, dict)
            else getattr(msg, "content", None)
        )

        # Incremental reasoning
        if reasoning_part is not None and (not content_part):
            reasoning += reasoning_part
            print(reasoning_part, end="", flush=True)
            continue

        # Final (or incremental) answer text
        if content_part:
            if not is_ans:
                print("\n" + "=" * 20 + "Complete Response" + "=" * 20)
                is_ans = True
            # content_part is a list of { "text": … }
            txt = (
                content_part[0]["text"]
                if isinstance(content_part, list)
                else str(content_part)
            )
            answer += txt
            print(txt, end="", flush=True)

    print("\n" + "=" * 57 + "\n")

    # -------- post-process ----------------------------------------
    reasoning_numbered = number_reasoning(reasoning)
    return f"<think>\n{reasoning_numbered}\n</think>", answer.strip()

def extract_option(ans):                           # A / B / C / D
    m=re.search(r"\b([A-D])\b", ans)
    return m.group(1) if m else ans

def extract_frame(ans):                            # integer
    m=re.search(r"\b(\d+)\b", ans)
    return int(m.group(1)) if m else None

# ------------ main ------------------------------------------------
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

    # Keys we expect for every fully-processed video
    required_keys = {"weather_and_light", "location", "road_type",
                     "accident_type", "accident_reason",
                     "prevention_method", "prevention_frame"}

    for vid, feats in cfg.items():

        # ---------- NEW: skip if already complete ------------------
        if vid in results and required_keys.issubset(results[vid]):
            print(f"✔ {vid} already finished – skipping.")
            continue
        # -----------------------------------------------------------

        print(f"\n##########  {vid}  ##########")
        frames, fps, start, step = sample_frames(vid, lookback=5, step=1)
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
            time_sec = start + sel*step if sel is not None else None

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