"""Trained image + frame-sampled video authenticity detector."""
from __future__ import annotations
import io, math, os, tempfile
from pathlib import Path
from PIL import Image, ImageChops, ImageStat
IMAGE_EXTS={"jpg","jpeg","png","webp","bmp","tiff"}
VIDEO_EXTS={"mp4","mov","avi","mkv","webm","m4v"}
BASE=Path(__file__).parent
MODEL_PATH=BASE/"model"/"image_detector.pt"

def _entropy(img):
    gray=img.convert("L"); hist=gray.histogram(); total=sum(hist) or 1
    return -sum((c/total)*math.log2(c/total) for c in hist if c)

def _ela(img):
    rgb=img.convert("RGB"); b=io.BytesIO(); rgb.save(b,format="JPEG",quality=90); b.seek(0)
    rec=Image.open(b).convert("RGB"); d=ImageChops.difference(rgb,rec)
    return float(sum(ImageStat.Stat(d).mean)/3)

class _SmallCNN:
    """Lazily-built nn.Module matching train_image.py's from-scratch architecture.
    Defined here (not imported from train_image.py) so inference doesn't need torchvision's
    ImageFolder/dataset machinery - just torch itself."""
    _cls = None

    @classmethod
    def build(cls, image_size):
        if cls._cls is None:
            from torch import nn

            class SmallCNN(nn.Module):
                def __init__(self, image_size):
                    super().__init__()

                    def block(cin, cout):
                        return nn.Sequential(
                            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                            nn.MaxPool2d(2),
                        )

                    self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))
                    final = image_size // 16
                    self.classifier = nn.Sequential(
                        nn.Flatten(), nn.Linear(256 * final * final, 128), nn.ReLU(inplace=True),
                        nn.Dropout(0.4), nn.Linear(128, 2),
                    )

                def forward(self, x):
                    return self.classifier(self.features(x))

            cls._cls = SmallCNN
        return cls._cls(image_size)

def _trained_model():
    if not MODEL_PATH.exists(): return None
    try:
        import torch
        from torchvision import transforms
        ck=torch.load(MODEL_PATH,map_location="cpu")
        image_size=ck.get("image_size",64)
        m=_SmallCNN.build(image_size)
        m.load_state_dict(ck["model_state"]); m.eval()
        return m, transforms.Compose([transforms.Resize((image_size,image_size)),transforms.ToTensor()]), ck.get("class_to_idx",{})
    except Exception:
        return None

def _predict(img):
    bundle=_trained_model()
    if not bundle: return None
    import torch
    m,tf,class_to_idx=bundle
    with torch.no_grad(): p=torch.softmax(m(tf(img.convert("RGB")).unsqueeze(0)),dim=1)[0]
    inv={v:k for k,v in class_to_idx.items()}
    probs={inv.get(i,str(i)):float(p[i]) for i in range(len(p))}
    # ImageFolder labels are usually FAKE=0, REAL=1.
    fake=p[class_to_idx.get("FAKE",0)].item() if "FAKE" in class_to_idx else p[0].item()
    real=1-fake
    label="AI-GENERATED" if fake>=.5 else "REAL"
    return label,max(fake,real),fake,real

def _metadata_signal(exif, software, info):
    """Two kinds of metadata evidence:
    1) A generator's NAME literally present (EXIF Software tag or a PNG text chunk) -
       e.g. 'Midjourney', 'Stable Diffusion'. Strong but rare (most tools don't write this).
    2) PNG generation-PARAMETER keys - the "parameters" chunk Automatic1111 writes, or
       "workflow"/"prompt" from ComfyUI, or "sampler"/"seed"/"cfg_scale" from similar tools.
       These almost never appear on a real photo, and are a far more common real-world
       signal than a tool literally writing its own name - most AI images carry this even
       when they don't name the tool.
    Either kind is treated as near-certain ground truth: it overrides the probabilistic
    CNN prediction, the same way a confirmed fingerprint match would outrank a sketch.
    """
    ai_terms = ("midjourney", "stable diffusion", "dall-e", "dalle", "flux", "comfyui",
                "automatic1111", "novelai", "leonardo", "diffusion")
    name_hit = any(t in software.lower() for t in ai_terms) or any(t in str(v).lower() for v in info.values() for t in ai_terms)
    param_keys = ("parameters", "workflow", "prompt", "negative_prompt", "sampler", "cfg_scale", "seed")
    param_hit = any(k in info for k in param_keys)
    if name_hit:
        return True, "file metadata names a generative tool"
    if param_hit:
        return True, "PNG metadata contains generation parameters (prompt/sampler/seed) typical of AI image tools"
    return False, None

def analyze_image_bytes(data,filename="image"):
    try:
        img=Image.open(io.BytesIO(data)); img.load()
    except Exception as exc: return {"ok":False,"error":f"Could not open image: {exc}"}
    width,height=img.size; exif=getattr(img,"getexif",lambda:{})() or {}
    software=str(exif.get(305,"")) if exif else ""
    info=getattr(img,"info",{}) or {}
    metadata_ai, metadata_reason = _metadata_signal(exif, software, info)
    pred=_predict(img)
    ela=_ela(img); entropy=_entropy(img)
    signals=[]
    if metadata_ai:
        # Near-certain evidence - overrides the model regardless of what it predicts.
        label,conf,fake_prob="AI-GENERATED",0.99,0.99
        signals.append(metadata_reason)
        if pred:
            signals.append(f"trained CNN prediction (overridden by metadata): {pred[0]} ({pred[1]:.1%})")
    elif pred:
        label,conf,fake_prob,real_prob=pred
        signals.append(f"trained CNN prediction: {label} ({conf:.1%})")
    else:
        # Fallback only when model has not been trained.
        risk=min(.99,max(.0,(ela-4)/15)) if ela>4 else .05
        label="AI-GENERATED" if risk>=.5 else "REAL"
        conf=max(risk,1-risk)
        fake_prob=risk
        signals.append("trained image model not found; showing forensic fallback only")
    if not exif: signals.append("no EXIF metadata (not evidence of AI; social platforms often strip metadata)")
    return {"ok":True,"filename":filename,"label":label,"confidence":conf,"ai_risk":fake_prob,
            "width":width,"height":height,"format":img.format,"entropy":round(entropy,3),"ela":round(ela,3),
            "has_exif":bool(exif),"software":software,"signals":signals,
            "reason":"Primary result comes from the trained image classifier, unless file metadata gives near-certain evidence of a generative tool (which takes priority)."}

def analyze_video_bytes(data,filename="video",max_frames=16):
    try: import cv2
    except ImportError: return {"ok":False,"error":"OpenCV is not installed."}
    suffix=Path(filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False,suffix=suffix) as f: f.write(data); path=f.name
    cap=cv2.VideoCapture(path)
    try:
        if not cap.isOpened(): return {"ok":False,"error":"Could not open the video file."}
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0); fps=float(cap.get(cv2.CAP_PROP_FPS) or 0)
        duration=total/fps if fps else 0
        indices=sorted(set(int(i*max(total-1,1)/max_frames) for i in range(max_frames))) if total else []
        results=[]
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES,idx); ok,frame=cap.read()
            if not ok: continue
            img=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            b=io.BytesIO(); img.save(b,format="JPEG",quality=92)
            r=analyze_image_bytes(b.getvalue(),f"frame_{idx}.jpg")
            if r.get("ok"): results.append(r)
    finally:
        cap.release()
        try: os.remove(path)
        except OSError: pass
    if not results: return {"ok":False,"error":"No readable frames were found."}
    fake=sum(r["label"]=="AI-GENERATED" for r in results)/len(results)
    label="AI-GENERATED" if fake>=.5 else "REAL"
    conf=max(fake,1-fake)
    return {"ok":True,"filename":filename,"label":label,"confidence":min(conf,.99),
            "frames_checked":len(results),"total_frames":total,"fps":round(fps,2),
            "duration":round(duration,2),"avg_ai_risk":round(sum(r["ai_risk"] for r in results)/len(results),3),
            "frame_results":results,
            "reason":"Video result is frame-level trained image classification aggregated across sampled frames; it is not a dedicated temporal deepfake model."}
