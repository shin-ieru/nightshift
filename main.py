from pathlib import Path
from typing import List, Optional
import re
from io import BytesIO

import cv2

from fastapi import FastAPI, Request, UploadFile, File, Depends, Query
from fastapi.responses import RedirectResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from database import SessionLocal, init_db
from models import Frame, Detection
from process_video import process_video

from pydantic import BaseModel
from sam3_helper import run_sam3_on_path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# --- Safety cap for crash-prevention ---
MAX_PDF_FRAMES_HARD = 20

# ---------- SAM3 helper config ----------

# Nice distinct BGR colors for different prompts (SAM3)
SAM3_COLORS = [
    (255, 0, 255),   # magenta
    (0, 255, 255),   # yellow/cyan
    (0, 165, 255),   # orange
    (0, 255, 0),     # green
    (255, 0, 0),     # blue
    (255, 255, 0),   # cyan-ish yellow
]


def _split_prompts(raw: str) -> list[str]:
    """
    Split a single query string into multiple prompts.
    Supports separators: ',', ';', '|'.

    Example:
        'yellow car; bus | truck'
        -> ['yellow car', 'bus', 'truck']
    """
    if not raw:
        return []
    parts = re.split(r"[;,|]+", raw)
    prompts = [p.strip() for p in parts if p.strip()]
    if not prompts and raw.strip():
        prompts = [raw.strip()]
    return prompts


# ---------- DB + FastAPI setup ----------

init_db()

app = FastAPI()

FRAMES_DIR = "frames"
Path(FRAMES_DIR).mkdir(exist_ok=True)
app.mount("/frames", StaticFiles(directory=FRAMES_DIR), name="frames")

templates = Jinja2Templates(directory="templates")

VIDEOS_DIR = Path("videos")
VIDEOS_DIR.mkdir(exist_ok=True)
app.mount("/videos", StaticFiles(directory="videos"), name="videos")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class SmartSearchRequest(BaseModel):
    query: str                # natural language prompt(s), e.g. "yellow car; bus"
    max_frames: int = 50      # how many frames to scan
    min_score: float = 0.3    # filter low-confidence SAM3 hits
    video: Optional[str] = None  # optional video_path to restrict search


# ---------- Routes ----------

@app.get("/", response_class=HTMLResponse)
async def index(
    request: Request,
    label: str | None = Query(default=None),
    min_conf: float = Query(0.3, ge=0.0, le=1.0),
    db: Session = Depends(get_db),
):
    """
    Home page: upload, YOLO search, SMART search UI.
    """
    results: List[dict] = []

    # YOLO results for the timeline
    if label or min_conf is not None:
        q = (
            db.query(Detection, Frame)
            .join(Frame, Detection.frame_id == Frame.id)
            .order_by(Frame.timestamp)
        )

        if label:
            q = q.filter(Detection.label == label)

        if min_conf is not None:
            q = q.filter(Detection.score >= min_conf)

        q = q.limit(200)
        rows = q.all()

        for det, frame in rows:
            results.append(
                {
                    "id": frame.id,
                    "timestamp": frame.timestamp,
                    "label": det.label,
                    "score": det.score,
                    "frame_path": frame.frame_path,
                    "video_path": frame.video_path,
                }
            )

    # available videos for dropdown (distinct video_path)
    video_rows = db.query(Frame.video_path).distinct().all()
    video_paths = [row[0] for row in video_rows]

    current_video = None
    if results:
        # prefer the video that has hits in the current filter
        current_video = results[0]["video_path"]
    elif video_paths:
        # fallback to most recently indexed video (roughly)
        current_video = video_paths[-1]

    video_src = None
    if current_video:
        video_src = request.url_for("videos", path=Path(current_video).name)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "results": results,
            "label": label,
            "min_conf": min_conf,
            "video_src": video_src,
            "video_paths": video_paths,
            "current_video": current_video,
        },
    )


@app.post("/upload")
async def upload_video(video: UploadFile = File(...)):
    """
    Save uploaded video, run YOLO indexing pipeline, redirect home.
    """
    suffix = Path(video.filename).suffix or ".mp4"
    out_path = VIDEOS_DIR / f"video_{video.filename}"
    with out_path.open("wb") as f:
        content = await video.read()
        f.write(content)

    # run your YOLO+DB pipeline
    process_video(str(out_path))

    return RedirectResponse(url="/", status_code=303)


@app.post("/smart-search")
def smart_search(
    payload: SmartSearchRequest,
    db: Session = Depends(get_db),
):
    """
    Run SAM3 on a subset of frames and return matching detections
    (natural-language smart search).

    Supports multi-prompt queries, e.g.:
      "yellow car; bus; truck"

    Each prompt gets its own color when rendered by /annotated_sam3.

    Also supports restricting search to a single video:
      payload.video == Frame.video_path
    """
    prompts = _split_prompts(payload.query)
    if not prompts:
        return {
            "query": payload.query,
            "min_score": payload.min_score,
            "scanned_frames": 0,
            "num_hits": 0,
            "results": [],
            "prompts": [],
            "video": payload.video,
        }

    q = db.query(Frame)
    if payload.video:
        q = q.filter(Frame.video_path == payload.video)

    frames = (
        q.order_by(Frame.timestamp)
        .limit(payload.max_frames)
        .all()
    )

    results: list[dict] = []

    for frame in frames:
        frame_path = Path(FRAMES_DIR) / frame.frame_path

        for prompt in prompts:
            sam_dets = run_sam3_on_path(str(frame_path), prompt)

            for det in sam_dets:
                score = float(det["score"])
                if score < payload.min_score:
                    continue

                results.append(
                    {
                        "frame_id": frame.id,
                        "timestamp": frame.timestamp,
                        "prompt": prompt,
                        "label": det.get("label", ""),
                        "score": score,
                        "bbox": det["bbox"],
                        "frame_path": frame.frame_path,
                        "video_path": frame.video_path,
                    }
                )

    results.sort(key=lambda r: r["timestamp"])

    return {
        "query": payload.query,
        "min_score": payload.min_score,
        "scanned_frames": len(frames),
        "num_hits": len(results),
        "results": results,
        "prompts": prompts,
        "video": payload.video,
    }


@app.get("/annotated/{frame_id}")
def annotated_frame(frame_id: int, db: Session = Depends(get_db)):
    """
    Return the frame image with YOLO bounding boxes drawn on it.
    Used for the YOLO timeline thumbnails + overlay.
    """
    frame = db.query(Frame).filter(Frame.id == frame_id).first()
    if not frame:
        return Response(status_code=404)

    img_path = Path(FRAMES_DIR) / frame.frame_path
    img = cv2.imread(str(img_path))
    if img is None:
        return Response(status_code=404)

    for det in frame.detections:
        x = int(det.bbox["x"])
        y = int(det.bbox["y"])
        w = int(det.bbox["w"])
        h = int(det.bbox["h"])
        x2 = x + w
        y2 = y + h

        cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 2)
        label_text = f"{det.label} {det.score:.2f}"
        cv2.putText(
            img,
            label_text,
            (x, max(0, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    ok, buffer = cv2.imencode(".jpg", img)
    if not ok:
        return Response(status_code=500)

    return Response(content=buffer.tobytes(), media_type="image/jpeg")


@app.get("/annotated_sam3/{frame_id}")
def annotated_sam3(
    frame_id: int,
    query: str = Query(..., description="Natural language prompt(s), e.g. 'yellow car; bus'"),
    min_score: float = Query(0.3, ge=0.0, le=1.0),
    db: Session = Depends(get_db),
):
    """
    Return the frame image with SAM3 masks drawn on it for one or more prompts.

    - `query` can contain multiple prompts, separated by ',', ';', or '|'
        e.g. "yellow car; bus; truck"
    - Each prompt gets its own semi-transparent color from SAM3_COLORS.
    - Fill-only masks + small label pill (no hard box outlines).
    """
    frame = db.query(Frame).filter(Frame.id == frame_id).first()
    if not frame:
        return Response(status_code=404)

    img_path = Path(FRAMES_DIR) / frame.frame_path
    img = cv2.imread(str(img_path))
    if img is None:
        return Response(status_code=404)

    prompts = _split_prompts(query)
    if not prompts:
        prompts = [query.strip()] if query.strip() else []

    if not prompts:
        ok, buffer = cv2.imencode(".jpg", img)
        return (
            Response(content=buffer.tobytes(), media_type="image/jpeg")
            if ok
            else Response(status_code=500)
        )

    overlay = img.copy()
    alpha = 0.45  # how strong the color tint is

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    pills_to_draw: list[dict] = []

    for idx, prompt in enumerate(prompts):
        color = SAM3_COLORS[idx % len(SAM3_COLORS)]

        sam_dets = run_sam3_on_path(str(img_path), prompt)

        for det in sam_dets:
            score = float(det["score"])
            if score < min_score:
                continue

            x, y, w, h = det["bbox"]
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            x2 = x + w
            y2 = y + h

            mask = det.get("mask")

            if mask is not None:
                m = mask.astype(bool)
                overlay[m] = color
            else:
                cv2.rectangle(overlay, (x, y), (x2, y2), color, -1)

            label_text = f"{prompt} {score:.2f}"
            pills_to_draw.append(
                {
                    "text": label_text,
                    "x": x,
                    "y": y,
                    "color": color,
                }
            )

    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    for pill in pills_to_draw:
        label_text = pill["text"]
        x = pill["x"]
        y = pill["y"]
        color = pill["color"]

        (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

        pill_x1 = x
        pill_y1 = max(0, y - th - baseline - 6)
        pill_x2 = x + tw + 8
        pill_y2 = y

        cv2.rectangle(img, (pill_x1, pill_y1), (pill_x2, pill_y2), color, -1)

        text_org = (pill_x1 + 4, pill_y2 - baseline - 2)
        cv2.putText(
            img,
            label_text,
            text_org,
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    ok, buffer = cv2.imencode(".jpg", img)
    if not ok:
        return Response(status_code=500)

    return Response(content=buffer.tobytes(), media_type="image/jpeg")


@app.get("/sam3-report", response_class=HTMLResponse)
def sam3_report(
    request: Request,
    query: str = Query(...),
    video: str = Query(...),
    min_score: float = Query(0.3, ge=0.0, le=1.0),
    max_frames: int = Query(200, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    """
    HTML contact sheet of SAM3 hits (still available, but not used by the UI).
    """
    frames = (
        db.query(Frame)
        .filter(Frame.video_path == video)
        .order_by(Frame.timestamp)
        .limit(max_frames)
        .all()
    )

    hits: list[dict] = []
    for frame in frames:
        img_path = Path(FRAMES_DIR) / frame.frame_path
        prompts = _split_prompts(query) or [query]

        for prompt in prompts:
            sam_dets = run_sam3_on_path(str(img_path), prompt)

            for det in sam_dets:
                score = float(det["score"])
                if score < min_score:
                    continue

                hits.append(
                    {
                        "frame_id": frame.id,
                        "timestamp": frame.timestamp,
                        "score": score,
                        "prompt": prompt,
                        "img_url": (
                            f"/annotated_sam3/{frame.id}"
                            f"?query={query}&min_score={min_score}"
                        ),
                    }
                )

    hits.sort(key=lambda h: h["timestamp"])

    return templates.TemplateResponse(
        "sam3_report.html",
        {
            "request": request,
            "query": query,
            "video": video,
            "min_score": min_score,
            "hits": hits,
        },
    )


@app.get("/sam3-report-pdf")
def sam3_report_pdf(
    request: Request,
    query: str = Query(...),
    video: str = Query(...),
    min_score: float = Query(0.3, ge=0.0, le=1.0),
    max_frames: int = Query(200, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    """
    Generate a downloadable PDF contact sheet of SAM3 hits
    for a given query + video.

    Each page = one frame with at least one SAM3 hit,
    annotated with filled masks (same semantics as /annotated_sam3).

    Extra: per-frame object counter, e.g.
      "Objects in frame: 7 (yellow car: 4, bus: 3)"
    """
    frames = (
        db.query(Frame)
        .filter(Frame.video_path == video)
        .order_by(Frame.timestamp)
        .limit(max_frames)
        .all()
    )

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    page_w, page_h = A4
    margin = 36  # pt

    prompts = _split_prompts(query) or [query]
    any_page = False

    for frame in frames:
        img_path = Path(FRAMES_DIR) / frame.frame_path
        base = cv2.imread(str(img_path))
        if base is None:
            continue

        overlay = base.copy()
        alpha = 0.45
        any_hit = False
        hit_prompts: set[str] = set()

        # NEW: counters for this frame
        frame_total = 0
        counts_by_prompt: dict[str, int] = {}

        for idx, prompt in enumerate(prompts):
            color = SAM3_COLORS[idx % len(SAM3_COLORS)]  # BGR
            sam_dets = run_sam3_on_path(str(img_path), prompt)

            for det in sam_dets:
                score = float(det["score"])
                if score < float(min_score):
                    continue

                any_hit = True
                hit_prompts.add(prompt)

                # increment counters
                frame_total += 1
                counts_by_prompt[prompt] = counts_by_prompt.get(prompt, 0) + 1

                x, y, w, h = det["bbox"]
                x = int(x)
                y = int(y)
                w = int(w)
                h = int(h)
                x2 = x + w
                y2 = y + h

                mask = det.get("mask")
                if mask is not None:
                    m = mask.astype(bool)
                    overlay[m] = color
                else:
                    cv2.rectangle(overlay, (x, y), (x2, y2), color, -1)

        if not any_hit:
            continue

        any_page = True
        blended = cv2.addWeighted(overlay, alpha, base, 1 - alpha, 0)

        ok, img_buf = cv2.imencode(".jpg", blended)
        if not ok:
            continue
        img_bytes = img_buf.tobytes()
        img_reader = ImageReader(BytesIO(img_bytes))

        # --- header text ---
        c.setFont("Helvetica-Bold", 14)
        c.drawString(
            margin,
            page_h - margin,
            f"SAM3 Report – {Path(video).name}",
        )
        c.setFont("Helvetica", 10)
        c.drawString(
            margin,
            page_h - margin - 16,
            f"Query: {query} | min_score={float(min_score):.2f}",
        )
        c.drawString(
            margin,
            page_h - margin - 30,
            f"Frame time: {frame.timestamp:.2f}s",
        )
        if hit_prompts:
            c.drawString(
                margin,
                page_h - margin - 44,
                "Prompts hit: " + ", ".join(sorted(hit_prompts)),
            )

        # NEW: object counter line
        if frame_total > 0:
            counts_str = ", ".join(
                f"{p}: {counts_by_prompt[p]}" for p in sorted(counts_by_prompt)
            )
            c.drawString(
                margin,
                page_h - margin - 58,
                f"Objects in frame: {frame_total} ({counts_str})",
            )
            header_height = 80
        else:
            header_height = 60

        # --- image placement ---
        max_img_w = page_w - 2 * margin
        max_img_h = page_h - 2 * margin - header_height

        img_w, img_h = img_reader.getSize()
        scale = min(max_img_w / img_w, max_img_h / img_h, 1.0)
        draw_w = img_w * scale
        draw_h = img_h * scale

        x = (page_w - draw_w) / 2
        y = margin

        c.drawImage(img_reader, x, y, width=draw_w, height=draw_h)
        c.showPage()

    # If no pages had hits, create a simple "no hits" page
    if not any_page:
        c.setFont("Helvetica-Bold", 16)
        c.drawString(margin, page_h - margin, "SAM3 Report")
        c.setFont("Helvetica", 12)
        c.drawString(
            margin,
            page_h - margin - 24,
            f"No detections for query '{query}' on video {Path(video).name}.",
        )
        c.showPage()

    c.save()
    buf.seek(0)

    filename = f"sam3_report_{Path(video).stem}.pdf"
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"'
    }
    return Response(
        content=buf.getvalue(),
        media_type="application/pdf",
        headers=headers,
    )