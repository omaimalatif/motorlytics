"""Car pictures (no Streamlit code).

Which picture a car shows, in order:
  1. the user's own photo (uploaded on the My car page), stored under Data/uploads/user<id>/cars/
  2. an example photo of the model from Wikipedia / Wikimedia Commons, downloaded once per model into
     Data/uploads/model_photos/ and always shown with its credit (author, licence, link): these photos are
     free to reuse under Creative Commons / public-domain terms that require attribution. It may show a
     different generation or year than the user's car, so it is labelled "Example photo".
  3. a simple body-type drawing (assets/cars/*.svg, drawn for this project) if there is no photo or no internet.
Only the make and model name are sent to Wikipedia, with a generic app name as user agent.
"""
import html
import io
import json
import re

import requests

from core import db

ASSETS = db.ROOT / "assets" / "cars"
MODEL_CACHE = db.DB_PATH.parent / "uploads" / "model_photos"
USER_AGENT = "Motorlytics-student-app/1.0 (educational project)"
SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"
API = "https://en.wikipedia.org/w/api.php"
CAR_WORDS = re.compile(r"\b(car|cars|vehicle|automobile|suv|crossover|hatchback|sedan|saloon|minivan|mpv|van|"
                       r"pickup|truck|kei|roadster|coupe|supermini)\b", re.I)
MAX_BYTES = 8 * 1024 * 1024
MAX_SIDE = 1280
BODY_ICONS = {"hatchback": "hatchback.svg", "sedan": "sedan.svg", "compact sedan": "sedan.svg", "coupe": "sedan.svg",
              "convertible": "sedan.svg", "station wagon": "hatchback.svg", "suv": "suv.svg", "crossover": "suv.svg",
              "compact suv": "suv.svg", "double cabin": "suv.svg", "pick up": "suv.svg", "mini van": "suv.svg",
              "van": "suv.svg", "mpv": "suv.svg", "micro van": "suv.svg"}


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


def _get(s: requests.Session, url: str, timeout: int, **params) -> requests.Response:
    """GET with up to 3 attempts; waits when Wikimedia says 'too many requests' (429) or is busy (5xx)."""
    import time
    for attempt in range(3):
        r = s.get(url, params=params or None, timeout=timeout)
        if r.status_code in (429, 500, 502, 503, 504) and attempt < 2:
            wait = r.headers.get("retry-after")
            time.sleep(min(float(wait), 10.0) if wait and wait.isdigit() else 2.0 * (attempt + 1))
            continue
        return r
    return r


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _plain(htmltext: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", htmltext or "")).strip()


def _title_variants(make: str, model: str) -> list:
    make, model = (make or "").strip(), (model or "").strip()
    out = []
    for mk in dict.fromkeys([make, make.title(), make.replace(" ", "-"), make.title().replace(" ", "-")]):
        for md in dict.fromkeys([model, model.replace(" Class", "-Class"), model.title()]):
            out.append(f"{mk} {md}")
    return list(dict.fromkeys(out))


def _file_name(url: str) -> str:
    """Commons file name from an image URL. Large originals come as resized copies:
    .../commons/thumb/a/ab/<File>.jpg/3840px-<File>.jpg -> the folder before the last part is the file."""
    parts = url.split("?")[0].split("/")
    name = parts[-2] if "thumb" in parts and len(parts) >= 2 and re.match(r"^\d+px-", parts[-1]) else parts[-1]
    return requests.utils.unquote(name)


def _save_jpeg(data: bytes, path, max_side: int = MAX_SIDE) -> None:
    """Check the bytes are a real image, shrink to max_side and save as JPEG (raises ValueError otherwise)."""
    from PIL import Image, ImageOps, UnidentifiedImageError
    if len(data) > MAX_BYTES:
        raise ValueError("the photo is larger than 8 MB")
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)                 # phone photos: respect the rotation flag
        img.thumbnail((max_side, max_side))
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError("this file is not a readable image") from e
    path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(path, "JPEG", quality=85)


def example_photo(make: str, model: str, timeout: int = 10) -> dict | None:
    """Wikipedia example photo for a model: {'path', 'credit': {...}} from the cache or downloaded once.
    None if no car article / picture is found or the network is unavailable."""
    if not make or not model:
        return None
    slug = _slug(f"{make} {model}")
    img_p, meta_p = MODEL_CACHE / f"{slug}.jpg", MODEL_CACHE / f"{slug}.json"
    if meta_p.exists():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        return {"path": str(img_p), "credit": meta} if meta.get("found") and img_p.exists() else None
    s = _session()
    try:
        page = None
        for title in _title_variants(make, model):
            r = _get(s, SUMMARY + requests.utils.quote(title.replace(" ", "_")), timeout)
            if r.status_code == 404:
                continue
            r.raise_for_status()
            j = r.json()
            if j.get("type") == "standard" and j.get("originalimage") and \
                    CAR_WORDS.search(f"{j.get('description', '')} {j.get('extract', '')[:300]}"):
                page = j
                break
        if not page:
            MODEL_CACHE.mkdir(parents=True, exist_ok=True)
            meta_p.write_text(json.dumps({"found": False, "query": f"{make} {model}"}), encoding="utf-8")
            return None
        fname = _file_name(page["originalimage"]["source"])
        q = _get(s, API, timeout, action="query", titles=f"File:{fname}", prop="imageinfo",
                 iiprop="extmetadata|url", iiurlwidth=800, format="json")
        q.raise_for_status()
        q = q.json()
        info = next(iter(q["query"]["pages"].values()))["imageinfo"][0]
        em = info.get("extmetadata", {})
        img = _get(s, info.get("thumburl") or info["url"], timeout + 10)
        img.raise_for_status()
        _save_jpeg(img.content, img_p, max_side=800)
    except (requests.RequestException, ValueError, KeyError, StopIteration):
        return None                                        # offline or unexpected answer: try again next time
    meta = {"found": True, "query": f"{make} {model}", "article": page.get("title"),
            "article_url": (page.get("content_urls", {}).get("desktop", {}) or {}).get("page"),
            "author": _plain(em.get("Artist", {}).get("value")) or "unknown author",
            "licence": _plain(em.get("LicenseShortName", {}).get("value")) or "see source",
            "licence_url": em.get("LicenseUrl", {}).get("value"), "source_url": info.get("descriptionurl")}
    meta_p.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return {"path": str(img_p), "credit": meta}


def save_user_photo(car: dict, data: bytes) -> str:
    """Store the user's own photo of the car (validated, resized, JPEG). Returns the saved path."""
    path = db.DB_PATH.parent / "uploads" / f"user{car['user_id']}" / "cars" / f"car{car['id']}.jpg"
    _save_jpeg(data, path)
    db.update_car(car["id"], car["user_id"], photo_path=str(path), photo_source="user", photo_credit=None)
    return str(path)


def remove_user_photo(car: dict) -> None:
    """Delete the user's photo; the example photo (or drawing) is shown again."""
    if car.get("photo_source") == "user" and car.get("photo_path"):
        from pathlib import Path
        Path(car["photo_path"]).unlink(missing_ok=True)
    db.update_car(car["id"], car["user_id"], photo_path=None, photo_source=None, photo_credit=None)


def attach_example_photo(car: dict) -> dict:
    """Look up the model's example photo and store it on the car (unless the user has their own photo).
    Records 'none' when nothing is found, so pages do not look it up again on every visit."""
    if car.get("photo_source") == "user":
        return car
    ex = example_photo(car.get("make"), car.get("model"))
    if ex:
        return db.update_car(car["id"], car["user_id"], photo_path=ex["path"], photo_source="wikipedia",
                             photo_credit=json.dumps(ex["credit"]))
    slug = _slug("{} {}".format(car.get("make"), car.get("model")))
    cached_miss = (MODEL_CACHE / (slug + ".json")).exists()
    if cached_miss:                                      # definitely no picture for this model
        return db.update_car(car["id"], car["user_id"], photo_path=None, photo_source="none", photo_credit=None)
    return car                                           # offline: leave unset, try again later


def car_picture(car: dict) -> dict:
    """What to show for a car: {'kind': 'user' | 'example' | 'drawing', 'path', 'caption', 'credit_url'}.
    Looks up the example photo once for cars that have never been checked."""
    from pathlib import Path
    if car and not car.get("photo_source"):
        car = attach_example_photo(car)
    if car and car.get("photo_path") and Path(car["photo_path"]).exists():
        if car.get("photo_source") == "user":
            return {"kind": "user", "path": car["photo_path"], "caption": "Your photo", "credit_url": None}
        cr = json.loads(car.get("photo_credit") or "{}")
        return {"kind": "example", "path": car["photo_path"],
                "caption": f"Example photo of a {car.get('make')} {car.get('model')} (may differ from your car's "
                           f"year). Photo: {cr.get('author', 'unknown')}, {cr.get('licence', '')}, "
                           "via Wikimedia Commons",
                "credit_url": cr.get("source_url")}
    body = ((car or {}).get("body_type") or "").lower()
    icon = ASSETS / BODY_ICONS.get(body, "sedan.svg")
    return {"kind": "drawing", "path": str(icon), "caption": f"{(car or {}).get('body_type') or 'Car'} (drawing)",
            "credit_url": None}
