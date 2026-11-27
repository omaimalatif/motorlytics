"""Car pictures: example photo on add (needs internet; tolerated offline), own photo, removal, drawing fallback."""
import io
from pathlib import Path

from _helpers import ROOT, Checks, setup

setup("photos")
from PIL import Image  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from core import auth, car_photos, cars, db  # noqa: E402

if not cars.SRC.exists():                      # e.g. a fresh clone / the online app: the catalogue is not in git
    print(f"  NOTE  skipped: the car catalogue {cars.SRC.name} is not in this checkout (Add a car is manual-only)")
    raise SystemExit(0)
c = Checks("test_photos")
db.init_db()
user, _ = auth.register("Photo Tester", "photos@example.com", "secret123", None)
at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
at.session_state["user_id"] = user["id"]
at.run()

at.switch_page("app_pages/my_car.py").run()
at.selectbox[0].set_value("Suzuki").run()
at.selectbox[1].set_value("Cultus").run()
at.selectbox[2].set_value(2018).run()
[b for b in at.button if b.label == "Add car"][0].click().run()
c.page(at, "car added")
car = db.get_cars(user["id"])[0]
if car["photo_source"] == "wikipedia":
    c.ok(Path(car["photo_path"]).exists() and "licence" in (car["photo_credit"] or ""), "example photo with credit")
else:
    print("  NOTE  no example photo (offline?) - drawing fallback is used:", car["photo_source"])

buf = io.BytesIO()
Image.new("RGB", (3000, 2000), (180, 30, 40)).save(buf, "PNG")
path = car_photos.save_user_photo(car, buf.getvalue())
c.ok(max(Image.open(path).size) == 1280, "own photo resized to 1280 px")
c.ok(car_photos.car_picture(db.get_car(car["id"]))["kind"] == "user", "own photo shown first")
try:
    car_photos.save_user_photo(car, b"not an image")
    c.ok(False, "invalid file rejected")
except ValueError:
    c.ok(True, "invalid file rejected")
at.switch_page("app_pages/my_car.py").run()
c.page(at, "my car with own photo")
car_photos.remove_user_photo(db.get_car(car["id"]))
c.ok(not Path(path).exists(), "own photo file deleted on removal")

other = db.add_car(user["id"], "Xyzzy", "Nomodel", None, 2.0, 4.0, body_type="SUV")
pic = car_photos.car_picture(other)
c.ok(pic["kind"] == "drawing" and pic["path"].endswith("suv.svg"), "unknown model -> body-type drawing")
at.session_state["car_id"] = other["id"]
at.switch_page("app_pages/overview.py").run()
c.page(at, "overview with drawing")
c.done()
