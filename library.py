from pathlib import Path

ROOT = Path(__file__).parent
CONTENT_DIR = ROOT / "assets" / "content"
STYLE_DIR = ROOT / "assets" / "styles"
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}

CONTENT_NAMES = {
    "car": "Car", "mirror": "Mirror", "time": "Time", "food": "Food",
    "camera": "Camera", "city": "City", "library": "Library",
    "ocean": "Ocean", "lamp": "Lamp", "cat": "Cat", "person": "Person", 
}

STYLE_META = {
    "fighting_temeraire": ("The Fighting Temeraire", "J.M.W. Turner", "1838"),
    "irises": ("Irises", "Vincent van Gogh", "1889"),
    "mont_sainte_victoire": ("Mont Sainte-Victoire", "Paul Cézanne", "1902–1904"),
    "basket_of_apples": ("The Basket of Apples", "Paul Cézanne", "c. 1893"),
    "sunflowers": ("Sunflowers", "Vincent van Gogh", "1888"),
    "water_lilies": ("Water Lilies (Nymphéas)", "Claude Monet", "c. 1914–1926"),
    "great_wave": ("The Great Wave off Kanagawa", "Katsushika Hokusai", "c. 1831"),
    "san_giorgio": ("San Giorgio Maggiore at Dusk", "Claude Monet", "1908–1912"),
    "starry_night": ("The Starry Night", "Vincent van Gogh", "1889"),
}


def _scan(folder: Path):
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)


def content_library() -> dict[str, str]:
    """display name -> file path"""
    out = {}
    for p in _scan(CONTENT_DIR):
        out[CONTENT_NAMES.get(p.stem, p.stem.replace("_", " ").title())] = str(p)
    return out


def style_library() -> dict[str, dict]:
    """display name -> {artist, year, path}"""
    out = {}
    for p in _scan(STYLE_DIR):
        title, artist, year = STYLE_META.get(
            p.stem, (p.stem.replace("_", " ").title(), "Unknown artist", "")
        )
        out[title] = {"artist": artist, "year": year, "path": str(p)}
    return out


def style_caption(name: str, lib: dict | None = None) -> str:
    lib = lib or style_library()
    item = lib.get(name)
    if not item:
        return "*Custom style image*"
    year = f" · {item['year']}" if item["year"] else ""
    return f"**{name}** — {item['artist']}{year}"
