import base64
import io
import json
import re
import unicodedata
from pathlib import Path

from PIL import Image, ImageChops, ImageOps, ImageStat


DATA = Path(__file__).resolve().parent / "data"
TAILLE_PHOTO_MAX = 2_000_000


def mots(texte: str) -> set[str]:
    decompose = unicodedata.normalize("NFKD", texte.casefold())
    sans_accents = "".join(c for c in decompose if not unicodedata.combining(c))
    return set(re.findall(r"[a-z0-9]+", sans_accents))


def signature(image: Image.Image) -> Image.Image:
    gris = ImageOps.exif_transpose(image).convert("L")
    masque = gris.point(lambda pixel: 255 if pixel < 160 else 0)
    cadre = masque.getbbox()
    if cadre is None:
        raise ValueError("Pictogramme absent")
    return ImageOps.pad(masque.crop(cadre), (48, 48), color=0)


def chercher_figure(photo_base64: str, figures: list[dict]) -> dict | None:
    if not figures or len(photo_base64) > 4 * ((TAILLE_PHOTO_MAX + 2) // 3):
        return None
    try:
        contenu = base64.b64decode(photo_base64, validate=True)
        if len(contenu) > TAILLE_PHOTO_MAX:
            return None
        with Image.open(io.BytesIO(contenu)) as photo:
            if photo.width * photo.height > 12_000_000:
                return None
            requete = signature(photo)
    except (ValueError, OSError, Image.DecompressionBombError):
        return None

    scores = []
    for figure in figures:
        with Image.open(DATA / figure["fichier"]) as image:
            difference = ImageChops.difference(requete, signature(image))
        scores.append((ImageStat.Stat(difference).mean[0] / 255, figure))
    scores.sort(key=lambda resultat: resultat[0])
    meilleur_score, meilleure_figure = scores[0]
    correspondance_unique = len(scores) == 1 or scores[1][0] - meilleur_score >= 0.06
    if meilleur_score <= 0.18 and correspondance_unique:
        return meilleure_figure
    return None


def repondre(question: str, photo_base64: str | None = None) -> dict:
    manuel = json.loads((DATA / "manuel_px200.json").read_text(encoding="utf-8"))
    termes = mots(question)
    figure = chercher_figure(photo_base64, manuel["figures"]) if photo_base64 else None
    classement = []
    for passage in manuel["passages"]:
        score = len(termes.intersection(passage["mots"]))
        if figure and figure["passage"] == passage["id"]:
            score += 5
        classement.append((score, passage))
    score, passage = max(classement, key=lambda resultat: resultat[0])
    if score == 0:
        return {"reponse": "Précisez le modèle, le code affiché ou le symptôme observé.", "sources": []}

    sources = [f"{manuel['titre']}, p. {passage['page']}, passage {passage['id']}"]
    reponse = passage["extrait"]
    if figure and figure["passage"] == passage["id"]:
        sources.append(f"{manuel['titre']}, p. {figure['page']}, fig. {figure['id']} ({figure['legende']})")
    if passage["id"] == "PX200-V":
        reponse += " Indiquez le code ou joignez une photo où le pictogramme apparaît clairement."
    return {"reponse": reponse, "sources": sources}
