import argparse
import asyncio
import base64
import json
import re
import sys
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from rag import DATA, TAILLE_PHOTO_MAX, mots


SERIE = re.compile(r"\b[A-Z][A-Z0-9]{1,15}-[0-9]{4,16}\b", re.IGNORECASE)
PHOTOS = {"img_px200_01": DATA / "img_px200_01.png"}
QUESTION_DOC = "Mon imprimante PX-200 affiche un voyant clignotant. Que signifie le pictogramme sur la photo jointe ?"
QUESTION_GARANTIE = "Est-ce que mon imprimante est encore sous garantie ? Son numéro de série est PX200-0042."


class Etat(TypedDict, total=False):
    question: str
    photo: str | None
    url_mcp: str | None
    categorie: str
    agent: str
    reponse: str
    sources: list[str]
    parcours: list[str]


def orchestrateur(etat: Etat) -> dict:
    mise_a_jour = {"parcours": [*etat.get("parcours", []), "orchestrateur"]}
    if "reponse" in etat:
        return mise_a_jour
    termes = mots(etat["question"])
    demande_garantie = termes.intersection({"garantie", "garanti", "garantis", "garanties"})
    if demande_garantie or SERIE.fullmatch(etat["question"].strip()):
        return {**mise_a_jour, "categorie": "garantie", "agent": "agent_garantie"}
    return {**mise_a_jour, "categorie": "documentation", "agent": "agent_documentaire"}


def prochaine_etape(etat: Etat) -> str:
    return END if "reponse" in etat else etat["agent"]


@asynccontextmanager
async def transport_mcp(url: str | None):
    if url:
        async with streamable_http_client(url) as (lecture, ecriture, _):
            yield lecture, ecriture
    else:
        serveur = Path(__file__).with_name("serveur_mcp.py")
        parametres = StdioServerParameters(command=sys.executable, args=[str(serveur)])
        async with stdio_client(parametres) as flux:
            yield flux


async def agent_documentaire(etat: Etat) -> dict:
    parcours = [*etat["parcours"], "agent_documentaire"]
    try:
        photo_base64 = charger_photo(etat.get("photo"))
    except (OSError, ValueError):
        return {"reponse": "Joignez une photo lisible de 2 Mo maximum.", "sources": [], "parcours": parcours}
    try:
        async with asyncio.timeout(15):
            async with transport_mcp(etat.get("url_mcp")) as (lecture, ecriture):
                async with ClientSession(lecture, ecriture) as session:
                    await session.initialize()
                    retour = await session.call_tool(
                        "rag_multimodal",
                        {"question": etat["question"], "photo_base64": photo_base64},
                    )
                    resultat = retour.structuredContent
                    if retour.isError or not resultat or not isinstance(resultat.get("reponse"), str):
                        raise RuntimeError("Réponse MCP invalide")
                    reponse = resultat["reponse"]
                    sources = resultat.get("sources", [])
                    if not isinstance(sources, list) or not all(isinstance(source, str) for source in sources):
                        raise RuntimeError("Sources MCP invalides")
    except Exception as erreur:
        print(f"Erreur MCP ({type(erreur).__name__})", file=sys.stderr)
        reponse = "Le service documentaire est indisponible. Réessayez plus tard."
        sources = []
    return {"reponse": reponse, "sources": sources, "parcours": parcours}


def consulter_garantie(numero: str, *, aujourdhui: date | None = None) -> str:
    registre = json.loads((DATA / "garanties.json").read_text(encoding="utf-8"))
    numero = numero.strip().upper()
    appareil = registre.get(numero)
    if appareil is None:
        return f"Le numéro de série {numero} est absent du registre."
    expiration = date.fromisoformat(appareil["expiration"])
    statut = "active" if (aujourdhui or date.today()) <= expiration else "expirée"
    return f"{appareil['modele']} ({numero}) — Garantie {statut}. Date de fin le {expiration:%d/%m/%Y}."


def agent_garantie(etat: Etat) -> dict:
    numero = SERIE.search(etat["question"])
    reponse = consulter_garantie(numero.group()) if numero else "Indiquez le numéro de série de l’appareil."
    return {"reponse": reponse, "sources": [], "parcours": [*etat["parcours"], "agent_garantie"]}


def construire_graphe():
    graphe = StateGraph(Etat)
    graphe.add_node("orchestrateur", orchestrateur)
    graphe.add_node("agent_documentaire", agent_documentaire)
    graphe.add_node("agent_garantie", agent_garantie)
    graphe.add_edge(START, "orchestrateur")
    graphe.add_conditional_edges("orchestrateur", prochaine_etape, ["agent_documentaire", "agent_garantie", END])
    graphe.add_edge("agent_documentaire", "orchestrateur")
    graphe.add_edge("agent_garantie", "orchestrateur")
    return graphe.compile()


def charger_photo(reference: str | None) -> str | None:
    if reference is None:
        return None
    chemin = PHOTOS.get(reference, Path(reference))
    with chemin.open("rb") as fichier:
        contenu = fichier.read(TAILLE_PHOTO_MAX + 1)
    if not contenu or len(contenu) > TAILLE_PHOTO_MAX:
        raise ValueError("La photo doit être non vide et ne pas dépasser 2 Mo.")
    return base64.b64encode(contenu).decode("ascii")


async def main():
    parser = argparse.ArgumentParser(description="Service après-vente PX-200")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true", help="Exécuter les deux exemples de l’énoncé")
    mode.add_argument("--question", help="Traiter une question")
    parser.add_argument("--photo", help="Référence de démonstration ou chemin d’une photo")
    parser.add_argument("--mcp-url", help="URL du serveur MCP Streamable HTTP")
    args = parser.parse_args()
    if args.demo and args.photo:
        parser.error("--photo s’utilise avec --question")

    cas = [(QUESTION_DOC, "img_px200_01"), (QUESTION_GARANTIE, None)] if args.demo else [(args.question, args.photo)]
    graphe = construire_graphe()
    for question, photo in cas:
        etat = await graphe.ainvoke({"question": question, "photo": photo, "url_mcp": args.mcp_url})
        print(f"\n{etat['categorie']} | {etat['agent']}\n{' → '.join(etat['parcours'])}\n{etat['reponse']}")
        for source in etat["sources"]:
            print(f"  - {source}")


if __name__ == "__main__":
    asyncio.run(main())
