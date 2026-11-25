import argparse

from mcp.server.fastmcp import FastMCP

from rag import ReponseDocumentaire, repondre


mcp = FastMCP("documentation-sav", json_response=True, stateless_http=True)


@mcp.tool()
def rag_multimodal(question: str, photo_base64: str | None = None) -> ReponseDocumentaire:
    """Recherche les passages du manuel et, si fournie, compare la photo aux figures."""
    return repondre(question, photo_base64)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", action="store_true", help="Servir MCP sur http://127.0.0.1:8000/mcp")
    args = parser.parse_args()
    mcp.run(transport="streamable-http" if args.http else "stdio")
