# src/export_diagram.py
"""Izvozi tačan dijagram LangGraph pipeline-a (za izveštaj/PDF).

Crta sam graf koji se zaista izvršava (build_full_graph), pa se dijagram
ne može razminuti sa kodom. Generiše:
  - out/hazop_graph.mmd  (Mermaid tekst — renderuj u PDF/SVG)
  - out/hazop_graph.png  (slika, best-effort: treba internet ili graphviz)

Usage:
  python -m src.export_diagram --outdir out

Pravljenje PDF-a iz izlaza:
  - vektorski (najbolje za štampu):
        npm install -g @mermaid-js/mermaid-cli
        mmdc -i out/hazop_graph.mmd -o hazop_graph.pdf
    ili nalepi sadržaj .mmd u https://mermaid.live -> Export
  - brzo: ubaci out/hazop_graph.png u Word/Docs -> Export as PDF
"""
from __future__ import annotations
import argparse
import os
import sys

from .graph_full import build_full_graph

# Windows konzola (cp1252) ume da pukne na ne-ASCII ispisu -> forsiraj UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Izvoz dijagrama HazopLLM pipeline-a")
    ap.add_argument("--outdir", default="out", help="Izlazni direktorijum (default: out)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    mmd_path = os.path.join(args.outdir, "hazop_graph.mmd")
    png_path = os.path.join(args.outdir, "hazop_graph.png")

    gobj = build_full_graph().compile().get_graph()

    # --- Mermaid tekst (uvek uspeva, ne zahteva mrežu) ---
    mermaid = gobj.draw_mermaid()
    with open(mmd_path, "w", encoding="utf-8") as f:
        f.write(mermaid)
    print(mermaid)
    print(f"\n[OK] Mermaid sacuvan: {mmd_path}")

    # --- PNG slika (best-effort) ---
    png_ok = False
    try:
        png = gobj.draw_mermaid_png()  # koristi mermaid.ink API (treba internet)
        with open(png_path, "wb") as f:
            f.write(png)
        png_ok = True
        print(f"[OK] PNG sacuvan: {png_path}")
    except Exception as e1:
        try:
            png = gobj.draw_png()  # graphviz/pygraphviz fallback (offline)
            with open(png_path, "wb") as f:
                f.write(png)
            png_ok = True
            print(f"[OK] PNG sacuvan (graphviz): {png_path}")
        except Exception as e2:
            print(
                "[!] PNG nije generisan (nema mreze za mermaid.ink, niti graphviz/pygraphviz)."
                f"\n    mermaid.ink greska: {e1}"
                f"\n    graphviz greska:    {e2}"
                f"\n    Mermaid (.mmd) je sacuvan -- renderuj ga u PDF/SVG:"
                f"\n      mmdc -i {mmd_path} -o hazop_graph.pdf"
                "\n    ili nalepi sadrzaj u https://mermaid.live -> Export."
            )

    # --- Rezime / uputstvo za PDF ---
    print("\n=== Za PDF izvestaj ===")
    print(f"  Vektorski: mmdc -i {mmd_path} -o hazop_graph.pdf   (ili mermaid.live -> Export)")
    if png_ok:
        print(f"  Brzo:      ubaci {png_path} u Word/Docs/LaTeX -> Export as PDF")


if __name__ == "__main__":
    main()
