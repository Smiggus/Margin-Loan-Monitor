"""Command line entry point: `mlm <step>` or `mlm all`."""
import argparse

STEPS = {
    "search": "mlm.ingest.edgar.search",      # EDGAR full-text search for pledge disclosures
    "fetch": "mlm.ingest.edgar.fetch",        # download filings, keep pledge sentences
    "extract": "mlm.ingest.edgar.extract",    # sentences -> pledge and lender edges
    "issuers": "mlm.ingest.edgar.issuers",    # SIC industry / sector per issuer
    "prices": "mlm.ingest.prices",            # daily prices, dollar value of each pledge
    "uk": "mlm.ingest.companies_house",       # UK charges for UK-registered holders (layer 2, optional)
    "graph": "mlm.graph.build",               # vertices and edges tables
    "analytics": "mlm.graph.analytics",       # GraphFrames motifs, components, PageRank
}


def main() -> None:
    ap = argparse.ArgumentParser(prog="mlm", description="Share-pledge exposure graph pipeline")
    ap.add_argument("step", choices=[*STEPS, "all"])
    args = ap.parse_args()
    import importlib
    for name in (STEPS if args.step == "all" else [args.step]):
        print(f"== {name}")
        importlib.import_module(STEPS[name]).main()


if __name__ == "__main__":
    main()
