#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["axiom-axle>=1.0.2"]
# ///
"""
Verify Lean proofs using AXLE (Axiom Lean Engine).

Usage:
    uv run verify.py                          # verifies EFDAProofs.lean (default)
    uv run verify.py path/to/Proofs.lean      # verifies any Lean file
"""

import asyncio
import sys
from pathlib import Path

from axle import AxleClient

ENV = "lean-4.28.0"

# Propositions we care about for the paper table (Prop 1–4 + support lemmas)
PAPER_THEOREMS = {
    "first_moment_identity",
    "fisher_info_eq_A''",
    "efda_consistency_alpha",
    "efda_consistency_eta_from_Tbar",
    "efda_calibration",
    "efda_efficiency",
    "efda_mse_misspecification",
}


TRIVIAL_BODIES = {"trivial", "True"}


def is_trivial_stub(content: str) -> bool:
    """Return True if the proof body is `True := trivial` or similar stub."""
    return ":= trivial" in content or "trivial\n" in content


def status_icon(okay: bool, is_sorry: bool, content: str) -> str:
    if okay and not is_sorry and not is_trivial_stub(content):
        return "✓ PROVED"
    if okay and is_trivial_stub(content):
        return "† TRIVIAL STUB"
    if okay and is_sorry:
        return "~ SORRY (typechecks)"
    return "✗ FAILED"


async def verify_file(lean_file: Path) -> None:
    content = lean_file.read_text()

    async with AxleClient() as client:

        # ── 1. Full-file check ────────────────────────────────────────────────
        print(f"\n{'═'*60}")
        print(f"  AXLE verification: {lean_file.name}")
        print(f"{'═'*60}\n")

        check = await client.check(content, environment=ENV, ignore_imports=True)
        print(f"File typechecks : {'yes' if check.okay else 'NO'}")
        if check.lean_messages.errors:
            for e in check.lean_messages.errors:
                print(f"  ERROR   : {e}")
        if check.lean_messages.warnings:
            for w in check.lean_messages.warnings[:5]:   # cap at 5
                print(f"  warning : {w}")
        print()

        # ── 2. Extract theorems ───────────────────────────────────────────────
        print("Extracting theorems …")
        extracted = await client.extract_theorems(content, environment=ENV, ignore_imports=True)
        docs = extracted.documents   # dict[str, Document]
        print(f"Found {len(docs)} declarations\n")

        # ── 3. verify_proof for each theorem ─────────────────────────────────
        print(f"{'─'*60}")
        print(f"  {'Theorem':<40} {'sorry?':>7}  {'Result'}")
        print(f"{'─'*60}")

        results = []
        for name, doc in docs.items():
            has_local_deps = bool(
                doc.local_type_dependencies
                or doc.local_value_dependencies
                or doc.local_syntactic_dependencies
            )

            if has_local_deps:
                # AXLE cannot verify this theorem in isolation because it
                # depends on file-local structures/axioms that formal_statement
                # cannot include. Fall back to the file-level typecheck: if the
                # full file compiled (check.okay) and the theorem has no sorry,
                # it is considered proved.
                okay   = check.okay and not doc.is_sorry
                errors = []
            else:
                # No local deps: verify in isolation as normal
                sorry_resp = await client.theorem2sorry(
                    doc.content, environment=ENV, names=[doc.name], ignore_imports=True
                )
                vr = await client.verify_proof(
                    formal_statement=sorry_resp.content,
                    content=doc.content,
                    environment=ENV,
                    ignore_imports=True,
                )
                okay   = vr.okay
                errors = vr.lean_messages.errors

            icon = status_icon(okay, doc.is_sorry, doc.content)
            marker = " ◀ paper" if name in PAPER_THEOREMS else ""
            dep_note = " [file-verified]" if has_local_deps else ""
            print(f"  {name:<40} {'yes' if doc.is_sorry else 'no':>7}  {icon}{dep_note}{marker}")

            results.append({
                "name": name,
                "content": doc.content,
                "is_sorry": doc.is_sorry,
                "okay": okay,
                "errors": errors,
            })

        print(f"{'─'*60}\n")

        # ── 4. Summary for paper theorems ─────────────────────────────────────
        paper_results = [r for r in results if r["name"] in PAPER_THEOREMS]
        proved   = sum(1 for r in paper_results if r["okay"] and not r["is_sorry"] and not is_trivial_stub(r.get("content", "")))
        trivial  = sum(1 for r in paper_results if r["okay"] and is_trivial_stub(r.get("content", "")))
        sorryed  = sum(1 for r in paper_results if r["is_sorry"] and not is_trivial_stub(r.get("content", "")))
        failed   = sum(1 for r in paper_results if not r["okay"])

        print("Paper theorem summary")
        print(f"  Fully proved (no sorry) : {proved}/{len(paper_results)}")
        print(f"  Trivial stub (True)     : {trivial}/{len(paper_results)}")
        print(f"  Sorry (typechecks only) : {sorryed}/{len(paper_results)}")
        print(f"  Failed                  : {failed}/{len(paper_results)}")

        # Any errors worth surfacing
        for r in paper_results:
            if r["errors"]:
                print(f"\n  Errors in {r['name']}:")
                for e in r["errors"]:
                    print(f"    {e}")

        print()


def main() -> None:
    lean_dir = Path(__file__).parent.parent
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
        if not target.is_absolute():
            target = lean_dir / target
    else:
        target = lean_dir / "EFDAProofs.lean"

    if not target.exists():
        print(f"File not found: {target}", file=sys.stderr)
        sys.exit(1)

    asyncio.run(verify_file(target))


if __name__ == "__main__":
    main()
