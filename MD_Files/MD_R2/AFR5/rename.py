"""
rename_AFR5_R1.py

Scans EVERY subfolder inside AFR5 (and all nested subfolders within
each of them) and renames all files. New folders added under AFR5
later are picked up automatically -- no need to edit this script,
unless a new folder also needs the PCA-style rule below.

Rule per top-level folder under AFR5:
  - PCA: files that start with a number (e.g. "2_something.png")
    KEEP that leading number -> 2AFR5_R2.ext, 3AFR5_R2.ext, etc.
    Files with no leading number fall back to the "replace" rule.
  - Every other folder (Backbone_rmsd, DCCM, and anything else found
    under AFR5): full replace -> AFR5_R2.ext, AFR5_R2_2.ext, ...
    (numbering is per extension, per folder)

PROTECTED_LABELS lists names of files that are already correctly
renamed from a previous pass (e.g. "AFR5_R1") -- those are left
completely untouched and never re-renamed or swept into numbering.

SAFETY: DRY_RUN is True by default -- it only PRINTS what it would
rename, without touching any files. Set DRY_RUN to False once the
preview output looks right, then run again to actually rename.
"""

import re
from pathlib import Path
from collections import defaultdict

DRY_RUN = False

BASE_NAME = "AFR5_R2"

AFR5_ROOT = r"C:\Users\HP\Desktop\SilicoScientia\AntiFungal-Project3\MD_plots_R1_R2\AFR5"

# top-level folder names (directly under AFR5_ROOT) that use the
# "prefix" rule. Everything else under AFR5_ROOT uses "replace".
PREFIX_RULE_FOLDERS = {"PCA"}

# file extensions to leave completely untouched (e.g. helper/automation
# scripts that aren't run-specific output data). Add more if needed,
# e.g. {".py", ".sh", ".ipynb"}
SKIP_EXTENSIONS = {".py"}

# labels from earlier passes that are already correctly named --
# these files are left alone (matches "AFR5_R1", "2AFR5_R1",
# "AFR5_R1_2", etc.). Add more labels here as you add more passes.
PROTECTED_LABELS = {"AFR5_R1"}

LEADING_DIGITS = re.compile(r"^(\d+)")


def is_protected(f: Path) -> bool:
    """True if this file already matches a protected (already-renamed) label."""
    stem = f.stem
    for label in PROTECTED_LABELS:
        if re.match(r"^\d*" + re.escape(label) + r"(_\d+)?$", stem):
            return True
    return False


def plan_replace(files):
    """Full replace, numbered per extension. Returns list of (file, new_name)."""
    plan = []
    by_ext = defaultdict(list)
    for f in files:
        by_ext[f.suffix].append(f)
    for ext, group in by_ext.items():
        group.sort(key=lambda f: f.name)
        for i, f in enumerate(group, start=1):
            new_name = f"{BASE_NAME}{ext}" if i == 1 else f"{BASE_NAME}_{i}{ext}"
            plan.append((f, new_name))
    return plan


def plan_prefix(files):
    """Keep a file's leading number, then AFR5_R1, then extension.
    Files with no leading number fall back to the replace scheme."""
    plan = []
    no_prefix_files = []
    for f in files:
        match = LEADING_DIGITS.match(f.stem)
        if match:
            digits = match.group(1)
            plan.append((f, f"{digits}{BASE_NAME}{f.suffix}"))
        else:
            no_prefix_files.append(f)
    plan.extend(plan_replace(no_prefix_files))
    return plan


def dedupe(plan):
    """Safety net: if two files would land on the same new name, append _2, _3..."""
    seen = {}
    result = []
    for f, new_name in plan:
        if new_name not in seen:
            seen[new_name] = 1
            result.append((f, new_name))
        else:
            stem = Path(new_name).stem
            ext = Path(new_name).suffix
            seen[new_name] += 1
            candidate = f"{stem}_{seen[new_name]}{ext}"
            while candidate in seen:
                seen[new_name] += 1
                candidate = f"{stem}_{seen[new_name]}{ext}"
            seen[candidate] = 1
            result.append((f, candidate))
    return result


def rename_folder(directory: Path, rule: str) -> None:
    all_files = [f for f in directory.iterdir() if f.is_file()]
    skipped = [f for f in all_files if f.suffix.lower() in SKIP_EXTENSIONS]
    protected = [f for f in all_files if f not in skipped and is_protected(f)]
    files = [f for f in all_files if f not in skipped and f not in protected]

    for f in skipped:
        print(f"{f.name}  -- skipped (extension in SKIP_EXTENSIONS)   [{directory}]")
    for f in protected:
        print(f"{f.name}  -- already renamed, left as-is   [{directory}]")

    if not files:
        return

    plan = plan_prefix(files) if rule == "prefix" else plan_replace(files)
    plan = dedupe(plan)

    for f, new_name in plan:
        if f.name == new_name:
            continue
        print(f"{f.name}  ->  {new_name}   [{directory}]")
        if not DRY_RUN:
            f.rename(f.with_name(new_name))


def main() -> None:
    mode = "DRY RUN - no files will be renamed" if DRY_RUN else "LIVE - files WILL be renamed"
    print(f"=== {mode} ===\n")

    root_path = Path(AFR5_ROOT)
    if not root_path.exists():
        print(f"WARNING: folder not found: {AFR5_ROOT}")
        return

    prefix_names_upper = {n.upper() for n in PREFIX_RULE_FOLDERS}

    top_level_dirs = [d for d in root_path.iterdir() if d.is_dir()]
    if not top_level_dirs:
        print(f"No subfolders found in {AFR5_ROOT}")
        return

    for top in top_level_dirs:
        rule = "prefix" if top.name.upper() in prefix_names_upper else "replace"
        all_dirs = [top] + [d for d in top.rglob("*") if d.is_dir()]
        for d in all_dirs:
            rename_folder(d, rule)

    print(
        "\nDone."
        + (" Nothing was actually renamed (dry run)." if DRY_RUN else " Files have been renamed.")
    )


if __name__ == "__main__":
    main()