"""Install verified ID-APT data assets downloaded from the id-apt-v1 release."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from zipfile import ZipFile


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def install(source, research_root):
    root = research_root.resolve()
    metadata = json.loads((source / "idapt-manifest.json").read_text(encoding="utf-8"))
    for asset in metadata["assets"]:
        name = asset["asset"]
        if Path(name).name != name or not name.startswith("idapt-"):
            raise ValueError(f"Unexpected archive name: {name}")
        path = source / name
        if path.stat().st_size != asset["bytes"] or sha256(path) != asset["sha256"]:
            raise ValueError(f"Archive verification failed: {name}")
        with ZipFile(path) as archive:
            members = {entry["path"]: entry for entry in asset["members"]}
            if sorted(archive.namelist()) != sorted(members):
                raise ValueError(f"Archive contents differ from manifest: {name}")
            for member, spec in members.items():
                parts = PurePosixPath(member).parts
                target = (root / member).resolve()
                if parts[:2] != ("data", "reference") or (root / "data" / "reference") not in target.parents:
                    raise ValueError(f"Unexpected output path: {member}")
                if archive.getinfo(member).file_size != spec["bytes"]:
                    raise ValueError(f"Unexpected length: {member}")
                if target.exists():
                    if sha256(target) != spec["sha256"]:
                        raise FileExistsError(f"Existing file has different contents: {target}")
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                h = hashlib.sha256()
                with NamedTemporaryFile(dir=target.parent, delete=False) as temp:
                    temp_path = Path(temp.name)
                try:
                    with temp_path.open("wb") as out, archive.open(member) as data:
                        for block in iter(lambda: data.read(4 * 1024**2), b""):
                            out.write(block)
                            h.update(block)
                    if h.hexdigest() != spec["sha256"]:
                        raise ValueError(f"Graph file failed verification: {member}")
                    os.replace(temp_path, target)
                finally:
                    temp_path.unlink(missing_ok=True)
        print(f"Verified {name}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("download_dir", type=Path)
    parser.add_argument("research_root", type=Path)
    args = parser.parse_args()
    install(args.download_dir, args.research_root)
