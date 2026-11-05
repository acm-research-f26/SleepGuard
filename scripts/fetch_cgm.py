"""Fetch only CGM members of the public PhysioCGM ZIPs using HTTP byte ranges.

The multi-gigabyte ECG archives are never downloaded in full. Existing files
are preserved. ZIP CRC verification is performed by zipfile when reading.
"""
import argparse
import io
import json
from pathlib import Path
import subprocess
import tempfile
import uuid
import zipfile


class RemoteZip(io.RawIOBase):
    def __init__(self, url, size, max_range=10_000_000):
        self.url, self.size, self.position = url, size, 0
        self.max_range = max_range

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        return self.position

    def tell(self):
        return self.position

    def read(self, size=-1):
        size = min(self.size - self.position, size if size >= 0 else self.size)
        if size <= 0:
            return b""
        if size > self.max_range:
            raise ValueError(f"Refusing a range larger than {self.max_range} bytes")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "range.bin"
            headers = Path(directory) / "headers.txt"
            # A range-specific query avoids caches returning a previous range.
            url = (f"{self.url}?range_start={self.position}&range_size={size}"
                   f"&request_id={uuid.uuid4().hex}")
            subprocess.run([
                "curl", "-fsSL", "--retry", "2", "--max-time", "180",
                "--max-filesize", str(size), "--range",
                f"{self.position}-{self.position + size - 1}",
                "-D", str(headers), "-o", str(output), url,
            ], check=True)
            data = output.read_bytes()
            expected = f"content-range: bytes {self.position}-{self.position + size - 1}/{self.size}"
            if len(data) != size or expected not in headers.read_text().lower():
                raise ValueError("Server did not honor the requested byte range")
        self.position += size
        return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    article = json.loads(args.index.read_text())
    sources = []
    for entry in article["files"]:
        if not entry["name"].endswith("_raw.zip"):
            continue
        subject = entry["name"].split("_")[0]
        destination = args.output / subject / "cgm.csv"
        if destination.exists():
            print(f"Preserved {destination}", flush=True)
            continue
        remote = RemoteZip(entry["download_url"], entry["size"])
        with zipfile.ZipFile(remote) as archive:
            candidates = [item for item in archive.infolist()
                          if Path(item.filename).name.lower() == "cgm.csv"
                          and not item.filename.startswith("__MACOSX/")]
            if len(candidates) != 1 or candidates[0].file_size > 20_000_000:
                raise ValueError(f"Unexpected CGM members in {entry['name']}")
            member = candidates[0]
            content = archive.read(member)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        sources.append({"subject": subject, "url": entry["download_url"],
                        "zip_member": member.filename, "crc32": member.CRC,
                        "bytes": len(content)})
        (args.output / "cgm_download_sources.json").write_text(json.dumps(sources, indent=2))
        print(f"Fetched {subject}: {len(content):,} bytes; ZIP CRC verified", flush=True)


if __name__ == "__main__":
    main()
