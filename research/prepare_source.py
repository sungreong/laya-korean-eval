"""Download the pinned upstream source; do not vendor it into this repository."""
from pathlib import Path
import io, json, shutil, tempfile, zipfile
import requests

ROOT=Path(__file__).resolve().parents[1]
COMMIT='8a6e1328cce2460a0e5aa348ad465bb1b5821cd2'

def main():
    dest=ROOT/'research/laya'
    marker=dest/'.upstream-revision'
    if dest.exists():
        if marker.exists() and marker.read_text().strip()==COMMIT:
            print('Pinned Laya source already prepared.',flush=True)
            return
        raise RuntimeError('Existing research/laya has no matching revision marker; preserve it and choose a clean checkout.')
    url=f'https://codeload.github.com/NandhaKishorM/laya/zip/{COMMIT}'
    response=requests.get(url,timeout=120);response.raise_for_status()
    with tempfile.TemporaryDirectory(dir=ROOT/'research') as tmp:
        target=Path(tmp).resolve()
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            for entry in archive.infolist():
                member=(target/entry.filename).resolve()
                if not member.is_relative_to(target):
                    raise ValueError('Archive entry escapes extraction directory')
            archive.extractall(target)
        extracted=target/f'laya-{COMMIT}'
        if not (extracted/'laya/__init__.py').is_file():
            raise RuntimeError('Unexpected source archive layout')
        shutil.move(str(extracted),str(dest))
    marker.write_text(COMMIT+'\n')
    print('Prepared Laya',COMMIT,flush=True)

if __name__=='__main__':main()
