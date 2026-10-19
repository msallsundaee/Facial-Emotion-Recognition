"""Download the public Kaggle folder release without requiring account credentials."""
from pathlib import Path
import hashlib
import json
import urllib.request
import zipfile
from datetime import datetime
from zoneinfo import ZoneInfo

root = Path(__file__).resolve().parents[1] / 'data'
archive = root / 'downloads/fer2013.zip'
url = 'https://www.kaggle.com/api/v1/datasets/download/msambare/fer2013'
if (root / 'train').exists() or (root / 'test').exists():
    raise SystemExit('Dataset folders already exist. Refusing to overwrite source images.')
archive.parent.mkdir(parents=True, exist_ok=True)
if not archive.exists():
    temporary = archive.with_suffix('.partial')
    print(f'Downloading {url}', flush=True)
    urllib.request.urlretrieve(url, temporary)
    temporary.replace(archive)
with zipfile.ZipFile(archive) as zipped:
    for entry in zipped.infolist():
        target = (root / entry.filename).resolve()
        if not target.is_relative_to(root.resolve()):
            raise ValueError('Unsafe path in archive.')
    if zipped.testzip() is not None:
        raise ValueError('Archive checksum error.')
    zipped.extractall(root)
(root / 'source.json').write_text(json.dumps({
    'source': 'https://www.kaggle.com/datasets/msambare/fer2013',
    'download_url': url,
    'download_date': datetime.now(ZoneInfo('Asia/Manila')).date().isoformat(),
    'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
    'notes': 'Kaggle folder-format repackaging; do not claim original PrivateTest-only evaluation.'
}, indent=2))
print('Extracted FER2013 to', root)
