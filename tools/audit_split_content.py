"""Read-only audit of COCO split identities and exact image-content overlap."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_split(path, root):
    raw = Path(path).read_bytes()
    document = json.loads(raw)
    images = document['images']
    names = [image['file_name'].replace('\\', '/') for image in images]
    files, pixels = defaultdict(list), defaultdict(list)
    missing = []
    from PIL import Image
    for name in names:
        image_path = root / name
        if not image_path.is_file():
            missing.append(name)
            continue
        files[sha(image_path.read_bytes())].append(name)
        with Image.open(image_path) as image:
            rgb = image.convert('RGB')
            digest = sha(str(rgb.size).encode('ascii') + b'\0RGB\0' + rgb.tobytes())
        pixels[digest].append(name)
    summary = dict(images=len(images), annotations=len(document['annotations']),
        json_sha256=sha(raw), categories=document.get('categories', []),
        duplicate_filenames={k:v for k,v in Counter(names).items() if v > 1},
        duplicate_image_ids={str(k):v for k,v in Counter(i['id'] for i in images).items() if v > 1},
        annotation_categories=dict(Counter(str(a['category_id']) for a in document['annotations'])),
        crowd_annotations=sum(bool(a.get('iscrowd', 0)) for a in document['annotations']),
        ignored_annotations=sum(bool(a.get('ignore', 0)) for a in document['annotations']),
        missing_images=missing,
        within_split_identical_pixels=[value for value in pixels.values() if len(value) > 1])
    return summary, set(names), files, pixels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-json', required=True)
    parser.add_argument('--val-json', required=True)
    parser.add_argument('--image-root', required=True)
    args = parser.parse_args()
    root = Path(args.image_root)
    train, train_names, train_files, train_pixels = read_split(args.train_json, root)
    val, val_names, val_files, val_pixels = read_split(args.val_json, root)
    report = dict(train=train, val=val,
        shared_filenames=sorted(train_names & val_names),
        cross_split_identical_files=[dict(train=train_files[k], val=val_files[k])
                                     for k in sorted(set(train_files) & set(val_files))],
        cross_split_identical_pixels=[dict(train=train_pixels[k], val=val_pixels[k])
                                      for k in sorted(set(train_pixels) & set(val_pixels))],
        note='Image IDs are local to each JSON. Exact-content checks do not establish patient independence or exclude near-duplicates.')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
