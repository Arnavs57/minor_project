import os
import xml.etree.ElementTree as ET

# Paths relative to project root (works from repo root or from scripts/)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
RAW_DIR = os.path.join(PROJECT_ROOT, "dataset", "raw")
LABELS_DIR = os.path.join(PROJECT_ROOT, "dataset", "labels")

# Marine-waste targets only: training and metrics should not be diluted by
# non-target VOC labels (e.g. ROV, general bio, timestamps).
CLASS_ORDER = ("plastic", "metal", "rubber", "cloth", "fishing_net")
ALLOWED_CLASSES = frozenset(CLASS_ORDER)

# Names we never export (including normalized forms like "paper" after fixing "papper").
IGNORED_CLASSES = frozenset(
    {
        "bio",
        "rov",
        "paper",
        "papper",
        "timestamp",
        "unknown",
        "wood",
    }
)


def normalize_class_name(raw):
    """
    Map inconsistent VOC strings to one canonical token before allowlist checks.
    "papper" and "paper" both become "paper" (then dropped via ignore list).
    "fishing" is standardized to "fishing_net" to match the project's class name.
    """
    if raw is None:
        return None
    name = raw.strip().lower()
    if name == "papper":
        return "paper"
    if name == "fishing":
        return "fishing_net"
    return name


def class_id(canonical):
    """Stable YOLO class index from the ordered marine-waste list."""
    return CLASS_ORDER.index(canonical)


def convert_box(size, box):
    """
    Pascal VOC uses pixel boxes (xmin, xmax, ymin, ymax). YOLO needs normalized
    center-x, center-y, width, height in [0, 1] relative to image width/height,
    so boxes are comparable across different resolutions.
    """
    w_img, h_img = size[0], size[1]
    dw = 1.0 / w_img
    dh = 1.0 / h_img

    xmin, xmax, ymin, ymax = box
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin

    return (
        x_center * dw,
        y_center * dh,
        w * dw,
        h * dh,
    )


def annotation_disposition(canonical):
    """
    Return "kept" if this normalized label is exported, "ignored" otherwise.
    Anything not in the marine-waste allowlist is skipped so class ids stay aligned
    with data.yaml and training only sees the intended categories.
    """
    if canonical is None or canonical == "":
        return "ignored"
    if canonical in IGNORED_CLASSES:
        return "ignored"
    if canonical in ALLOWED_CLASSES:
        return "kept"
    return "ignored"


def process_xml_file(path):
    """
    Convert one VOC XML to YOLO lines and per-file kept/ignored counts.
    """
    kept = 0
    ignored = 0
    lines = []

    tree = ET.parse(path)
    root = tree.getroot()

    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    for obj in root.findall("object"):
        name_el = obj.find("name")
        raw = name_el.text if name_el is not None else None
        canonical = normalize_class_name(raw)
        disp = annotation_disposition(canonical)

        if disp == "ignored":
            ignored += 1
            continue

        kept += 1

        xmlbox = obj.find("bndbox")
        if xmlbox is None:
            kept -= 1
            ignored += 1
            continue

        xmin = float(xmlbox.find("xmin").text)
        xmax = float(xmlbox.find("xmax").text)
        ymin = float(xmlbox.find("ymin").text)
        ymax = float(xmlbox.find("ymax").text)

        bb = convert_box((width, height), (xmin, xmax, ymin, ymax))
        cid = class_id(canonical)
        lines.append(f"{cid} {' '.join(map(str, bb))}\n")

    return lines, kept, ignored


def main():
    os.makedirs(LABELS_DIR, exist_ok=True)

    kept_total = 0
    ignored_total = 0

    for file in os.listdir(RAW_DIR):
        if not file.endswith(".xml"):
            continue

        path = os.path.join(RAW_DIR, file)
        lines, kept, ignored = process_xml_file(path)
        kept_total += kept
        ignored_total += ignored

        label_path = os.path.join(LABELS_DIR, file.replace(".xml", ".txt"))
        with open(label_path, "w") as f:
            f.writelines(lines)

    with open(os.path.join(PROJECT_ROOT, "classes.txt"), "w") as f:
        for c in CLASS_ORDER:
            f.write(c + "\n")

    print("Final class list (YOLO id order):", list(CLASS_ORDER))
    print("Total annotations kept:", kept_total)
    print("Total annotations ignored:", ignored_total)
    print("Conversion completed.")


if __name__ == "__main__":
    main()
