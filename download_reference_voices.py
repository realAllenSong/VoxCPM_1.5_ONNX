#!/usr/bin/env python3
import argparse
import html
import json
import os
import re
import urllib.parse
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download official VoxCPM demo prompt voices")
    parser.add_argument("--page-url", default="https://openbmb.github.io/VoxCPM-demopage/", help="Demo page URL")
    parser.add_argument("--output-dir", default="reference", help="Directory to save prompt audio")
    parser.add_argument("--voices-file", default="voices.json", help="voices.json to update")
    parser.add_argument("--max-voices", type=int, default=0, help="Limit number of voices (0 = all)")
    parser.add_argument("--reset", action="store_true", help="Reset voices.json to only official prompts")
    return parser.parse_args()


def clean_text(raw: str) -> str:
    text = re.sub(r"<br\s*/?>", " ", raw, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = " ".join(text.split())
    return text.strip()


def load_existing_voices(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("voices.json must be a JSON object")
    return data


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with urllib.request.urlopen(args.page_url, timeout=30) as r:
        html_text = r.read().decode("utf-8", errors="ignore")

    # Capture audio src and following text in the same table cell.
    pattern = re.compile(
        r"<audio[^>]*>\s*<source\s+src=\"(?P<src>[^\"]+)\"[^>]*/>\s*</audio>(?:<br>)?(?P<text>.*?)</td>",
        re.IGNORECASE | re.DOTALL,
    )

    entries = []
    for match in pattern.finditer(html_text):
        src = match.group("src")
        if "/prompt/" not in src and not ("/emotion/" in src and "_prompt" in src):
            continue
        text = clean_text(match.group("text"))
        entries.append((src, text))

    # Deduplicate by src
    unique_entries = []
    seen = set()
    for src, text in entries:
        if src in seen:
            continue
        seen.add(src)
        unique_entries.append((src, text))

    if args.max_voices > 0:
        unique_entries = unique_entries[: args.max_voices]

    if args.reset:
        existing = load_existing_voices(args.voices_file)
        voices = {}
        if "default" in existing:
            voices["default"] = existing["default"]
    else:
        voices = load_existing_voices(args.voices_file)

    for src, prompt_text in unique_entries:
        url = urllib.parse.urljoin(args.page_url, src)
        filename = os.path.basename(src)
        out_path = os.path.join(args.output_dir, filename)
        if not os.path.isfile(out_path):
            urllib.request.urlretrieve(url, out_path)

        voice_name = os.path.splitext(filename)[0].lower()
        entry = voices.get(voice_name, {}) if isinstance(voices.get(voice_name), dict) else {}
        entry.setdefault("prompt_audio", os.path.join(args.output_dir, filename))
        entry.setdefault("prompt_text", prompt_text)
        entry.setdefault("source_url", url)
        voices[voice_name] = entry

    with open(args.voices_file, "w", encoding="utf-8") as f:
        json.dump(voices, f, ensure_ascii=False, indent=2)

    print(f"Downloaded {len(unique_entries)} prompt voices to {args.output_dir}")
    print(f"Updated {args.voices_file}")


if __name__ == "__main__":
    main()
