import argparse
import os
import pathlib
import sys

import markdown

STYLE = """
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', Arial, 'Noto Sans', sans-serif; color: #222; line-height: 1.6; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
  h1, h2, h3 { color: #111; }
  pre { background: #f6f8fa; padding: 0.8rem; overflow-x: auto; border-radius: 6px; }
  code { background: #f6f8fa; padding: 0.15rem 0.35rem; border-radius: 4px; }
  table { border-collapse: collapse; }
  th, td { border: 1px solid #ddd; padding: 6px 8px; }
  img { max-width: 100%; height: auto; }
  .toc { background: #f9fafb; border: 1px solid #eee; padding: 0.75rem; border-radius: 6px; }
</style>
"""

TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
{style}
</head>
<body>
{body}
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description="Convert Markdown to standalone HTML")
    ap.add_argument("--inp", "--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--title", dest="title", default="Document")
    args = ap.parse_args()

    md_text = pathlib.Path(args.inp).read_text(encoding="utf-8")
    html_body = markdown.markdown(
        md_text,
        extensions=[
            "toc",
            "fenced_code",
            "tables",
            "sane_lists",
            "attr_list",
            "smarty",
        ],
        output_format="html5",
    )

    html = TEMPLATE.format(title=args.title, style=STYLE, body=html_body)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    print(f"Wrote HTML: {out_path}")


if __name__ == "__main__":
    main()
