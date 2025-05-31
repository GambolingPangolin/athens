import sys
import ijson

# Usage:
#   cat dataset.json | uv run python scripts/json_to_jsonl.py - output.jsonl
#   or with tar.xz file:
#   tar -xOf dataset.tar.xz  | uv run python scripts/json_to_jsonl.py - output.jsonl
#
# The script reads JSON array elements from stdin if source path is '-', otherwise from given file path.
# Then writes one JSON object per line to the output JSONL file.

def convert_json_to_jsonl(src_path, tgt_path):
    if src_path == '-':
        src_file = sys.stdin
    else:
        src_file = open(src_path, 'r', encoding='utf-8')

    with src_file, open(tgt_path, 'w', encoding='utf-8') as tgt_file:
        # Use ijson to parse the top-level array incrementally
        objects = ijson.items(src_file, 'item')
        for obj in objects:
            tgt_file.write(f'{obj}\n')


def main():
    if len(sys.argv) != 3:
        print(f'Usage: {sys.argv[0]} <source.json or -> <target.jsonl>')
        sys.exit(1)

    src_file = sys.argv[1]
    tgt_file = sys.argv[2]

    convert_json_to_jsonl(src_file, tgt_file)


if __name__ == '__main__':
    main()
