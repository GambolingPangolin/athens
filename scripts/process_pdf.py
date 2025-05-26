import argparse
import torch
from pathlib import Path
from pdf2image import convert_from_path
from transformers import AutoProcessor, AutoModelForVision2Seq
from docling_core.types.doc import DoclingDocument
from docling_core.types.doc.document import DocTagsDocument

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def convert_pdf_to_docling(
    input_pdf_path: str,
    output_path: str,
    whole_document: bool = False,
    output_markdown: bool = False,
):
    print(f"Loading PDF from: {input_pdf_path}")
    pages = convert_from_path(input_pdf_path, dpi=300)

    print("Loading SmolDocling model and processor...")
    processor = AutoProcessor.from_pretrained("ds4sd/SmolDocling-256M-preview")
    model = AutoModelForVision2Seq.from_pretrained(
        "ds4sd/SmolDocling-256M-preview",
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
    )
    model.to(DEVICE)

    all_doctags = []
    all_images = []

    if not whole_document:
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

    for i, image in enumerate(pages, start=1):
        print(f"Processing page {i}/{len(pages)}...")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Convert this page to docling."},
                ],
            }
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt")
        inputs = inputs.to(DEVICE)

        generated_ids = model.generate(**inputs, max_new_tokens=8192)
        prompt_length = inputs.input_ids.shape[1]
        trimmed_generated_ids = generated_ids[:, prompt_length:]

        doctags = processor.batch_decode(
            trimmed_generated_ids, skip_special_tokens=False
        )[0].lstrip()

        if not whole_document:
            doctags_doc = DocTagsDocument.from_doctags_and_image_pairs(
                [doctags], [image]
            )
            doc = DoclingDocument.load_from_doctags(
                doctags_doc, document_name=f"{Path(input_pdf_path).stem}_page_{i}"
            )

            if output_markdown:
                md_output = doc.export_to_markdown()
                file_path = output_dir / f"page_{i}.md"
                print(f"Saving page {i} as Markdown to: {file_path}")
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(md_output)
            else:
                file_path = output_dir / f"page_{i}.docling"
                print(f"Saving page {i} as native docling to: {file_path}")
                doc.save(file_path)
        else:
            all_doctags.append(doctags)
            all_images.append(image)

    if whole_document:
        print("Creating DocTagsDocument for whole document...")
        doctags_doc = DocTagsDocument.from_doctags_and_image_pairs(
            all_doctags, all_images
        )
        doc = DoclingDocument.load_from_doctags(
            doctags_doc, document_name=Path(input_pdf_path).stem
        )

        if output_markdown:
            print(f"Exporting whole document as Markdown to: {output_path}")
            md_output = doc.export_to_markdown()
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(md_output)
        else:
            print(f"Saving whole document in native docling format to: {output_path}")
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            doc.save(output_path)

    print("Conversion complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF to Docling or Markdown format using SmolDocling."
    )
    parser.add_argument("--input_pdf", type=str, help="Path to the input PDF file")
    parser.add_argument(
        "--output_path",
        type=str,
        help="Output file path (whole document) or directory (per-page)",
    )
    parser.add_argument(
        "--whole-document",
        action="store_true",
        help="Save whole converted document as one file (output path is a file)",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Export output as markdown instead of native docling format",
    )

    args = parser.parse_args()

    if args.whole_document:
        if Path(args.output_path).is_dir():
            parser.error(
                "For --whole-document flag, output path should be a file, not a directory."
            )
    else:
        # Saving per page, ensure output path is a directory or create it
        if Path(args.output_path).exists() and not Path(args.output_path).is_dir():
            parser.error("Without --whole-document, output path should be a directory.")

    convert_pdf_to_docling(
        input_pdf_path=args.input_pdf,
        output_path=args.output_path,
        whole_document=args.whole_document,
        output_markdown=args.markdown,
    )


if __name__ == "__main__":
    main()
