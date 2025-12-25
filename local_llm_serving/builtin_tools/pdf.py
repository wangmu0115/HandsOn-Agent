from datetime import datetime
from io import BytesIO

import httpx
import PyPDF2


def parse_pdf(url: str) -> dict:
    """
    Parse a PDF document from URL or local file
    """
    try:
        if url.startswith("file://") or url.startswith("/") or url.startswith("./"):
            file_path = url.replace("file://", "")
            with open(file_path) as f:
                pdf_content = f.read()
        else:
            resp = httpx.get(url, timeout=30.0)
            resp.raise_for_status()
            pdf_content = resp.content
        # Parse PDF
        pdf_file = BytesIO(pdf_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        text_content = []
        for page_num, page in enumerate(pdf_reader.pages, 1):
            text = page.extract_text()
            text_content.append({"page": page_num, "text": text[:1000]})

        return {
            "url": url,
            "num_pages": len(pdf_reader.pages),
            "content": text_content[:5],  # Limit 5 pages
        }
    except Exception as e:
        return {
            "error": str(e),
            "url": url,
            "timestamp": datetime.now(tz=datetime.now().astimezone().tzinfo).strftime("%Y-%m-%d %H:%M:%S%z"),
        }


if __name__ == "__main__":
    print(parse_pdf("https://arxiv.org/pdf/1706.03762"))  # Attention Is All You Need
