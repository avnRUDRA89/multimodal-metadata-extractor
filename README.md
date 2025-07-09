# 📁 Multimodal Metadata Extractor

This project extracts structured metadata and content from various file types (videos, images, documents, slides, spreadsheets, etc.) and saves the results as JSON files for downstream tasks such as search, indexing, and analysis.

It supports:
- 📹 Scene description + speech transcription from videos
- 🖼 OCR-based image captioning
- 📄 Text extraction from PDFs, Word docs, PPT slides, spreadsheets, and more

---

## 📌 Supported File Types

| Type         | Description                                      |
|--------------|--------------------------------------------------|
| `.mp4`, `.mov`, `.avi`, etc. | Scene descriptions (BLIP), Whisper transcription |
| `.jpg`, `.png`, `.bmp`, etc. | OCR text via Tesseract            |
| `.pdf`       | Text extraction with `pdfplumber`                |
| `.pptx`      | Slide text extraction with `python-pptx`         |
| `.xlsx`      | Sheet parsing with `pandas`                      |
| `.docx`      | Word doc parsing via `docx2txt`                  |
| `.txt`, `.md`, etc. | Plain text parsing                       |

---

## 🛠 Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
