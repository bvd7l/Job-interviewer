import json
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

extracted_text = pytesseract.image_to_string(
    Image.open("./cv sample/JD.jpg"), lang="eng"
)
text_content = [] 
for page in extracted_text.splitlines(): 
    txt = page.strip()
    if txt: 
        text_content.append(txt)



with open("./data/job_data.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "file_path": "./cv sample/JD.jpg",
            "words_count": len(" ".join(text_content).split()),
            "raw_text": text_content},
        f,
        ensure_ascii=False,
        indent=4,
    )

print(f"Done 2") 