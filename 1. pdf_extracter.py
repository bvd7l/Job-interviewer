from pypdf import PdfReader 
import json 

cv_path = "./cv sample/abdulrahman-rahhal-5.pdf" 
reader = PdfReader(cv_path) 

text_content = [] 
for page in reader.pages: 
    txt = page.extract_text() 
    if txt: 
        lines = [line.strip() for line in txt.split("\n") if line.strip()]
        text_content.extend(lines)

with open("./data/cv_data.json", "w", encoding="utf-8") as f:
    json.dump(
        {"file_path": cv_path,
        "words_count": len(" ".join(text_content).split()),
        "raw_text": text_content
        },
        f,
        ensure_ascii=False,
        indent=4,
    )

print(f"Done 1") 