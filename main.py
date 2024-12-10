import fitz  # PyMuPDF
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import os

model_id = "vikhyatk/moondream2"
revision = "2024-08-26"
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision
)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

def load_images_from_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    images = []
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images

pdf_path = "docs/2_5366183129474161642.pdf"
pdf_images = load_images_from_pdf(pdf_path)

# Создаем директорию output, если она не существует
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Открываем файл для записи результатов
output_file_path = os.path.join(output_dir, "descriptions.txt")
with open(output_file_path, "w") as output_file:
    for i, image in enumerate(pdf_images):
        enc_image = model.encode_image(image)
        description = model.answer_question(enc_image, "Describe this image.", tokenizer)
        print(f"Image {i + 1}: {description}")
        output_file.write(f"Image {i + 1}: {description}\n")

print(f"Descriptions saved to {output_file_path}")