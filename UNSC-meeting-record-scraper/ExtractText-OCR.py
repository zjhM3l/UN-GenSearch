import os
import re
from pdf2image import convert_from_path
import pytesseract
import nltk
from nltk.tokenize import TextTilingTokenizer
from tqdm import tqdm

# 下载 NLTK 数据（如果第一次使用）
nltk.download('punkt')
pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files\Tesseract-OCR\tesseract.exe'


def crop_header_footer(image, header_ratio=0.1, footer_ratio=0.1):
    """
    裁剪图像以去除页眉和页脚。
    参数：
        header_ratio: 从上方裁剪比例（默认10%）
        footer_ratio: 从下方裁剪比例（默认10%）
    返回裁剪后的图像。
    """
    width, height = image.size
    top = int(height * header_ratio)
    bottom = int(height * (1 - footer_ratio))
    return image.crop((0, top, width, bottom))


def extract_cover_info(pdf_file):
    """
    利用 pdfplumber 读取 PDF 封面（第一页），并提取 S/PV、Year、Meeting_Number、Day_Date_Time、
    President、Members 和 Agenda 等信息，返回包含这些信息的字典。
    """
    import pdfplumber
    with pdfplumber.open(pdf_file) as pdf:
        first_page = pdf.pages[0]
        text = first_page.extract_text()

    info = {
        "S/PV": None,
        "Year": None,
        "Meeting_Number": None,
        "Day_Date_Time": None,
        "President": None,
        "Members": None,
        "Agenda": None
    }

    spv_match = re.search(r'S/PV\.\s*\d+', text)
    if spv_match:
        info["S/PV"] = spv_match.group().strip()

    year_match = re.search(r'(?:Provisional\s+)?([Ff]ifty[-\s]?third\s+Year)', text)
    if year_match:
        info["Year"] = year_match.group(1).strip()

    meeting_match = re.search(r'(\d+\s*(?:st|nd|rd|th)?\s*Meeting)', text, re.IGNORECASE)
    if meeting_match:
        info["Meeting_Number"] = meeting_match.group().strip()

    datetime_match = re.search(r'([A-Za-z]+,\s+\d+\s+[A-Za-z]+\s+\d+,\s+\d+\.\d+\s+p\.m\.)', text)
    if datetime_match:
        info["Day_Date_Time"] = datetime_match.group().strip()

    president_match = re.search(r'President:\s*([^(\n]+)', text)
    if president_match:
        info["President"] = president_match.group().strip()

    members_match = re.search(r'Members:\s*(.*?)\s*Agenda', text, re.DOTALL)
    if members_match:
        members_text = members_match.group(1).strip()
        members_text = re.sub(r'\.+', '', members_text)
        info["Members"] = members_text

    agenda_match = re.search(r'Agenda\s*\n\s*(.+)', text)
    if agenda_match:
        info["Agenda"] = agenda_match.group(1).strip()

    return info


# 保持 ocr_pdf 函数不做修改
def ocr_pdf(pdf_file, dpi=300, header_ratio=0.1, footer_ratio=0.1):
    """
    利用 pdf2image 将 PDF 转换为图像，再用 pytesseract 对每页进行 OCR，返回全文文本。
    此处跳过第一页（封面），从第二页开始处理，同时删除页眉和页脚区域（默认上10%和下10%）。
    """
    try:
        images = convert_from_path(
            pdf_file,
            dpi=dpi,
            first_page=2  # 从第2页开始处理，跳过封面
        )
        if not images:
            raise ValueError("没有找到指定页码范围的页面，请检查PDF文件总页数")
    except Exception as e:
        print(f"PDF转换错误: {e}")
        return ""

    ocr_text = ""
    for idx, image in enumerate(images, start=2):
        # 裁剪页眉和页脚
        cropped_image = crop_header_footer(image, header_ratio, footer_ratio)
        text = pytesseract.image_to_string(cropped_image, lang="eng")
        ocr_text += text + '\n'
    return ocr_text


def segment_text_with_texttiling(text, w=12, k=3, smoothing_width=1, cutoff_policy=0.25):
    """
    使用 NLTK 的 TextTilingTokenizer 对文本进行段落分割。
    可调节参数：
        w: 窗口大小，默认12
        k: 块大小，默认3
        smoothing_width: 平滑宽度，默认1
        cutoff_policy: 分段阈值，默认0.25
    """
    try:
        tokenizer = TextTilingTokenizer(w=w, k=k, smoothing_width=smoothing_width, cutoff_policy=cutoff_policy)
    except TypeError:
        print("当前 NLTK 版本不支持自定义参数，使用默认参数进行分段。")
        tokenizer = TextTilingTokenizer()
    segments = tokenizer.tokenize(text)
    return segments


def save_text_to_txt(text, output_path):
    """
    将文本写入到 output_path 指定的文件中。
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text.strip())


def save_segments(segments, output_folder, base_name):
    """
    将每个分段结果保存到 output_folder 中，文件名格式为 “[base_name]-i.txt”
    """
    for i, seg in enumerate(segments, start=1):
        seg_path = os.path.join(output_folder, f"{base_name}-{i}.txt")
        save_text_to_txt(seg, seg_path)


def process_pdf(pdf_file, output_root):
    """
    对单个 PDF 文件进行处理，提取封面和正文内容，并保存到指定的 output_root 下，
    每个 PDF 的提取结果存放在 [PDF文件名]-segments 的子文件夹中。
    """
    base_name = os.path.splitext(os.path.basename(pdf_file))[0]
    output_folder = os.path.join(output_root, f"{base_name}-segments")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 提取封面信息并保存到 “[PDF文件名]-cover.txt”
    cover_info = extract_cover_info(pdf_file)
    cover_text = "\n\n".join([f"{key}: {value}" for key, value in cover_info.items()])
    cover_output_path = os.path.join(output_folder, f"{base_name}-cover.txt")
    save_text_to_txt(cover_text, cover_output_path)

    # 提取正文文本（跳过封面，从第二页开始），并去除页眉和页脚
    full_text = ocr_pdf(pdf_file, dpi=300, header_ratio=0.1, footer_ratio=0.1)
    if not full_text:
        return
    segments = segment_text_with_texttiling(full_text, w=12, k=3, smoothing_width=1, cutoff_policy=0.25)
    save_segments(segments, output_folder, base_name)


def process_all_pdfs(folder):
    """
    读取指定文件夹下所有 PDF 文件进行提取，提取结果放到名为 "text-segments" 的文件夹下，
    每个 PDF 的提取结果存放在 "text-segments\\[PDF文件名]-segments" 的子文件夹中。
    使用 tqdm 实时显示进度，完成每个 PDF 后输出提取完成的信息。
    """
    output_root = "text-segments"
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    checkpoint_file = os.path.join(output_root, "processed.txt")
    processed = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            for line in f:
                processed.add(line.strip())

    pdf_files = [f for f in os.listdir(folder) if f.lower().endswith('.pdf')]
    pdf_paths = [os.path.join(folder, pdf) for pdf in pdf_files if pdf not in processed]

    for pdf_path in tqdm(pdf_paths, desc="Processing PDFs"):
        try:
            process_pdf(pdf_path, output_root)
            with open(checkpoint_file, "a", encoding="utf-8") as f:
                f.write(os.path.basename(pdf_path) + "\n")
            print(f"Finished processing {os.path.basename(pdf_path)}")
        except Exception as e:
            print(f"Error processing {os.path.basename(pdf_path)}: {e}")

    print("All PDF files processed.")


if __name__ == "__main__":
    pdf_folder = "UN_Voting_PDFs-1"  # 请替换为包含 PDF 文件的文件夹路径
    process_all_pdfs(pdf_folder)
