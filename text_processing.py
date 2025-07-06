import re
import docx
import pandas as pd
from pathlib import Path
# توجه: Normalizer, POSTagger, word_tokenize از hazm در اینجا وارد نمی‌شوند
# بلکه به عنوان آرگومان به توابع پاس داده خواهند شد.

MAX_WORDS_LIB, IDEAL_WORDS_LIB = 250, 150 # استفاده از نام متفاوت برای جلوگیری از تداخل

def get_text_from_docx(file_path: Path) -> str | None:
    """متن را از یک فایل .docx استخراج می‌کند."""
    try:
        doc = docx.Document(file_path)
        return '\n'.join([para.text for para in doc.paragraphs])
    except Exception:
        # در صورت بروز خطا، می‌توان لاگ ثبت کرد یا خطا را مدیریت کرد
        return None

def load_correction_list(file_path: Path | None) -> dict:
    """لیست اصلاحات را از یک فایل اکسل بارگذاری می‌کند."""
    if not file_path or not file_path.exists():
        return {}
    try:
        all_sheets = pd.read_excel(file_path, sheet_name=None, header=None, names=['incorrect', 'correct'], engine='openpyxl')
        correction_dict = {}
        for _, df_sheet in all_sheets.items(): # تغییر نام متغیر df به df_sheet
            df_sheet.dropna(inplace=True)
            correction_dict.update(dict(zip(df_sheet['incorrect'].astype(str), df_sheet['correct'].astype(str))))
        return correction_dict
    except Exception:
        # مدیریت خطا
        return {}

def make_corrections_fast(text: str, correction_dict: dict) -> str:
    """اصلاحات را بر اساس دیکشنری داده شده روی متن اعمال می‌کند."""
    if not correction_dict or not text:
        return text
    # ساخت یک regex واحد برای تمام کلیدها برای بهبود سرعت
    regex = r'\b(' + '|'.join(re.escape(key) for key in correction_dict.keys()) + r')\b'
    return re.sub(regex, lambda m: correction_dict[m.group(0)], text)

def find_best_split_point(text: str, ideal_pos: int, max_words_limit: int) -> int:
    """بهترین نقطه برای تقسیم یک پاراگراف طولانی را پیدا می‌کند."""
    # این تابع کمکی برای _process_paragraphs است
    end_chars = {'.', '!', '?', ':', '؟'}
    # جستجو را کمی فراتر از نقطه ایده‌آل گسترش می‌دهیم
    # اما نه بیشتر از طول متن یا یک محدوده معقول (مثلا ideal_pos + 50)
    search_limit = min(len(text), ideal_pos + 50)
    search_range = text[:search_limit]
    
    last_punc_pos = -1
    for char_punc in end_chars: # تغییر نام متغیر char به char_punc
        pos = search_range.rfind(char_punc)
        if pos > last_punc_pos:
            last_punc_pos = pos
            
    # اگر یک علامت نقطه‌گذاری مناسب پیدا شد و در نیمه دوم بخش جستجو قرار دارد
    if last_punc_pos != -1 and last_punc_pos > ideal_pos / 2 :
        return last_punc_pos + 1

    # اگر علامت نقطه‌گذاری مناسبی پیدا نشد، به دنبال آخرین فاصله بگرد
    # در محدوده‌ای نزدیک به ideal_pos
    # محدود کردن جستجوی فاصله به ideal_pos + 20 برای جلوگیری از برش‌های خیلی بزرگ
    space_search_range_end = min(len(text), ideal_pos + 20)
    space_pos = text.rfind(' ', 0, space_search_range_end)
    
    if space_pos != -1 and space_pos > ideal_pos / 3: # اطمینان از اینکه فاصله خیلی ابتدایی نیست
        return space_pos + 1
        
    # اگر هیچ فاصله یا علامت نقطه‌گذاری مناسبی پیدا نشد،
    # سعی کن در همان حوالی ideal_pos برش بزنی، یا اگر متن کوتاه‌تر است، در انتهای متن
    return min(ideal_pos, len(text))


def process_paragraphs(paragraphs: list[str], normalizer, max_words: int, ideal_words: int) -> list[str]:
    """
    پاراگراف‌ها را نرمال‌سازی، ادغام، و در صورت نیاز تقسیم می‌کند.
    normalizer: یک نمونه از hazm.Normalizer
    """
    if normalizer is None: # بررسی اضافه شده برای اطمینان
        raise ValueError("Normalizer cannot be None in process_paragraphs")

    end_chars = {'.', '!', '?', ':', '؟'}
    merged_paragraphs, buffer = [], ""
    
    for para_text in paragraphs: # تغییر نام متغیر para به para_text
        cleaned_para = normalizer.normalize(para_text.strip()) # نرمال‌سازی در اینجا انجام شود
        if not cleaned_para:
            continue
        
        buffer = (buffer + " " + cleaned_para) if buffer else cleaned_para
        # اگر پاراگراف با یکی از کاراکترهای پایانی تمام شود، آن را به لیست اضافه کن
        if buffer.endswith(tuple(end_chars)):
            merged_paragraphs.append(buffer)
            buffer = ""
            
    if buffer: # اضافه کردن بافر باقیمانده
        merged_paragraphs.append(buffer)
        
    final_segments = []
    for current_segment in merged_paragraphs: # تغییر نام متغیر para به current_segment
        words = current_segment.split() # استفاده از split پیش‌فرض برای شمارش کلمات
        if len(words) <= max_words:
            final_segments.append(current_segment)
            continue
        
        # اگر پاراگراف خیلی طولانی است، آن را تقسیم کن
        temp_paragraph_to_split = current_segment
        while len(temp_paragraph_to_split.split()) > max_words:
            # استفاده از max_words و ideal_words پاس داده شده
            split_point = find_best_split_point(temp_paragraph_to_split, ideal_words, max_words)
            
            segment_to_add = temp_paragraph_to_split[:split_point].strip()
            if segment_to_add: # اطمینان از اینکه بخش جدا شده خالی نیست
                 final_segments.append(segment_to_add)
            temp_paragraph_to_split = temp_paragraph_to_split[split_point:].strip()
            
        if temp_paragraph_to_split: # اضافه کردن بخش باقیمانده پس از تقسیم
            final_segments.append(temp_paragraph_to_split)
            
    return final_segments
