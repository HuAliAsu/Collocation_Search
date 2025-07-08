import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import os
import pandas as pd
import threading
from collections import Counter, defaultdict
from hazm import Normalizer, word_tokenize, POSTagger
import traceback
import pickle
from pathlib import Path
import re
import docx

# بخش توابع پردازش متن (text_processing.py)
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
        all_sheets = pd.read_excel(file_path, sheet_name=None, header=None, names=['incorrect', 'correct'],
                                   engine='openpyxl')
        correction_dict = {}
        for _, df_sheet in all_sheets.items():
            df_sheet.dropna(inplace=True)
            correction_dict.update(dict(zip(df_sheet['incorrect'].astype(str), df_sheet['correct'].astype(str))))
        return correction_dict
    except Exception:
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
    end_chars = {'.', '!', '?', ':', '؟'}
    search_limit = min(len(text), ideal_pos + 50)
    search_range = text[:search_limit]

    last_punc_pos = -1
    for char_punc in end_chars:
        pos = search_range.rfind(char_punc)
        if pos > last_punc_pos:
            last_punc_pos = pos

    if last_punc_pos != -1 and last_punc_pos > ideal_pos / 2:
        return last_punc_pos + 1

    space_search_range_end = min(len(text), ideal_pos + 20)
    space_pos = text.rfind(' ', 0, space_search_range_end)

    if space_pos != -1 and space_pos > ideal_pos / 3:
        return space_pos + 1

    return min(ideal_pos, len(text))


def process_paragraphs(paragraphs: list[str], normalizer, max_words: int, ideal_words: int) -> list[str]:
    """
    پاراگراف‌ها را نرمال‌سازی، ادغام، و در صورت نیاز تقسیم می‌کند.
    """
    if normalizer is None:
        raise ValueError("Normalizer cannot be None in process_paragraphs")

    end_chars = {'.', '!', '?', ':', '؟'}
    merged_paragraphs, buffer = [], ""

    for para_text in paragraphs:
        cleaned_para = normalizer.normalize(para_text.strip())
        if not cleaned_para:
            continue

        buffer = (buffer + " " + cleaned_para) if buffer else cleaned_para
        if buffer.endswith(tuple(end_chars)):
            merged_paragraphs.append(buffer)
            buffer = ""

    if buffer:
        merged_paragraphs.append(buffer)

    final_segments = []
    for current_segment in merged_paragraphs:
        words = current_segment.split()
        if len(words) <= max_words:
            final_segments.append(current_segment)
            continue

        temp_paragraph_to_split = current_segment
        while len(temp_paragraph_to_split.split()) > max_words:
            split_point = find_best_split_point(temp_paragraph_to_split, ideal_words, max_words)

            segment_to_add = temp_paragraph_to_split[:split_point].strip()
            if segment_to_add:
                final_segments.append(segment_to_add)
            temp_paragraph_to_split = temp_paragraph_to_split[split_point:].strip()

        if temp_paragraph_to_split:
            final_segments.append(temp_paragraph_to_split)

    return final_segments

# ==============================================================================
# کلاس اصلی برنامه
class TextAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.base_title = "ابزار یکپارچه تحلیل متن"
        self.root.title(self.base_title)
        self.root.geometry("1200x800")
        try:
            self.root.state('zoomed')
        except tk.TclError:
            pass
        self.MAX_WORDS, self.IDEAL_WORDS = 250, 150
        self.tagged_data, self.sentence_mapping, self.last_search_phrase = [], defaultdict(list), ""
        self.direct_phrase_sources = defaultdict(list)
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.cache_path = os.path.join(self.script_dir, 'preprocessed_data.pkl')
        try:
            model_path = os.path.join(self.script_dir, 'pos_tagger.model')
            if not os.path.exists(model_path):
                messagebox.showerror("فایل مدل یافت نشد", f"فایل 'pos_tagger.model' یافت نشد.")
                self.root.destroy();
                return
            self.normalizer = Normalizer()
            self.pos_tagger = POSTagger(model=model_path)
        except Exception as e:
            messagebox.showerror("خطای Hazm", f"خطا در بارگذاری مدل‌های Hazm:\n{e}")
            self.root.destroy();
            return
        # نقشه تگ‌های دستوری به نام‌های فارسی
        self.pos_map = {
            "اسم": {"NOUN", "NOUN,EZ"}, "فعل": {"VERB"}, "صفت": {"ADJ", "ADJ,EZ"},
            "قید": {"ADV"}, "ضمیر": {"PRON"}, "عدد": {"NUM", "NUM,EZ"},
            "حرف اضافه": {"ADP", "ADP,EZ"}, "حرف ربط": {"CCONJ", "SCONJ"},
            "نقطه‌گذاری": {"PUNCT"}, "تعیین‌کننده": {"DET"}, "حرف ندا": {"INTJ"}
        }
        # ایجاد نقشه معکوس برای تبدیل تگ به نام فارسی
        self.reverse_pos_map = {tag: name for name, tags in self.pos_map.items() for tag in tags}

        # *** FIXED ***: متغیر برای منوی کشویی مدل هایلایت با مقدار پیش‌فرض صحیح
        self.highlight_model_var = tk.StringVar(value="مدل ۲ (معکوس کامل)")
        self.current_found_word = None
        self.current_source_sentences_for_export = []
        self._create_widgets()
        self.root.after(100, self._initiate_loading_process)

    def _create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        style = ttk.Style(self.root)
        style.configure("Treeview.Heading", font=('Tahoma', 10, 'bold'), anchor='center')
        style.configure("Treeview", rowheight=25, font=('Tahoma', 9))
        style.configure("Custom.Treeview", background="#cccccc", fieldbackground="#cccccc", foreground="black")

        search_controls_main_frame = ttk.Frame(main_frame, padding="5")
        search_controls_main_frame.pack(fill=tk.X, side=tk.TOP, pady=5)

        self.search_type_var = tk.StringVar(value="کلمات مجاور")
        search_type_choices = ["کلمات مجاور", "عین عبارت کلیدی"]
        self.search_type_combo = ttk.Combobox(search_controls_main_frame, textvariable=self.search_type_var,
                                              values=search_type_choices, state="readonly", width=15, justify='right')
        ttk.Label(search_controls_main_frame, text=":نوع جستجو").pack(side=tk.RIGHT, padx=(2, 0), pady=5)
        self.search_type_combo.pack(side=tk.RIGHT, padx=(0, 5), pady=5)
        self.search_type_combo.bind("<<ComboboxSelected>>", self._on_search_type_change)

        self.collocation_tools_frame = ttk.Frame(search_controls_main_frame, padding="3", relief="groove", borderwidth=1)
        self.collocation_tools_frame.columnconfigure(1, weight=1)
        self.collocation_tools_frame.columnconfigure(3, weight=1)
        self.collocation_tools_frame.columnconfigure(5, weight=1)
        ttk.Label(self.collocation_tools_frame, text="حالت:").grid(row=0, column=0, padx=(0, 1), pady=1,
                                                                        sticky=tk.E)
        self.mode_var = tk.StringVar(value="هر دو")
        self.mode_combo = ttk.Combobox(self.collocation_tools_frame, textvariable=self.mode_var,
                                       values=["هر دو", "کلمه قبلی", "کلمه بعدی"], state="readonly", width=8,
                                       justify='right')
        self.mode_combo.grid(row=0, column=1, padx=(0, 2), pady=1, sticky=tk.EW)
        ttk.Label(self.collocation_tools_frame, text="شرط:").grid(row=0, column=2, padx=(2, 1), pady=1,
                                                                       sticky=tk.E)
        self.condition_var = tk.StringVar(value="فرقی نمی‌کند")
        self.condition_combo = ttk.Combobox(self.collocation_tools_frame, textvariable=self.condition_var,
                                            values=["فرقی نمی‌کند", "حاوی", "شروع با"], state="readonly", width=10,
                                            justify='right')
        self.condition_combo.grid(row=0, column=3, padx=(0, 2), pady=1, sticky=tk.EW)
        self.condition_combo.bind("<<ComboboxSelected>>", self._toggle_condition_entry)
        self.condition_entry = ttk.Entry(self.collocation_tools_frame, width=10, justify='right')
        self.condition_entry.grid(row=0, column=4, padx=(0, 2), pady=1, sticky=tk.EW)
        ttk.Label(self.collocation_tools_frame, text="نقش:").grid(row=0, column=5, padx=(2, 1), pady=1,
                                                                       sticky=tk.E)
        self.pos_var = tk.StringVar(value="هر نقشی")
        pos_options = ["هر نقشی"] + list(self.pos_map.keys())
        self.pos_combo = ttk.Combobox(self.collocation_tools_frame, textvariable=self.pos_var, values=pos_options,
                                      state="readonly", width=10, justify='right')
        self.pos_combo.grid(row=0, column=6, padx=(0, 1), pady=1, sticky=tk.EW)

        self.search_button = ttk.Button(search_controls_main_frame, text="جستجو", command=self._start_search,
                                        state=tk.DISABLED)
        self.search_button.pack(side=tk.RIGHT, padx=(2, 5), pady=5, ipady=2)  # توجه: side به RIGHT تغییر کرد
        self.keyword_entry = ttk.Entry(search_controls_main_frame, justify='right', width=40)
        self.keyword_entry.pack(side=tk.RIGHT, padx=(5, 5), pady=5, fill=tk.X, expand=True)
        # *** FIXED ***: بازگرداندن منوی کشویی مدل‌های هایلایت
        highlight_model_choices = ["بدون هایلایت", "مدل ۱ (عادی)", "مدل ۲ (معکوس کامل)", "مدل ۳ (جفت آخر عادی)",
                                   "مدل ۴ (جفت اول عادی)"]
        self.highlight_model_combo = ttk.Combobox(search_controls_main_frame, textvariable=self.highlight_model_var,
                                                  values=highlight_model_choices, state="readonly", width=20,
                                                  justify='right')
        self.highlight_model_combo.pack(side=tk.LEFT, padx=(10, 5), pady=5)
        self.highlight_model_combo.bind("<<ComboboxSelected>>", self._on_highlight_option_change)
        ttk.Label(search_controls_main_frame, text=":مدل هایلایت").pack(side=tk.LEFT, padx=(2, 0), pady=5)

        self._on_search_type_change()
        self._toggle_condition_entry()

        self.keyword_entry.bind('<Return>', lambda event: self._start_search())
        self.condition_entry.bind('<Return>', lambda event: self._start_search())

        output_pane = ttk.PanedWindow(main_frame, orient=tk.VERTICAL)
        output_pane.pack(fill=tk.BOTH, expand=True, pady=5)
        results_frame = tk.Frame(output_pane, bg='#cccccc', width=380, height=200)

        cols = ("نمونه", "کلمه", "نقش دستوری", "فراوانی", "موقعیت")
        self.results_tree = ttk.Treeview(results_frame, columns=cols, show='headings', style="Custom.Treeview")
        for col in cols:
            self.results_tree.heading(col, text=col, command=lambda c=col: self._sort_treeview(c, False))
            self.results_tree.column(col, anchor=tk.E)
        self.results_tree.column("نمونه", width=400)
        self.results_tree.column("کلمه", width=150)
        self.results_tree.column("نقش دستوری", width=120)
        self.results_tree.column("فراوانی", width=100)
        self.results_tree.column("موقعیت", width=100)

        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.results_tree.bind("<<TreeviewSelect>>", self._on_result_click)
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        output_pane.add(results_frame, weight=1)

        sentences_frame = ttk.LabelFrame(output_pane, text="جملات منبع")
        sentences_frame.configure(style="Custom.TLabelframe")
        style.configure("Custom.TLabelframe", background="#cccccc")
        style.configure("Custom.TLabelframe.Label", background="#cccccc")
        self.source_text = scrolledtext.ScrolledText(sentences_frame, wrap=tk.WORD, state=tk.DISABLED,
                                                     font=("Tahoma", 11), bg="#cccccc")
        self.source_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        output_pane.add(sentences_frame, weight=2)

        self.source_text.tag_configure("rtl_align", justify='right')
        self.source_text.tag_configure("highlight_found", background="#ADD8E6", relief=tk.RAISED, borderwidth=1)

        # *** FIXED ***: بازگرداندن تعریف متغیر و نوار وضعیت
        self.results_count_var = tk.StringVar(value="")
        status_bar_frame = ttk.Frame(self.root, relief=tk.SUNKEN, padding=0)
        status_bar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_var = tk.StringVar(value="آماده برای بارگذاری داده‌ها...")
        status_text_label = ttk.Label(status_bar_frame, textvariable=self.status_var, anchor=tk.W)
        status_text_label.pack(side=tk.LEFT, padx=(5, 0), pady=2)
        status_results_count_label = ttk.Label(status_bar_frame, textvariable=self.results_count_var, anchor=tk.E,
                                               font=('Tahoma', 9, 'bold'))
        status_results_count_label.pack(side=tk.RIGHT, padx=(0, 5), pady=2)

        self.progressbar = ttk.Progressbar(self.root, orient='horizontal', mode='determinate')

        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="پردازش مجدد داده‌ها", command=self._force_reprocess)
        file_menu.add_separator()
        file_menu.add_command(label="خروجی نتایج به اکسل", command=self._export_results_to_excel)
        file_menu.add_command(label="خروجی جملات منبع به اکسل", command=self._export_source_sentences_to_excel)
        menubar.add_cascade(label="فایل", menu=file_menu)

        font_menu = tk.Menu(menubar, tearoff=0)
        font_families = ["Tahoma", "Arial", "Times New Roman", "Courier New", "Dubai"]
        font_sizes = [9, 10, 11, 12, 13, 14, 16, 18, 20, 22]
        self.selected_font_family = tk.StringVar(value="Dubai")
        self.selected_font_size = tk.IntVar(value=13)
        self.selected_rowheight = tk.IntVar(value=40)

        def set_treeview_font():
            font_tuple = (self.selected_font_family.get(), self.selected_font_size.get())
            style = ttk.Style(self.root)
            style.configure("Custom.Treeview", font=font_tuple)
            style.configure("Custom.Treeview", rowheight=self.selected_rowheight.get())

        set_treeview_font()
        font_family_menu = tk.Menu(font_menu, tearoff=0)
        for family in font_families:
            font_family_menu.add_radiobutton(label=family, variable=self.selected_font_family, value=family,
                                             command=set_treeview_font)
        font_menu.add_cascade(label="نوع فونت", menu=font_family_menu)
        font_size_menu = tk.Menu(font_menu, tearoff=0)
        for size in font_sizes:
            font_size_menu.add_radiobutton(label=str(size), variable=self.selected_font_size, value=size,
                                           command=set_treeview_font)
        font_menu.add_cascade(label="سایز فونت", menu=font_size_menu)
        menubar.add_cascade(label="فونت نتایج", menu=font_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="نمایش راهنما", command=self._show_help)
        menubar.add_cascade(label="راهنما", menu=help_menu)
        self.root.config(menu=menubar)

    def _on_highlight_option_change(self, event=None):
        if self.results_tree.selection():
            self._apply_or_remove_highlights()

    def _on_search_type_change(self, event=None):
        search_type = self.search_type_var.get()
        if search_type == "کلمات مجاور":
            self.collocation_tools_frame.pack(side=tk.RIGHT, before=self.keyword_entry, padx=(5, 5), pady=5,
                                                   fill=tk.NONE, expand=False)
        else:
            self.collocation_tools_frame.pack_forget()

    def _show_help(self):
        try:
            readme_path = os.path.join(self.script_dir, "README.md")
            if not os.path.exists(readme_path):
                messagebox.showerror("خطا", "فایل راهنما (README.md) یافت نشد.")
                return
            with open(readme_path, "r", encoding="utf-8") as f:
                readme_content = f.read()
            help_window = tk.Toplevel(self.root)
            help_window.title("راهنمای برنامه")
            help_window.geometry("800x600")
            try:
                help_window.state('zoomed')
            except tk.TclError:
                pass
            text_area = scrolledtext.ScrolledText(help_window, wrap=tk.WORD, font=("Tahoma", 10))
            text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
            text_area.insert(tk.END, readme_content)
            text_area.config(state=tk.DISABLED)
            text_area.tag_configure("rtl_align_help", justify='right')
            text_area.tag_add("rtl_align_help", "1.0", tk.END)
            help_window.transient(self.root)
            help_window.grab_set()
            self.root.wait_window(help_window)
        except Exception as e:
            messagebox.showerror("خطا در نمایش راهنما", f"خطایی رخ داد: {e}")

    def _on_result_click(self, event=None):
        selected_items = self.results_tree.selection()
        if not selected_items: return
        selected_item = selected_items[0]
        values = self.results_tree.item(selected_item, "values")
        item_tags = self.results_tree.item(selected_item, "tags")

        if not values or len(values) < 5 or values[1] == "هیچ نتیجه‌ای یافت نشد.": return

        sample_display_text = values[0]
        term_in_table = values[1]
        position_type = values[4]

        self.current_found_word = None
        sources_to_display = []

        if 'direct_hit' in item_tags:
            self.current_found_word = term_in_table
            key_for_sources = term_in_table
            unique_sources = set(self.direct_phrase_sources.get(key_for_sources, []))
            if not unique_sources:
                sources_to_display = [("جمله‌ای برای عبارت کلیدی یافت نشد.", "")]
            else:
                sources_to_display = sorted(list(unique_sources), key=lambda x: (x[1] is None, x[1], x[0]))
        elif 'substring_hit' in item_tags:
            self.current_found_word = term_in_table
            key_for_sources = term_in_table
            unique_sources = set(self.direct_phrase_sources.get(key_for_sources, []))
            if not unique_sources:
                sources_to_display = [("جمله‌ای برای این کلمه یافت نشد.", "")]
            else:
                sources_to_display = sorted(list(unique_sources), key=lambda x: (x[1] is None, x[1], x[0]))
        elif 'collocation_hit' in item_tags:
            self.current_found_word = term_in_table
            key_for_mapping = (position_type, term_in_table)
            unique_sources = set(self.sentence_mapping.get(key_for_mapping, []))
            if not unique_sources:
                sources_to_display = [("جمله‌ای برای این کلمه هم‌نشین یافت نشد.", "")]
            else:
                sources_to_display = sorted(list(unique_sources), key=lambda x: (x[1] is None, x[1], x[0]))
        else:
            sources_to_display = [("نوع نتیجه نامشخص است.", "")]

        self.current_source_sentences_for_export = sources_to_display
        self._apply_or_remove_highlights()

    # *** MODIFIED ***: تابع بازچینی برای استفاده از مدل معکوس کامل و تضمین فاصله
    def _reorder_text_for_bidi_fix(self, text, phrase_to_highlight, model):
        """
        بخش‌های متن را بر اساس مدل انتخابی برای مقابله با باگ رندرینگ bidi بازچینی می‌کند.
        """
        if not phrase_to_highlight or phrase_to_highlight not in text:
            return text

        try:
            parts = re.split(f'({re.escape(phrase_to_highlight)})', text, flags=re.IGNORECASE)
            text_segments = parts[::2]
            delimiters = parts[1::2]

            reordered_segments = []

            if model == "مدل ۱ (عادی)":
                reordered_segments = text_segments

            elif model == "مدل ۲ (معکوس کامل)":
                reordered_segments = list(reversed(text_segments))

            elif model == "مدل ۳ (جفت آخر عادی)":
                segments = list(text_segments)
                if len(segments) >= 2:
                    last_pair = segments[-2:]
                    other_segments = segments[:-2]
                    reordered_segments = list(reversed(other_segments)) + last_pair
                else:
                    reordered_segments = segments

            elif model == "مدل ۴ (جفت اول عادی)":
                segments = list(text_segments)
                if len(segments) >= 2:
                    first_pair = segments[:2]
                    other_segments = segments[2:]
                    reordered_segments = first_pair + list(reversed(other_segments))
                else:
                    reordered_segments = segments

            # بازسازی رشته نهایی با تضمین وجود فاصله در اطراف جداکننده
            final_text_parts = []
            for i, segment in enumerate(reordered_segments):
                final_text_parts.append(segment.strip())
                if i < len(delimiters):
                    final_text_parts.append(f" {delimiters[i]} ")

            return "".join(final_text_parts)

        except (re.error, IndexError):
            return text

    def _apply_or_remove_highlights(self):
        self.source_text.config(state=tk.NORMAL)
        self.source_text.delete(1.0, tk.END)

        highlight_model = self.highlight_model_var.get()

        for i, (original_sentence, book_path) in enumerate(self.current_source_sentences_for_export):
            if i > 0:
                self.source_text.insert(tk.END, "\n\n---\n\n", "rtl_align")

            is_special_message = original_sentence.startswith("جمله‌ای") or original_sentence == "نوع نتیجه نامشخص است."
            if is_special_message:
                self.source_text.insert(tk.END, f"{original_sentence}\n", "rtl_align")
                continue

            if book_path:
                file_info_start_idx = self.source_text.index(tk.INSERT)
                book_name_ext = f"{Path(book_path).name}.docx"

                if hasattr(self, 'root_folder_path') and self.root_folder_path:
                    full_b_path = self.root_folder_path / f"{book_path}.docx"
                    tag_name = f"book_link_{i}"
                    file_info_display = f"({book_name_ext})\n"

                    self.source_text.insert(tk.INSERT, file_info_display, ("rtl_align", tag_name))
                    end_idx_file_link = self.source_text.index(f"{tk.INSERT}-1c")
                    self.source_text.tag_add(tag_name, file_info_start_idx, end_idx_file_link)
                    self.source_text.tag_config(tag_name, foreground="darkblue", underline=True)
                    self.source_text.tag_bind(tag_name, "<Button-1>",
                                              lambda e, p=str(full_b_path): self._open_file(Path(p)))
                else:
                    self.source_text.insert(tk.END, f"({book_name_ext}) (مسیر فایل در دسترس نیست)\n", "rtl_align")

            sentence_to_display = original_sentence
            if highlight_model != "بدون هایلایت" and self.current_found_word:
                sentence_to_display = self._reorder_text_for_bidi_fix(original_sentence, self.current_found_word, highlight_model)

            self.source_text.insert(tk.END, sentence_to_display, "rtl_align")

        self.source_text.tag_remove("highlight_found", "1.0", tk.END)
        if highlight_model != "بدون هایلایت":
            self._highlight_text_in_range(
                text_widget=self.source_text,
                phrase=self.current_found_word,
                tag="highlight_found",
                start_range="1.0",
                end_range=tk.END
            )

        self.source_text.config(state=tk.DISABLED)

    def _highlight_text_in_range(self, text_widget, phrase, tag, start_range, end_range):
        if not phrase: return
        current_pos = start_range
        while True:
            pos = text_widget.search(phrase, current_pos, stopindex=end_range, nocase=True, regexp=False, elide=False)
            if not pos:
                break

            end_idx_highlight_str = f"{pos}+{len(phrase)}c"
            actual_end_idx_highlight = text_widget.index(end_idx_highlight_str)

            is_start_boundary = (pos == "1.0" or not text_widget.get(f"{pos}-1c", pos).isalnum())

            is_end_boundary = True
            try:
                next_char = text_widget.get(actual_end_idx_highlight, f"{actual_end_idx_highlight}+1c")
                if next_char.isalnum():
                    is_end_boundary = False
            except tk.TclError:
                pass

            if is_start_boundary and is_end_boundary:
                text_widget.tag_add(tag, pos, actual_end_idx_highlight)

            current_pos = actual_end_idx_highlight

    def _open_file(self, file_path: Path):
        try:
            if not file_path.exists():
                messagebox.showerror("فایل یافت نشد", f"فایل در مسیر زیر یافت نشد:\n{file_path}")
                return
            os.startfile(str(file_path))
        except Exception as e:
            messagebox.showerror("خطا در باز کردن فایل", f"امکان باز کردن فایل وجود نداشت:\n{e}")

    def _export_results_to_excel(self):
        if not self.results_tree.get_children():
            messagebox.showinfo("خالی از نتیجه", "هیچ نتیجه‌ای برای خروجی گرفتن وجود ندارد.")
            return

        default_filename = ""
        if self.last_search_phrase:
            safe_phrase = re.sub(r'[\\/*?:"<>|]', "", self.last_search_phrase)
            default_filename = f"{safe_phrase}_نتایج.xlsx"

        file_path = filedialog.asksaveasfilename(
            initialfile=default_filename,
            defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
            title="ذخیره نتایج جستجو در فایل اکسل"
        )
        if not file_path: return
        try:
            data = []
            cols = ("نمونه", "کلمه", "نقش دستوری", "فراوانی", "موقعیت")
            for item_id in self.results_tree.get_children():
                values = self.results_tree.item(item_id, "values")
                if values and len(values) == len(cols) and values[1] != "هیچ نتیجه‌ای یافت نشد.":
                    data.append(dict(zip(cols, values)))
            if not data: messagebox.showinfo("خالی از نتیجه", "هیچ نتیجه معتبری برای خروجی گرفتن وجود ندارد."); return
            df = pd.DataFrame(data)
            df.to_excel(file_path, index=False, engine='openpyxl')
            messagebox.showinfo("موفقیت", f"نتایج با موفقیت در فایل زیر ذخیره شد:\n{file_path}")
        except Exception as e:
            messagebox.showerror("خطا در ذخیره‌سازی", f"خطایی در هنگام ذخیره فایل اکسل رخ داد:\n{e}")

    def _export_source_sentences_to_excel(self):
        if not self.current_source_sentences_for_export or \
                (len(self.current_source_sentences_for_export) == 1 and self.current_source_sentences_for_export[0][
                    0].startswith("جمله‌ای")):
            messagebox.showinfo("خالی از نتیجه", "هیچ جمله منبعی برای خروجی گرفتن وجود ندارد.")
            return

        default_filename = ""
        search_keyword = self.last_search_phrase
        if search_keyword:
            safe_phrase = re.sub(r'[\\/*?:"<>|]', "", search_keyword)
            default_filename = f"{safe_phrase}_جملات_منبع.xlsx"

        file_path = filedialog.asksaveasfilename(
            initialfile=default_filename,
            defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
            title="ذخیره جملات منبع در فایل اکسل"
        )
        if not file_path: return
        try:
            data_to_export = []
            for sentence, book_relative_path_no_ext in self.current_source_sentences_for_export:
                if sentence.startswith("جمله‌ای"): continue
                book_name = f"{Path(book_relative_path_no_ext).name}.docx" if book_relative_path_no_ext else ""
                data_to_export.append({"جمله منبع": sentence, "نام فایل": book_name})
            if not data_to_export: messagebox.showinfo("خالی از نتیجه",
                                                       "هیچ جمله منبع معتبری برای خروجی گرفتن وجود ندارد."); return
            df = pd.DataFrame(data_to_export)
            df.to_excel(file_path, index=False, engine='openpyxl')
            messagebox.showinfo("موفقیت", f"جملات منبع با موفقیت در فایل زیر ذخیره شدند:\n{file_path}")
        except Exception as e:
            messagebox.showerror("خطا در ذخیره‌سازی", f"خطایی در هنگام ذخیره فایل اکسل رخ داد:\n{e}")

    def _prepare_for_loading(self):
        self.search_button.config(state=tk.DISABLED);
        self.progressbar.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 2));
        self.progressbar['value'] = 0

    def _initiate_loading_process(self):
        self._prepare_for_loading()
        if os.path.exists(self.cache_path):
            self._update_status("فایل کش یافت شد. در حال بارگذاری...");
            threading.Thread(target=self._load_from_cache, daemon=True).start()
        else:
            self._update_status("فایل کش یافت نشد. لطفاً منابع را برای پردازش اولیه انتخاب کنید.")
            root_folder_path_str = filedialog.askdirectory(title="پوشه اصلی حاوی کتاب‌ها را انتخاب کنید")
            if not root_folder_path_str:
                self._enable_ui_after_load("عملیات لغو شد.")
                self.root.title(self.base_title)
                return
            self.root_folder_path = Path(root_folder_path_str).resolve()
            correction_path_str = filedialog.askopenfilename(title="فایل اکسل 'لیست اصلاحات' را انتخاب کنید (اختیاری)",
                                                             filetypes=(("Excel Files", "*.xlsx *.xlsm"),
                                                                        ("All files", "*.*")))
            threading.Thread(target=self._process_and_cache_worker,
                             args=(self.root_folder_path, Path(correction_path_str) if correction_path_str else None),
                             daemon=True).start()

    def _load_from_cache(self):
        try:
            with open(self.cache_path, 'rb') as f:
                cache_content = pickle.load(f)
                if isinstance(cache_content, tuple) and len(cache_content) == 2 and isinstance(cache_content[1], Path):
                    self.tagged_data, self.root_folder_path = cache_content
                else:
                    self.tagged_data = cache_content
                    self.root_folder_path = None
                    self.root.after(100, lambda: messagebox.showwarning("فایل کش قدیمی",
                                                                        "فایل کش شما قدیمی است و مسیر پوشه اصلی کتاب‌ها را ندارد. "
                                                                        "برای فعال شدن قابلیت کلیک روی نام فایل، "
                                                                        "لطفاً داده‌ها را با استفاده از منوی فایل و گزینه 'پردازش مجدد داده‌ها' دوباره پردازش کنید."))
            self.root.after(0, self._enable_ui_after_load,
                            f"داده‌ها با موفقیت از کش بارگذاری شد ({len(self.tagged_data)} ردیف).")
        except Exception as e:
            traceback.print_exc();
            self.root.after(0, self._show_generic_error,
                            f"خطا در خواندن فایل کش: {e}. لطفاً با پردازش مجدد، آن را بازسازی کنید.")

    def _force_reprocess(self):
        if messagebox.askyesno("تایید پردازش مجدد",
                               "آیا مطمئن هستید؟\nاین کار فایل کش فعلی را حذف کرده و فرآیند زمان‌بر پردازش تمام کتاب‌ها را دوباره آغاز می‌کند."):
            try:
                if os.path.exists(self.cache_path): os.remove(self.cache_path)
                self.root.title(self.base_title)
                self._initiate_loading_process()
            except Exception as e:
                messagebox.showerror("خطا", f"امکان حذف فایل کش وجود نداشت: {e}")

    def _process_and_cache_worker(self, root_folder: Path, correction_path: Path | None):
        try:
            self.root.after(0, self._update_status, "در حال خواندن لیست اصلاحات...")
            correction_dict = load_correction_list(correction_path)
            self.root_folder_path = root_folder.resolve()

            docx_files = list(self.root_folder_path.rglob('*.docx'))
            if not docx_files:
                self.root.after(0, self._enable_ui_after_load, "هیچ فایل .docx یافت نشد.");
                return

            all_segments_data = []
            total_files = len(docx_files)
            for i, file_path_obj in enumerate(docx_files):
                self.root.after(0, self._update_progress, (i / total_files) * 50,
                                f"پردازش فایل‌ها: {file_path_obj.name}")
                text_content = get_text_from_docx(file_path_obj)
                if text_content:
                    corrected_text = make_corrections_fast(text_content, correction_dict)
                    processed_segments = process_paragraphs(
                        corrected_text.split('\n'), self.normalizer, self.MAX_WORDS, self.IDEAL_WORDS
                    )
                    relative_file_path = file_path_obj.relative_to(self.root_folder_path)
                    for segment_text in processed_segments:
                        all_segments_data.append(
                            {'book_path': str(relative_file_path.with_suffix('')), 'sentence': segment_text})

            temp_tagged_data, total_segments = [], len(all_segments_data)
            for i, item in enumerate(all_segments_data):
                self.root.after(0, self._update_progress, 50 + (i / total_segments) * 50,
                                f"تحلیل دستوری بخش {i + 1} از {total_segments}...")
                tokens = word_tokenize(item['sentence']);
                tagged = self.pos_tagger.tag(tokens)
                temp_tagged_data.append((item['sentence'], tagged, item['book_path']))
            self.tagged_data = temp_tagged_data

            self.root.after(0, self._update_status, "در حال ذخیره داده‌های پردازش‌شده...")
            with open(self.cache_path, 'wb') as f:
                pickle.dump((self.tagged_data, self.root_folder_path), f)
            self.root.after(0, self._enable_ui_after_load,
                            f"پردازش و ذخیره‌سازی با موفقیت انجام شد ({len(self.tagged_data)} ردیف).")
        except Exception as e:
            traceback.print_exc();
            self.root.after(0, self._show_generic_error, f"خطا در پردازش و ذخیره‌سازی: {e}")

    def _enable_ui_after_load(self, message):
        self.progressbar.pack_forget();
        self._update_status(message)
        if self.tagged_data:
            self.search_button.config(state=tk.NORMAL)
        if hasattr(self, 'root_folder_path') and self.root_folder_path:
            self.root.title(f"{self.base_title} - {self.root_folder_path.name}")
        elif "لغو شد" in message or "یافت نشد" in message:
            self.root.title(self.base_title)
        else:
            self.root.title(self.base_title + " (مسیر پوشه نامشخص)")

    def _show_generic_error(self, exc_str):
        self.progressbar.pack_forget();
        self._update_status("خطا در بارگذاری داده‌ها.")
        self.root.title(self.base_title)
        messagebox.showerror("خطای پیش‌بینی نشده", f"خطایی رخ داد:\n\n{exc_str}")

    def _update_progress(self, value, text):
        self.progressbar['value'] = value;
        self.status_var.set(text)

    def _update_status(self, message):
        self.status_var.set(message)

    def _toggle_condition_entry(self, event=None):
        if self.condition_var.get() in ["حاوی", "شروع با"]:
            self.condition_entry.grid()
        else:
            self.condition_entry.grid_remove()

    def _start_search(self, event=None):
        phrase = self.keyword_entry.get().strip();
        if not phrase: messagebox.showwarning("ورودی نامعتبر", "لطفاً عبارت کلیدی را وارد کنید."); return
        self.last_search_phrase = phrase;

        if self.search_type_var.get() == "کلمات مجاور":
            condition_type = self.condition_var.get()
            condition_value = self.condition_entry.get().strip()
            if condition_type != "فرقی نمی‌کند" and not condition_value:
                messagebox.showwarning("ورودی ناقص", f"برای شرط '{condition_type}' باید یک مقدار وارد کنید.");
                return

        self.search_button.config(state=tk.DISABLED);
        self.results_tree.delete(*self.results_tree.get_children())
        self.source_text.config(state=tk.NORMAL);
        self.source_text.delete(1.0, tk.END);
        self.source_text.config(state=tk.DISABLED)
        self.results_count_var.set("");
        self._update_status(f"در حال جستجو برای عبارت '{phrase}'...")

        params = {
            "search_phrase": phrase,
            "mode": self.mode_var.get(),
            "condition_type": self.condition_var.get(),
            "condition_value": self.normalizer.normalize(self.condition_entry.get().strip()),
            "pos_filter": self.pos_var.get()
        }
        threading.Thread(target=self._perform_search, args=(params,), daemon=True).start()

    def _check_filters(self, word, pos, params):
        pos_filter = params["pos_filter"]
        if pos_filter != "هر نقشی":
            required_tags = self.pos_map.get(pos_filter, set())
            if pos not in required_tags: return False
        condition_type = params["condition_type"];
        condition_value = params["condition_value"]
        if condition_type == "حاوی" and condition_value not in word: return False
        if condition_type == "شروع با" and not word.startswith(condition_value): return False
        return True

    def _perform_search(self, params):
        search_type = self.search_type_var.get()
        user_search_phrase = params["search_phrase"]

        direct_phrase_info_list = []
        collocation_results = []
        self.direct_phrase_sources.clear()
        self.sentence_mapping.clear()

        if search_type == "عین عبارت کلیدی":
            normalized_user_phrase = self.normalizer.normalize(user_search_phrase)
            if not normalized_user_phrase.strip():
                self.root.after(0, self._update_ui_with_results, [], []);
                return

            substring_match_counter = Counter()
            for original_sentence, tagged_sentence, book_path_or_name in self.tagged_data:
                normalized_original_sentence = self.normalizer.normalize(original_sentence)
                if normalized_user_phrase in normalized_original_sentence:
                    for word, pos in tagged_sentence:
                        if normalized_user_phrase in self.normalizer.normalize(word):
                            substring_match_counter[(word, pos)] += 1
                            self.direct_phrase_sources[word].append((original_sentence, book_path_or_name))
                            break

            for (found_word, pos), count in substring_match_counter.most_common():
                sample_display = f"{found_word} ({user_search_phrase})"
                friendly_pos = self.reverse_pos_map.get(pos, pos)
                direct_phrase_info_list.append((sample_display, found_word, friendly_pos, count, "تطابق جزئی"))

        elif search_type == "کلمات مجاور":
            search_tokens = word_tokenize(self.normalizer.normalize(user_search_phrase))
            if not search_tokens:
                self.root.after(0, self._update_ui_with_results, [], []);
                return

            search_phrase_str = " ".join(search_tokens)
            phrase_len = len(search_tokens)
            mode = params["mode"]
            before_counter, after_counter = Counter(), Counter()
            exact_match_for_collocation_counter = 0
            exact_match_sources_for_collocation = []
            has_filters = params["pos_filter"] != "هر نقشی" or params["condition_type"] != "فرقی نمی‌کند"

            for original_sentence, tagged_sentence, book_path_or_name in self.tagged_data:
                words = [item[0] for item in tagged_sentence]
                for i in range(len(words) - phrase_len + 1):
                    if words[i:i + phrase_len] == search_tokens:
                        exact_match_for_collocation_counter += 1
                        exact_match_sources_for_collocation.append((original_sentence, book_path_or_name))
                        if mode in ["هر دو", "کلمه قبلی"] and i > 0:
                            prev_word, prev_pos = tagged_sentence[i - 1]
                            if not has_filters or self._check_filters(prev_word, prev_pos, params):
                                before_counter[(prev_word, prev_pos, f"{prev_word} {search_phrase_str}")] += 1
                                self.sentence_mapping[("قبل", prev_word)].append((original_sentence, book_path_or_name))
                        if mode in ["هر دو", "کلمه بعدی"] and i + phrase_len < len(words):
                            next_word, next_pos = tagged_sentence[i + phrase_len]
                            if not has_filters or self._check_filters(next_word, next_pos, params):
                                after_counter[(next_word, next_pos, f"{search_phrase_str} {next_word}")] += 1
                                self.sentence_mapping[("بعد", next_word)].append((original_sentence, book_path_or_name))

            if exact_match_for_collocation_counter > 0:
                direct_phrase_info_list.append(
                    (user_search_phrase, user_search_phrase, "-", exact_match_for_collocation_counter, "عبارت کلیدی"))
                self.direct_phrase_sources[user_search_phrase] = exact_match_sources_for_collocation

            if mode in ["هر دو", "کلمه قبلی"]:
                for (word, pos, sample), count in before_counter.most_common():
                    friendly_pos = self.reverse_pos_map.get(pos, pos)
                    collocation_results.append((sample, word, friendly_pos, count, "قبل"))
            if mode in ["هر دو", "کلمه بعدی"]:
                for (word, pos, sample), count in after_counter.most_common():
                    friendly_pos = self.reverse_pos_map.get(pos, pos)
                    collocation_results.append((sample, word, friendly_pos, count, "بعد"))

        self.root.after(0, self._update_ui_with_results, direct_phrase_info_list, collocation_results)

    def _update_ui_with_results(self, direct_phrase_info_list, collocation_results):
        self.results_tree.delete(*self.results_tree.get_children())
        total_results_count = 0

        if direct_phrase_info_list:
            for item_info in direct_phrase_info_list:
                tag_to_use = 'direct_hit' if item_info[4] == "عبارت کلیدی" else 'substring_hit'
                self.results_tree.insert("", "end", values=item_info, tags=(tag_to_use,))
                total_results_count += 1

        if collocation_results:
            for item in collocation_results:
                self.results_tree.insert("", "end", values=item, tags=('collocation_hit',))
            total_results_count += len(collocation_results)

        if total_results_count == 0:
            self.results_tree.insert("", "end", values=("", "هیچ نتیجه‌ای یافت نشد.", "", "", ""))
            self.results_count_var.set("نتایج: 0")
        else:
            self.results_count_var.set(f"نتایج: {total_results_count}")
            self.root.after(100, lambda: self._sort_treeview('فراوانی', True))

        self._update_status("پردازش کامل شد. آماده برای جستجوی بعدی.")
        self.search_button.config(state=tk.NORMAL)

    def _sort_treeview(self, col, reverse):
        try:
            data = [(self.results_tree.set(item, col), item) for item in self.results_tree.get_children('') if
                    self.results_tree.set(item, 'کلمه') != "هیچ نتیجه‌ای یافت نشد."]

            if col == "فراوانی":
                data.sort(key=lambda t: int(t[0]) if str(t[0]).isdigit() else 0, reverse=reverse)
            else:
                data.sort(key=lambda t: str(t[0]), reverse=reverse)
            for index, (val, item) in enumerate(data): self.results_tree.move(item, '', index)
            self.results_tree.heading(col, command=lambda: self._sort_treeview(col, not reverse))
        except (ValueError, tk.TclError):
            pass


if __name__ == "__main__":
    try:
        from ctypes import windll;

        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass
    root = tk.Tk()
    root.configure(bg='#333333')
    style = ttk.Style(root)
    style.theme_use("clam")
    app = TextAnalyzerApp(root)
    root.mainloop()
