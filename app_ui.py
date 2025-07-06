import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import os
import pandas as pd
import threading
from collections import Counter, defaultdict
from hazm import Normalizer, word_tokenize, POSTagger
import traceback
import pickle
# import docx # دیگر مستقیما اینجا نیاز نیست، به text_processing منتقل شد
# import re # دیگر مستقیما اینجا نیاز نیست، به text_processing منتقل شد
from pathlib import Path
import math
import text_processing # ماژول جدید ما


class TextAnalyzerApp:
    # --- بخش ۱: مقداردهی اولیه و پارامترها ---
    def __init__(self, root):
        self.root = root
        self.root.title("ابزار یکپارچه تحلیل متن")
        self.root.geometry("1200x800")
        try:
            self.root.state('zoomed')
        except tk.TclError:
            pass
        self.MAX_WORDS, self.IDEAL_WORDS = 250, 150 # اینها در کلاس اصلی باقی میمانند و به توابع پاس داده میشوند
        self.tagged_data, self.sentence_mapping, self.last_search_phrase = [], defaultdict(list), ""
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.cache_path = os.path.join(self.script_dir, 'preprocessed_data.pkl')
        try:
            # هزپ برای نرمالایزر و تگر در کلاس اصلی باقی میماند
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
        self.pos_map = {
            "اسم": {"NOUN", "NOUN,EZ"}, "فعل": {"VERB"}, "صفت": {"ADJ", "ADJ,EZ"},
            "قید": {"ADV"}, "ضمیر": {"PRON"}, "عدد": {"NUM", "NUM,EZ"},
            "حرف اضافه": {"ADP", "ADP,EZ"}, "حرف ربط": {"CCONJ", "SCONJ"},
            "نقطه‌گذاری": {"PUNCT"}, "تعیین‌کننده": {"DET"}, "حرف ندا": {"INTJ"}
        }
        self.highlight_enabled_var = tk.BooleanVar(value=True) # متغیر کنترل هایلایت
        self.current_found_word = None # برای نگهداری آخرین کلمه مجاور هایلایت شده
        self.current_source_sentences_for_export = [] # برای نگهداری جملات منبع جهت اکسپورت
        self._create_widgets()
        self.root.after(100, self._initiate_loading_process)

    # --- بخش ۲: ایجاد رابط کاربری (با تغییر) ---
    def _create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        style = ttk.Style(self.root)
        style.configure("Treeview.Heading", font=('Tahoma', 10, 'bold'), anchor='center')
        style.configure("Treeview", rowheight=25, font=('Tahoma', 9))
        style.configure("Custom.Treeview", background="#cccccc", fieldbackground="#cccccc", foreground="black")
        
        # فریم برای دکمه‌های بالا (پردازش مجدد و هایلایت)
        top_buttons_frame = ttk.Frame(main_frame)
        top_buttons_frame.pack(fill=tk.X, pady=(0,5), side=tk.TOP) # تغییر pady

        reprocess_button = ttk.Button(top_buttons_frame, text="پردازش مجدد داده‌ها", command=self._force_reprocess)
        reprocess_button.pack(side=tk.RIGHT, padx=(10,0), pady=5, ipady=5) # تغییر padx

        self.highlight_toggle_button = ttk.Checkbutton(top_buttons_frame, text="هایلایت", # تغییر متن دکمه
                                                       variable=self.highlight_enabled_var,
                                                       command=self._toggle_highlighting_status)
        self.highlight_toggle_button.pack(side=tk.RIGHT, padx=10, pady=5)

        # فریم برای ورودی‌های جستجو
        input_controls_frame = ttk.Frame(main_frame) # تغییر نام از top_frame به input_controls_frame
        input_controls_frame.pack(fill=tk.X, pady=5, side=tk.TOP)
        
        input_frame = ttk.Frame(input_controls_frame, padding="5") # این فریم داخل input_controls_frame قرار میگیرد
        input_frame.pack(fill=tk.X, side=tk.LEFT, expand=True)
        
        ttk.Label(input_frame, text="عبارت کلیدی:").grid(row=0, column=0, padx=(0, 2), pady=5, sticky=tk.W)
        self.keyword_entry = ttk.Entry(input_frame, justify='right')
        self.keyword_entry.grid(row=0, column=1, padx=(0, 5), pady=5, sticky=tk.EW)
        ttk.Label(input_frame, text="حالت:").grid(row=0, column=2, padx=(5, 2), pady=5, sticky=tk.W)
        self.mode_var = tk.StringVar(value="هر دو")
        self.mode_combo = ttk.Combobox(input_frame, textvariable=self.mode_var,
                                       values=["هر دو", "کلمه قبلی", "کلمه بعدی"], state="readonly", width=10,
                                       justify='right')
        self.mode_combo.grid(row=0, column=3, padx=(0, 5), pady=5, sticky=tk.W)
        ttk.Label(input_frame, text="شرط کلمه:").grid(row=0, column=4, padx=(5, 2), pady=5, sticky=tk.W)
        self.condition_var = tk.StringVar(value="فرقی نمی‌کند")
        self.condition_combo = ttk.Combobox(input_frame, textvariable=self.condition_var,
                                            values=["فرقی نمی‌کند", "حاوی", "شروع با"], state="readonly", width=12, # تغییر "کلمه خاص" به "حاوی"
                                            justify='right')
        self.condition_combo.grid(row=0, column=5, padx=(0, 5), pady=5, sticky=tk.W)
        self.condition_combo.bind("<<ComboboxSelected>>", self._toggle_condition_entry)
        self.condition_entry = ttk.Entry(input_frame, width=15, justify='right')
        self.condition_entry.grid(row=0, column=6, padx=(0, 5), pady=5, sticky=tk.W)
        self.condition_entry.grid_remove()
        ttk.Label(input_frame, text="نقش دستوری:").grid(row=0, column=7, padx=(5, 2), pady=5, sticky=tk.W)
        self.pos_var = tk.StringVar(value="هر نقشی")
        pos_options = ["هر نقشی"] + list(self.pos_map.keys())
        self.pos_combo = ttk.Combobox(input_frame, textvariable=self.pos_var, values=pos_options, state="readonly",
                                      width=12, justify='right')
        self.pos_combo.grid(row=0, column=8, padx=(0, 10), pady=5, sticky=tk.W)
        self.search_button = ttk.Button(input_frame, text="جستجو", command=self._start_search, state=tk.DISABLED)
        self.search_button.grid(row=0, column=9, padx=5, pady=5, ipady=5)
        self.results_count_var = tk.StringVar(value="")
        results_count_label = ttk.Label(input_frame, textvariable=self.results_count_var, font=('Tahoma', 9, 'bold'))
        results_count_label.grid(row=0, column=10, padx=(10, 5), pady=5, sticky=tk.W)
        input_frame.columnconfigure(1, weight=1)
        self.keyword_entry.bind('<Return>', lambda event: print("Enter on keyword_entry") or self._start_search())
        self.condition_entry.bind('<Return>', lambda event: print("Enter on condition_entry") or self._start_search())
        output_pane = ttk.PanedWindow(main_frame, orient=tk.VERTICAL)
        output_pane.pack(fill=tk.BOTH, expand=True, pady=5)
        results_frame = tk.Frame(output_pane, bg='#cccccc', width=380, height=200)
        cols = ("نمونه", "کلمه", "فراوانی", "موقعیت")
        self.results_tree = ttk.Treeview(results_frame, columns=cols, show='headings', style="Custom.Treeview")
        for col in cols:
            self.results_tree.heading(col, text=col, command=lambda c=col: self._sort_treeview(c, False))
            self.results_tree.column(col, anchor=tk.E)
        self.results_tree.column("نمونه", width=400)
        self.results_tree.column("کلمه", width=150)
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
        self.source_text.tag_configure("highlight_phrase", background="#FFD700", relief=tk.RAISED, borderwidth=1)
        self.source_text.tag_configure("highlight_found", background="#ADD8E6", relief=tk.RAISED, borderwidth=1)
        self.status_var = tk.StringVar(value="آماده برای بارگذاری داده‌ها...");
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=5)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.progressbar = ttk.Progressbar(self.root, orient='horizontal', mode='determinate')
        # --- اضافه کردن منو برای تغییر فونت نتایج ---
        menubar = tk.Menu(self.root)
        font_menu = tk.Menu(menubar, tearoff=0)
        font_families = ["Tahoma", "Arial", "Times New Roman", "Courier New", "Dubai"]
        font_sizes = [9, 10, 11, 12, 13, 14, 16]
        self.selected_font_family = tk.StringVar(value="Dubai")
        self.selected_font_size = tk.IntVar(value=13)
        self.selected_rowheight = tk.IntVar(value=40)
        # تابع اصلی تغییر فونت و فاصله خطوط هر دو بخش
        def set_treeview_font():
            font_tuple = (self.selected_font_family.get(), self.selected_font_size.get())
            style = ttk.Style(self.root)
            style.configure("Custom.Treeview", font=font_tuple)
            style.configure("Custom.Treeview", rowheight=self.selected_rowheight.get())
        set_treeview_font()
        font_family_menu = tk.Menu(font_menu, tearoff=0)
        for family in font_families:
            font_family_menu.add_radiobutton(label=family, variable=self.selected_font_family, value=family, command=set_treeview_font)
        font_menu.add_cascade(label="نوع فونت", menu=font_family_menu)
        font_size_menu = tk.Menu(font_menu, tearoff=0)
        for size in font_sizes:
            font_size_menu.add_radiobutton(label=str(size), variable=self.selected_font_size, value=size, command=set_treeview_font)
        font_menu.add_cascade(label="سایز فونت", menu=font_size_menu)
        menubar.add_cascade(label="فونت نتایج", menu=font_menu)

        # --- اضافه کردن منوی خروجی ---
        export_menu = tk.Menu(menubar, tearoff=0)
        export_menu.add_command(label="خروجی نتایج به اکسل", command=self._export_results_to_excel)
        export_menu.add_command(label="خروجی جملات منبع به اکسل", command=self._export_source_sentences_to_excel)
        menubar.add_cascade(label="خروجی", menu=export_menu)
        
        self.root.config(menu=menubar)

    # --- بخش ۳: منطق اصلی برنامه (با تغییر کلیدی در on_result_click) ---

    def _on_result_click(self, event=None):
        """
        تابع بازنویسی شده برای نمایش صحیح پاراگراف‌های RTL و ایجاد لینک به فایل.
        """
        selected_items = self.results_tree.selection()
        if not selected_items: return

        selected_item = selected_items[0]
        values = self.results_tree.item(selected_item, "values")
        if not values or not values[1]: return

        _, found_word, _, position = values
        key = (position, found_word)
        self.current_found_word = found_word # ذخیره کلمه مجاور فعلی

        unique_sources = set(self.sentence_mapping.get(key, [("جمله‌ای یافت نشد.", "")]))

        self.source_text.config(state=tk.NORMAL)
        self.source_text.delete(1.0, tk.END)

        # برای حفظ ترتیب ثابت و استفاده در اکسپورت، لیست را مرتب می‌کنیم و ذخیره می‌کنیم
        self.current_source_sentences_for_export = sorted(list(unique_sources), key=lambda x: (x[1] is None, x[1], x[0]))


        for i, (sentence, book_path) in enumerate(self.current_source_sentences_for_export): # استفاده از لیست ذخیره شده
            if i > 0:
                self.source_text.insert(tk.END, "\n\n---\n\n")
            
            if book_path and hasattr(self, 'root_folder_path') and self.root_folder_path:
                book_name_with_extension = f"{Path(book_path).name}.docx"
                full_book_path = self.root_folder_path / f"{book_path}.docx"
                tag_name = f"book_link_{i}"
                self.source_text.insert(tk.END, f"({book_name_with_extension})\n", ("rtl_align", tag_name))
                self.source_text.tag_config(tag_name, foreground="blue", underline=True)
                self.source_text.tag_bind(tag_name, "<Button-1>", lambda e, path=full_book_path: self._open_file(path))
            elif book_path:
                book_name_with_extension = f"{Path(book_path).name}.docx"
                self.source_text.insert(tk.END, f"({book_name_with_extension}) (مسیر فایل در دسترس نیست)\n", "rtl_align")
            elif sentence == "جمله‌ای یافت نشد.":
                 self.source_text.insert(tk.END, f"({sentence})\n", "rtl_align")

            self.source_text.insert(tk.END, sentence, "rtl_align")

        self._apply_or_remove_highlights() # اعمال یا حذف هایلایت‌ها بر اساس وضعیت دکمه

        self.source_text.config(state=tk.DISABLED)

    def _toggle_highlighting_status(self):
        """وضعیت هایلایت را بر اساس چک‌باکس تغییر داده و نمایش را به‌روز می‌کند."""
        self._apply_or_remove_highlights()

    def _apply_or_remove_highlights(self):
        """هایلایت‌ها را بر اساس وضعیت self.highlight_enabled_var اعمال یا حذف می‌کند و مشکل RTL را مدیریت می‌کند."""
        self.source_text.config(state=tk.NORMAL)
        
        # پاک کردن تگ‌های هایلایت قبلی
        self.source_text.tag_remove("highlight_phrase", "1.0", tk.END)
        self.source_text.tag_remove("highlight_found", "1.0", tk.END)

        # دریافت متن فعلی که توسط _on_result_click با محتوای اصلی پر شده است
        # این متن هنوز هایلایت نشده و بازآرایی هم نشده است.
        # نکته: _on_result_click باید اطمینان حاصل کند که متن اصلی را قبل از این تابع قرار داده است.
        # در واقع، _on_result_click ابتدا متن را می‌چیند، سپس این تابع را صدا می‌زند.
        # پس متن موجود در self.source_text در این لحظه، متن اصلی و مرتب شده (بر اساس فایل‌ها) است.

        current_text_content = self.source_text.get("1.0", tk.END).strip()
        # حذف تگ justify راست چین تا بعد از بازچینی دوباره اعمال شود
        self.source_text.tag_remove("rtl_align", "1.0", tk.END) 
        self.source_text.delete("1.0", tk.END) # پاک کردن کامل برای بازچینی

        processed_text_for_display = current_text_content

        if self.highlight_enabled_var.get() and self.current_found_word:
            # فقط اگر هایلایت کلمه مجاور فعال است و کلمه مجاور وجود دارد، بازآرایی را انجام بده
            # این فرض بر این است که مشکل به هم ریختگی فقط با highlight_found رخ می‌دهد.
            processed_text_for_display = self._reorder_text_for_highlighting(current_text_content, self.current_found_word)

        # قرار دادن متن (اصلی یا بازآرایی شده) در ویجت
        # ابتدا تمام متن را با تگ راست‌چین وارد می‌کنیم
        # اطمینان از اینکه در انتهای متن یک newline وجود دارد اگر متن اصلی داشته
        if processed_text_for_display and not processed_text_for_display.endswith('\n'):
            processed_text_for_display += '\n'
        self.source_text.insert(tk.END, processed_text_for_display)
        self.source_text.tag_add("rtl_align", "1.0", tk.END) # اعمال مجدد تگ راست چین به کل متن


        # اعمال هایلایت‌ها (اگر فعال باشند) روی متن جدید (که ممکن است بازآرایی شده باشد)
        if self.highlight_enabled_var.get():
            # if self.last_search_phrase: # هایلایت عبارت کلیدی حذف شد طبق درخواست
            #     self._highlight_text(self.source_text, self.last_search_phrase, "highlight_phrase")
            # هایلایت کلمه مجاور (highlight_found) روی متن بازآرایی شده اعمال می‌شود
            if self.current_found_word:
                self._highlight_text(self.source_text, self.current_found_word, "highlight_found")
        
        self.source_text.config(state=tk.DISABLED)

    def _reorder_text_for_highlighting(self, original_text, highlight_word):
        """
        ترتیب بخش‌های غیرهایلایت شده متن را معکوس می‌کند تا مشکل نمایش RTL در Tkinter را دور بزند.
        مثال: "P1 H P2 H P3" (که H هایلایت است) باید به "P3 H P2 H P1" تبدیل شود.
        """
        if not highlight_word or not original_text: # اضافه کردن بررسی original_text
            return original_text

        # برای جلوگیری از مشکلات با کاراکترهای خاص regex در highlight_word
        escaped_highlight_word = re.escape(highlight_word)
        
        # اگر کلمه هایلایت اصلا در متن نیست، متن اصلی را برگردان
        if not re.search(escaped_highlight_word, original_text):
            return original_text

        text_chunks = []    # بخش‌های متنی که هایلایت نمی‌شوند
        highlight_chunks = [] # خود کلمات هایلایت شده (که ترتیبشان ثابت می‌ماند)

        last_idx = 0
        # re.finditer برای یافتن تمام وقوع‌های غیرهمپوشان highlight_word
        for match in re.finditer(escaped_highlight_word, original_text):
            start_idx, end_idx = match.span()
            text_chunks.append(original_text[last_idx:start_idx]) # متن قبل از هایلایت
            highlight_chunks.append(original_text[start_idx:end_idx]) # خود کلمه هایلایت
            last_idx = end_idx
        text_chunks.append(original_text[last_idx:]) # متن باقیمانده بعد از آخرین هایلایت

        # معکوس کردن ترتیب فقط بخش‌های متنی (غیرهایلایت)
        text_chunks.reverse()

        # بازسازی متن نهایی
        # "P4 H P3 H P2 H P1"
        # text_chunks: [P4, P3, P2, P1]
        # highlight_chunks: [H, H, H]
        # result = P4 + H + P3 + H + P2 + H + P1
        
        reordered_text = ""
        # تعداد بخش‌های متنی همیشه یکی بیشتر از تعداد هایلایت‌هاست
        # (مگر اینکه متن با هایلایت شروع و با هایلایت تمام شود که در آن صورت مساوی هستند،
        # یا اگر اصلا هایلایتی نباشد که در آن صورت یک بخش متنی داریم)

        if not highlight_chunks: # اگر هیچ هایلایتی یافت نشد (نباید اینجا برسیم اگر چک اولیه انجام شده)
            return original_text # یا "".join(text_chunks)

        for i in range(len(highlight_chunks)):
            reordered_text += text_chunks[i]
            reordered_text += highlight_chunks[i]
        
        # اضافه کردن آخرین بخش متنی (که پس از آخرین هایلایت می‌آید یا اگر هایلایتی نبود، کل متن است)
        # اگر text_chunks طولانی‌تر از highlight_chunks باشد (که معمولا هست)
        if len(text_chunks) > len(highlight_chunks):
            reordered_text += text_chunks[-1]
            
        return reordered_text

    def _open_file(self, file_path: Path):
        try:
            # تبدیل شی Path به رشته قبل از ارسال به os.startfile
            os.startfile(str(file_path))
        except FileNotFoundError:
            messagebox.showerror("خطا در باز کردن فایل", f"فایل در مسیر زیر یافت نشد:\n{file_path}")
        except Exception as e:
            messagebox.showerror("خطا در باز کردن فایل", f"امکان باز کردن فایل وجود نداشت:\n{e}")

    def _fixed_map(self, option):
        style = ttk.Style(self.root)
        return [e for e in style.map("Treeview", query_opt=option) if e[:2] != ("!disabled", "!selected")]

    def _export_results_to_excel(self):
        if not self.results_tree.get_children():
            messagebox.showinfo("خالی از نتیجه", "هیچ نتیجه‌ای برای خروجی گرفتن وجود ندارد.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
            title="ذخیره نتایج جستجو در فایل اکسل"
        )
        if not file_path:
            return

        try:
            data = []
            cols = ("نمونه", "کلمه", "فراوانی", "موقعیت")
            for item_id in self.results_tree.get_children():
                values = self.results_tree.item(item_id, "values")
                if values and len(values) == len(cols): # اطمینان از وجود مقادیر و تعداد صحیح آنها
                     # بررسی مورد خاص "هیچ نتیجه‌ای یافت نشد."
                    if values[1] == "هیچ نتیجه‌ای یافت نشد.":
                        continue
                    data.append(dict(zip(cols, values)))
            
            if not data: # اگر پس از فیلتر کردن، داده‌ای باقی نمانده باشد
                messagebox.showinfo("خالی از نتیجه", "هیچ نتیجه معتبری برای خروجی گرفتن وجود ندارد.")
                return

            df = pd.DataFrame(data)
            df.to_excel(file_path, index=False, engine='openpyxl')
            messagebox.showinfo("موفقیت", f"نتایج با موفقیت در فایل زیر ذخیره شد:\n{file_path}")
        except Exception as e:
            messagebox.showerror("خطا در ذخیره‌سازی", f"خطایی در هنگام ذخیره فایل اکسل رخ داد:\n{e}")

    def _export_source_sentences_to_excel(self):
        if not self.current_source_sentences_for_export or \
           (len(self.current_source_sentences_for_export) == 1 and self.current_source_sentences_for_export[0][0] == "جمله‌ای یافت نشد."):
            messagebox.showinfo("خالی از نتیجه", "هیچ جمله منبعی برای خروجی گرفتن وجود ندارد.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
            title="ذخیره جملات منبع در فایل اکسل"
        )
        if not file_path:
            return

        try:
            data_to_export = []
            for sentence, book_relative_path_no_ext in self.current_source_sentences_for_export:
                if sentence == "جمله‌ای یافت نشد.": # از این مورد صرف نظر کن
                    continue
                book_name = ""
                if book_relative_path_no_ext:
                    book_name = f"{Path(book_relative_path_no_ext).name}.docx"
                data_to_export.append({"جمله منبع": sentence, "نام فایل": book_name})
            
            if not data_to_export: # اگر پس از فیلتر کردن، داده‌ای باقی نمانده باشد
                 messagebox.showinfo("خالی از نتیجه", "هیچ جمله منبع معتبری برای خروجی گرفتن وجود ندارد.")
                 return

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
            # ذخیره مسیر پوشه اصلی
            root_folder_path_str = filedialog.askdirectory(title="پوشه اصلی حاوی کتاب‌ها را انتخاب کنید")
            if not root_folder_path_str:
                self._enable_ui_after_load("عملیات لغو شد.")
                return
            self.root_folder_path = Path(root_folder_path_str).resolve() # ذخیره مسیر مطلق

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
                # بررسی فرمت جدید کش که شامل مسیر پوشه اصلی است
                if isinstance(cache_content, tuple) and len(cache_content) == 2 and isinstance(cache_content[1], Path):
                    self.tagged_data, self.root_folder_path = cache_content
                else: # سازگاری با فرمت کش قدیمی
                    self.tagged_data = cache_content # یا cache_content[0] اگر مطمئنید که همیشه یک تاپل بوده
                    self.root_folder_path = None # نشان می‌دهد که مسیر پوشه در کش نیست
                    # نمایش هشدار به کاربر پس از بارگذاری کامل UI
                    self.root.after(100, lambda: messagebox.showwarning("فایل کش قدیمی",
                                                                       "فایل کش شما قدیمی است و مسیر پوشه اصلی کتاب‌ها را ندارد. "
                                                                       "برای فعال شدن قابلیت کلیک روی نام فایل در بخش جملات منبع، "
                                                                       "لطفاً داده‌ها را با استفاده از دکمه 'پردازش مجدد داده‌ها' دوباره پردازش کنید."))
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
                self._initiate_loading_process()
            except Exception as e:
                messagebox.showerror("خطا", f"امکان حذف فایل کش وجود نداشت: {e}")

    # توابع _get_text_from_docx, _load_correction_list, _make_corrections_fast, 
    # _process_paragraphs, _find_best_split_point به text_processing.py منتقل شدند.

    def _process_and_cache_worker(self, root_folder: Path, correction_path: Path | None):
        try:
            self.root.after(0, self._update_status, "در حال خواندن لیست اصلاحات...")
            # استفاده از تابع منتقل شده
            correction_dict = text_processing.load_correction_list(correction_path)
            
            if not hasattr(self, 'root_folder_path') or not self.root_folder_path:
                 self.root_folder_path = root_folder.resolve()
            else: 
                 self.root_folder_path = Path(self.root_folder_path).resolve()

            docx_files = list(self.root_folder_path.rglob('*.docx'))
            if not docx_files:
                self.root.after(0, self._enable_ui_after_load, "هیچ فایل .docx یافت نشد.");
                return

            all_segments_data, total_files = [], len(docx_files) # تغییر نام all_segments به all_segments_data
            for i, file_path_obj in enumerate(docx_files): # تغییر نام file_path به file_path_obj
                self.root.after(0, self._update_progress, (i / total_files) * 50, f"پردازش فایل‌ها: {file_path_obj.name}")
                # استفاده از تابع منتقل شده
                text_content = text_processing.get_text_from_docx(file_path_obj)
                if text_content:
                    # نرمال‌سازی و اصلاحات قبل از ارسال به process_paragraphs انجام می‌شود
                    # normalized_text = self.normalizer.normalize(text_content) # نرمالایزر خود کلاس استفاده میشود
                    # corrected_text = text_processing.make_corrections_fast(normalized_text, correction_dict)
                    # processed_segments = text_processing.process_paragraphs(
                    #     corrected_text.split('\n'), 
                    #     self.normalizer, # ارسال نرمالایزر نمونه از کلاس
                    #     self.MAX_WORDS, 
                    #     self.IDEAL_WORDS
                    # )
                    
                    # --- بازنگری در ترتیب عملیات ---
                    # 1. خواندن متن خام
                    # 2. اعمال اصلاحات سریع روی متن خام (اگر لازم است قبل از نرمالایزر باشد)
                    # 3. ارسال متن (احتمالا اصلاح شده) به process_paragraphs که خودش نرمالایز هم میکند.
                    
                    corrected_text = text_processing.make_corrections_fast(text_content, correction_dict)
                    processed_segments = text_processing.process_paragraphs(
                        corrected_text.split('\n'), # ارسال پاراگراف‌های خام (پس از اصلاح اولیه)
                        self.normalizer,           # نرمالایزر داخل process_paragraphs اعمال می‌شود
                        self.MAX_WORDS,
                        self.IDEAL_WORDS
                    )

                    relative_file_path = file_path_obj.relative_to(self.root_folder_path)
                    for segment_text in processed_segments: # تغییر نام segment به segment_text
                        all_segments_data.append({'book_path': str(relative_file_path.with_suffix('')), 'sentence': segment_text})

            temp_tagged_data, total_segments = [], len(all_segments_data) # اطمینان از استفاده از all_segments_data
            for i, item in enumerate(all_segments_data): # اطمینان از استفاده از all_segments_data
                self.root.after(0, self._update_progress, 50 + (i / total_segments) * 50,
                                f"تحلیل دستوری بخش {i + 1} از {total_segments}...")
                tokens = word_tokenize(item['sentence']);
                tagged = self.pos_tagger.tag(tokens)
                # 'book_path' شامل مسیر نسبی بدون پسوند است
                temp_tagged_data.append((item['sentence'], tagged, item['book_path']))
            self.tagged_data = temp_tagged_data

            self.root.after(0, self._update_status, "در حال ذخیره داده‌های پردازش‌شده...")
            with open(self.cache_path, 'wb') as f:
                pickle.dump((self.tagged_data, self.root_folder_path), f)
            self.root.after(0, self._enable_ui_after_load,
                            f"پردازش و ذخیره‌سازی با موفقیت انجام شد ({len(self.tagged_data)} ردیف).")
        except Exception as e:
            traceback.print_exc();
            if not hasattr(self, 'root_folder_path'): # اطمینان از وجود self.root_folder_path
                self.root_folder_path = None
            self.root.after(0, self._show_generic_error, f"خطا در پردازش و ذخیره‌سازی: {e}")

    def _enable_ui_after_load(self, message):
        self.progressbar.pack_forget();
        self._update_status(message)
        if self.tagged_data: self.search_button.config(state=tk.NORMAL)

    def _show_generic_error(self, exc_str):
        self.progressbar.pack_forget();
        self._update_status("خطا در بارگذاری داده‌ها.")
        messagebox.showerror("خطای پیش‌بینی نشده", f"خطایی رخ داد:\n\n{exc_str}")

    def _update_progress(self, value, text):
        self.progressbar['value'] = value; self.status_var.set(text)

    def _update_status(self, message):
        self.status_var.set(message)

    def _toggle_condition_entry(self, event=None):
        if self.condition_var.get() in ["حاوی", "شروع با"]: # تغییر "کلمه خاص" به "حاوی"
            self.condition_entry.grid()
        else:
            self.condition_entry.grid_remove()

    def _start_search(self, event=None): # event پارامتر را اضافه میکنیم تا از bind هم دریافت شود
        print(f"_start_search called by event: {event}") # برای دیباگ
        phrase = self.keyword_entry.get().strip();
        if not phrase: messagebox.showwarning("ورودی نامعتبر", "لطفاً عبارت کلیدی را وارد کنید."); return
        self.last_search_phrase = phrase;
        normalized_phrase = self.normalizer.normalize(phrase);
        search_tokens = word_tokenize(normalized_phrase)
        if not search_tokens: return
        condition_type = self.condition_var.get();
        condition_value = self.condition_entry.get().strip()
        if condition_type != "فرقی نمی‌کند" and not condition_value: messagebox.showwarning("ورودی ناقص",
                                                                                            f"برای شرط '{condition_type}' باید یک مقدار وارد کنید."); return
        self.search_button.config(state=tk.DISABLED);
        self.results_tree.delete(*self.results_tree.get_children())
        self.source_text.config(state=tk.NORMAL);
        self.source_text.delete(1.0, tk.END);
        self.source_text.config(state=tk.DISABLED)
        self.results_count_var.set("");
        self._update_status(f"در حال جستجو برای عبارت '{phrase}'...")
        params = {"search_tokens": search_tokens, "mode": self.mode_var.get(), "condition_type": condition_type,
                  "condition_value": self.normalizer.normalize(condition_value), "pos_filter": self.pos_var.get()}
        threading.Thread(target=self._perform_search, args=(params,), daemon=True).start()

    def _check_filters(self, word, pos, params):
        pos_filter = params["pos_filter"]
        if pos_filter != "هر نقشی":
            required_tags = self.pos_map.get(pos_filter, set())
            if pos not in required_tags: return False
        condition_type = params["condition_type"];
        condition_value = params["condition_value"]
        if condition_type == "حاوی" and condition_value not in word: return False # تغییر "کلمه خاص" به "حاوی" و اصلاح شرط
        if condition_type == "شروع با" and not word.startswith(condition_value): return False
        return True

    def _perform_search(self, params):
        search_tokens = params["search_tokens"];
        search_phrase = " ".join(search_tokens);
        phrase_len = len(search_tokens);
        mode = params["mode"]
        before_counter, after_counter = Counter(), Counter();
        temp_sentence_mapping = defaultdict(list)
        has_filters = params["pos_filter"] != "هر نقشی" or params["condition_type"] != "فرقی نمی‌کند"

        # book_path_or_name : می‌تواند مسیر نسبی یا نام فایل قدیمی باشد
        for original_sentence, tagged_sentence, book_path_or_name in self.tagged_data:
            words = [item[0] for item in tagged_sentence]
            for i in range(len(words) - phrase_len + 1):
                if words[i:i + phrase_len] == search_tokens:
                    # book_path_or_name به عنوان سومین عنصر تاپل در sentence_mapping ذخیره می‌شود
                    if mode in ["هر دو", "کلمه قبلی"] and i > 0:
                        prev_word, prev_pos = tagged_sentence[i - 1]
                        if not has_filters or self._check_filters(prev_word, prev_pos, params):
                            before_counter[(prev_word, f"{prev_word} {search_phrase}")] += 1;
                            temp_sentence_mapping[("قبل", prev_word)].append((original_sentence, book_path_or_name))
                    if mode in ["هر دو", "کلمه بعدی"] and i + phrase_len < len(words):
                        next_word, next_pos = tagged_sentence[i + phrase_len]
                        if not has_filters or self._check_filters(next_word, next_pos, params):
                            after_counter[(next_word, f"{search_phrase} {next_word}")] += 1;
                            temp_sentence_mapping[("بعد", next_word)].append((original_sentence, book_path_or_name))
        self.sentence_mapping = temp_sentence_mapping;
        results = []
        if mode in ["هر دو", "کلمه قبلی"]:
            for (word, sample), count in before_counter.most_common(): results.append((sample, word, count, "قبل"))
        if mode in ["هر دو", "کلمه بعدی"]:
            for (word, sample), count in after_counter.most_common(): results.append((sample, word, count, "بعد"))
        self.root.after(0, self._update_ui_with_results, results)

    def _update_ui_with_results(self, results):
        self.results_tree.delete(*self.results_tree.get_children())
        if not results:
            self.results_tree.insert("", "end", values=("", "هیچ نتیجه‌ای یافت نشد.", "", ""));
            self.results_count_var.set("نتایج: 0")
        else:
            for item in results: self.results_tree.insert("", "end", values=item)
            self.results_count_var.set(f"نتایج: {len(results)}")
            self.root.after(100, lambda: self._sort_treeview('فراوانی', True))
        self._update_status("پردازش کامل شد. آماده برای جستجوی بعدی.");
        self.search_button.config(state=tk.NORMAL)

    def _highlight_text(self, text_widget, phrase, tag):
        start_index = "1.0"
        text_widget.tag_remove(tag, "1.0", tk.END) # ابتدا تگ‌های قبلی این نوع را حذف می‌کنیم

        if not phrase: # اگر عبارت جستجو خالی است، کاری انجام نده
            return

        while True:
            pos = text_widget.search(phrase, start_index, stopindex=tk.END, nocase=True, regexp=False)
            if not pos:
                break
            
            # بررسی مرزهای کلمه برای جلوگیری از هایلایت بخشی از کلمات دیگر
            is_start_boundary = False
            if pos == "1.0": # اگر در ابتدای متن است
                is_start_boundary = True
            else:
                prev_char_index = f"{pos}-1c"
                prev_char = text_widget.get(prev_char_index, pos)
                # حروف فارسی، انگلیسی، اعداد و نیم فاصله را به عنوان بخشی از کلمه در نظر نمی‌گیریم
                # یعنی اگر کاراکتر قبلی جزو اینها نباشد، مرز کلمه است.
                if not (prev_char.isalnum() or prev_char == '\u200c'): # \u200c is ZWNJ
                    is_start_boundary = True

            is_end_boundary = False
            end_search_index_for_next_char = f"{pos}+{len(phrase)}c"
            # بررسی اینکه آیا به انتهای متن رسیده‌ایم
            # مقایسه end_search_index_for_next_char با tk.END به طور مستقیم کار نمی‌کند.
            # به جای آن، سعی می‌کنیم کاراکتر بعدی را بخوانیم و اگر خطا داد یعنی به انتها رسیده‌ایم.
            try:
                # کاراکتر بعد از پایان عبارت یافت شده
                next_char = text_widget.get(end_search_index_for_next_char, f"{end_search_index_for_next_char}+1c")
                if not (next_char.isalnum() or next_char == '\u200c'):
                    is_end_boundary = True
            except tk.TclError: # به احتمال زیاد به انتهای متن رسیده‌ایم
                is_end_boundary = True
            
            if is_start_boundary and is_end_boundary:
                end_index = f"{pos}+{len(phrase)}c"
                text_widget.tag_add(tag, pos, end_index)
            
            # حرکت به بعد از عبارت یافت شده (یا بخشی از عبارت) برای جستجوی بعدی
            # این کار برای جلوگیری از لوپ بی‌نهایت در حالتی است که عبارت جستجو بخشی از خودش باشد (که اینجا رخ نمی‌دهد)
            # یا برای اطمینان از پیشرفت جستجو
            start_index = f"{pos}+{max(1, len(phrase))}c"


    def _sort_treeview(self, col, reverse):
        try:
            data = [(self.results_tree.set(item, col), item) for item in self.results_tree.get_children('')]
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
        from ctypes import windll; windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass
    root = tk.Tk()
    root.configure(bg='#333333')  # دقیقاً gray 20%
    style = ttk.Style(root)
    style.theme_use("clam")
    app = TextAnalyzerApp(root)
    root.mainloop()
