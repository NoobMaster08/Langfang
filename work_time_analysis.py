# -*- coding: utf-8 -*-
# Анализ рабочего времени: Excel/CSV -> чанки -> LLM reasoning -> единый итог.

from __future__ import annotations
from pathlib import Path
from datetime import date, datetime
from typing import Any
import json, math, time
import numpy as np
import pandas as pd
import httpx
from openai import OpenAI, RateLimitError, APIConnectionError, APIStatusError, APITimeoutError
from IPython.display import display, Markdown

# ==================== CONFIG ====================
INPUT_FOLDER = Path('./input')
OUTPUT_FOLDER = Path('./output')
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

MODEL_NAME = 'qwen'
TEMPERATURE = 0.2
ENABLE_THINKING = True
MAX_CHUNK_CHARS = 20_000
OVERLAP_ROWS = 20
MAX_CONSOLIDATION_CHARS = 35_000
CHUNK_MAX_TOKENS = 4_000
CONSOLIDATION_MAX_TOKENS = 5_000
FINAL_MAX_TOKENS = 7_000
MAX_RETRIES = 10
REQUEST_DELAY_SECONDS = 15
RATE_LIMIT_WAIT_SECONDS = 65
SHOW_PROGRESS = False

CHUNK_PROMPT = '''
Ты — опытный аналитик данных, специализирующийся на анализе журналов активности пользователей в корпоративных информационных системах.
Перед тобой часть хронологического лога действий сотрудника за рабочий день.
Определи, какой работой занимался сотрудник в данном временном интервале.
Рассматривай записи как единую последовательность, группируй связанные действия в рабочие задачи, учитывай переходы между системами и вероятную цель действий.
Не перечисляй технические события построчно и не придумывай отсутствующие факты.
При неоднозначности используй формулировки «вероятно», «предположительно», «по всей видимости».
Верни компактный промежуточный аналитический материал с сохранением существенной хронологии.
'''

FINAL_PROMPT = '''
На основании анализа полного лога действий определи, чем занимался сотрудник в течение рабочего дня.
Результат — краткое связное описание объемом 3–7 предложений, отвечающее на вопрос «Чем занимался сотрудник в течение рабочего дня?».
Объединяй связанные действия в крупные рабочие задачи. Если видов деятельности несколько, опиши их в хронологическом порядке.
Не перечисляй отдельные технические события, окна и вкладки, если это не нужно для понимания работы. Не придумывай факты.
При неоднозначности используй «вероятно», «предположительно», «по всей видимости».
Стиль: деловой, лаконичный, связный текст без списков, акцент на рабочих процессах и задачах.
'''

llm_client = OpenAI(
    base_url='ВСТАВЬ_РАБОЧИЙ_BASE_URL',
    api_key='ВСТАВЬ_LITELLM_TOKEN',
    http_client=httpx.Client(verify=False),
    timeout=600.0,
)

def log(msg: str):
    if SHOW_PROGRESS: print(msg)

def normalize_value(value: Any) -> Any:
    if value is None: return None
    try:
        if pd.isna(value): return None
    except (TypeError, ValueError): pass
    if isinstance(value, (pd.Timestamp, datetime, date)): return value.isoformat()
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, np.bool_): return bool(value)
    if isinstance(value, (str, int, float, bool)): return value
    return str(value)

def find_single_file(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in {'.xlsx','.xls','.csv'} and not f.name.startswith('~$')]
    if not files: raise FileNotFoundError(f'В папке {folder.resolve()} не найден Excel или CSV.')
    if len(files) > 1: raise RuntimeError('В папке input должен находиться ровно один файл.')
    return files[0]

def load_log(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {'.xlsx','.xls'}: df = pd.read_excel(path)
    else:
        try: df = pd.read_csv(path)
        except UnicodeDecodeError: df = pd.read_csv(path, encoding='cp1251')
    df = df.dropna(axis=0, how='all').dropna(axis=1, how='all')
    df.columns = [str(c).strip() for c in df.columns]
    df = df.reset_index(drop=True)
    if df.empty: raise RuntimeError('Входной файл не содержит данных.')
    return df

def dataframe_rows_to_json(df: pd.DataFrame) -> list[str]:
    out=[]
    for i,row in df.iterrows():
        data={str(c): normalize_value(v) for c,v in row.items()}
        out.append(json.dumps({'row_number':i+1, **data}, ensure_ascii=False, separators=(',',':')))
    return out

def split_log(rows: list[str], max_chars=MAX_CHUNK_CHARS, overlap_rows=OVERLAP_ROWS) -> list[str]:
    chunks=[]; start=0
    while start < len(rows):
        cur=[]; size=0; end=start
        while end < len(rows):
            s=len(rows[end])+1
            if cur and size+s > max_chars: break
            cur.append(rows[end]); size+=s; end+=1
        if not cur: cur=[rows[start][:max_chars]]; end=start+1
        chunks.append('\n'.join(cur))
        if end >= len(rows): break
        start=max(end-overlap_rows, start+1)
    return chunks

def ask_model(system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    current=max_tokens; empty=0
    for attempt in range(1, MAX_RETRIES+1):
        try:
            r=llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}],
                temperature=TEMPERATURE,
                max_tokens=current,
                extra_body={'chat_template_kwargs':{'enable_thinking':ENABLE_THINKING}},
            )
            content=(r.choices[0].message.content or '').strip()
            if content: return content
            empty += 1
            if empty >= 3: raise RuntimeError('Модель выполнила reasoning, но не сформировала content.')
            current=min(current+2000,12000); time.sleep(10)
        except RateLimitError:
            if attempt>=MAX_RETRIES: raise
            time.sleep(RATE_LIMIT_WAIT_SECONDS)
        except (APIConnectionError, APITimeoutError):
            if attempt>=MAX_RETRIES: raise
            time.sleep(min(10*attempt,60))
        except APIStatusError as e:
            if e.status_code>=500 and attempt<MAX_RETRIES:
                time.sleep(min(10*attempt,60)); continue
            raise
    raise RuntimeError('Не удалось получить ответ модели.')

def analyze_chunk(chunk: str, n: int, total: int) -> str:
    return ask_model(CHUNK_PROMPT, f'''Перед тобой фрагмент {n} из {total} единого хронологического лога рабочего дня.
Не формируй окончательный вывод по всему дню. Верни компактный материал для последующего объединения.\n\nЛОГ:\n{chunk}''', CHUNK_MAX_TOKENS)

def split_text(text: str, max_chars: int) -> list[str]:
    text=text.strip()
    if len(text)<=max_chars: return [text]
    blocks=[]; start=0
    while start<len(text):
        end=min(start+max_chars,len(text))
        if end<len(text):
            p=text.rfind('\n\n',start,end)
            if p>start: end=p
        blocks.append(text[start:end].strip()); start=end
    return blocks

def consolidate_results(results: list[str]) -> str:
    current='\n\n'.join(f'===== ФРАГМЕНТ {i} =====\n{x}' for i,x in enumerate(results,1))
    while len(current)>MAX_CONSOLIDATION_CHARS:
        new=[]
        for block in split_text(current,MAX_CONSOLIDATION_CHARS):
            result=ask_model(
                'Ты консолидируешь результаты анализа последовательных фрагментов одного рабочего дня. Верни только компактный промежуточный материал, не окончательный ответ.',
                f'''Объедини материалы. Сохрани хронологию, объедини связанные задачи, удали дубли от пересечения фрагментов, не придумывай факты и сохрани неопределенность там, где она есть.\n\nМАТЕРИАЛЫ:\n{block}''',
                CONSOLIDATION_MAX_TOKENS,
            )
            new.append(result); time.sleep(REQUEST_DELAY_SECONDS)
        current='\n\n'.join(new)
    return current

def create_final_answer(material: str) -> str:
    return ask_model(
        'Ты выполняешь итоговый анализ рабочего дня по журналу активности. Используй reasoning, но верни только окончательный результат. Не упоминай чанки, консолидацию или технические этапы анализа.',
        f'''ЗАДАЧА И ФОРМАТ:\n{FINAL_PROMPT}\n\nСформируй один окончательный ответ по всему рабочему дню.\n\nМАТЕРИАЛ:\n{material}''',
        FINAL_MAX_TOKENS,
    )

# ==================== ЗАПУСК ====================
input_file=find_single_file(INPUT_FOLDER)
df=load_log(input_file)
rows=dataframe_rows_to_json(df)
chunks=split_log(rows)
log(f'Файл: {input_file.name}; строк: {len(df):,}; частей: {len(chunks)}')

intermediate=[]
for i,chunk in enumerate(chunks,1):
    intermediate.append(analyze_chunk(chunk,i,len(chunks)))
    if i<len(chunks): time.sleep(REQUEST_DELAY_SECONDS)

material=consolidate_results(intermediate)
final_result=create_final_answer(material)
(OUTPUT_FOLDER/'result.md').write_text(final_result,encoding='utf-8')
display(Markdown(final_result))
