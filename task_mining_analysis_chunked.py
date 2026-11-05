# -*- coding: utf-8 -*-
"""Task Mining -> LLM -> HTML. Запуск в Jupyter: %run task_mining_analysis.py

Для использования уже заданного PROCESS_ANALYSIS_PROMPT из ноутбука:
    exec(Path('task_mining_analysis.py').read_text(encoding='utf-8'))
или скопируйте текст промпта в файл process_analysis_prompt.txt.
"""
from pathlib import Path
from datetime import datetime
import html
import json
import re
import time
import traceback

import pandas as pd
import httpx
from openai import OpenAI, RateLimitError, APIStatusError, APIConnectionError, APITimeoutError
from IPython.display import display, HTML
from tqdm.auto import tqdm

INPUT_FOLDER = Path('./input_tm_esc')
OUTPUT_FOLDER = Path('./output_tm_esc')
MODEL_NAME = 'CodegenchatReasoning/codegen'
BASE_URL = 'https://ai-gateway.raisa.go.rshbank.ru/v1'
API_KEY = ''  # Укажите ключ только в защищённом окружении, не сохраняйте его в общем ноутбуке
VERIFY_SSL = False  # Временно для внутреннего контура; предпочтительно указать корпоративный CA
MAX_CHARS_PER_CHUNK = 18000  # При HTTP 400 уменьшите до 10000
MAX_OUTPUT_TOKENS = 3500
REQUEST_TIMEOUT = 240
REQUESTS_PER_MINUTE = 4  # Запросы/мин; не гарантирует соблюдение TPM
RETRY_COUNT = 2

INPUT_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Если PROCESS_ANALYSIS_PROMPT определён в ноутбуке, он используется автоматически.
# При отдельном запуске загрузите промпт из текстового файла.
PROMPT_FILE = Path('process_analysis_prompt.txt')
PROMPT = globals().get('PROCESS_ANALYSIS_PROMPT') or (PROMPT_FILE.read_text(encoding='utf-8') if PROMPT_FILE.exists() else '')
if not PROMPT.strip():
    raise ValueError('Не найден промпт: определите PROCESS_ANALYSIS_PROMPT в ноутбуке или создайте process_analysis_prompt.txt')
if not API_KEY:
    import os
    API_KEY = os.environ.get('RAISA_API_KEY', '')
if not API_KEY:
    raise ValueError('Не задан API-ключ: укажите RAISA_API_KEY в переменных окружения')

client = OpenAI(base_url=BASE_URL, api_key=API_KEY,
                http_client=httpx.Client(verify=VERIFY_SSL, timeout=REQUEST_TIMEOUT),
                max_retries=0)


def read_table(path):
    suffix = path.suffix.lower()
    if suffix in ('.xlsx', '.xlsm'):
        return pd.read_excel(path, dtype=str).fillna('')
    if suffix == '.csv':
        try:
            return pd.read_csv(path, sep=None, engine='python', dtype=str).fillna('')
        except UnicodeDecodeError:
            return pd.read_csv(path, sep=None, engine='python', encoding='cp1251', dtype=str).fillna('')
    raise ValueError(f'Неподдерживаемый формат: {suffix}')


def dataframe_chunks(df, max_chars):
    # Не разбиваем одну строку между частями; длинную строку отправляем отдельно.
    headers = list(df.columns)
    chunk, length = [], 0
    for row in df.itertuples(index=False, name=None):
        record = dict(zip(headers, map(str, row)))
        line = json.dumps(record, ensure_ascii=False, default=str)
        if chunk and length + len(line) > max_chars:
            yield chunk
            chunk, length = [], 0
        chunk.append(record)
        length += len(line)
    if chunk:
        yield chunk


last_request_at = 0.0

def ask_llm(messages, label):
    global last_request_at
    for attempt in range(RETRY_COUNT + 1):
        wait = max(0, 60 / REQUESTS_PER_MINUTE - (time.monotonic() - last_request_at))
        if wait:
            time.sleep(wait)
        last_request_at = time.monotonic()
        try:
            result = client.chat.completions.create(
                model=MODEL_NAME, messages=messages,
                max_tokens=MAX_OUTPUT_TOKENS, temperature=0,
            )
            choice = result.choices[0]
            usage = result.usage
            if usage:
                print(f'  {label}: токены input={usage.prompt_tokens}, output={usage.completion_tokens}, всего={usage.total_tokens}', flush=True)
            if choice.finish_reason == 'length':
                raise RuntimeError(f'{label}: ОТВЕТ ОБРЕЗАН (лимит выходных токенов). Уменьшите размер части или увеличьте MAX_OUTPUT_TOKENS.')
            if not choice.message.content:
                raise RuntimeError(f'{label}: модель вернула пустой ответ (возможен расход бюджета токенов на reasoning).')
            return choice.message.content
        except RateLimitError as exc:
            print(f'❌ {label}: HTTP 429 — превышен лимит запросов/токенов. {exc}', flush=True)
            if attempt == RETRY_COUNT:
                raise
            time.sleep(30 * (attempt + 1))
        except APIStatusError as exc:
            print(f'❌ {label}: HTTP {exc.status_code}; ответ сервера: {str(exc)[:1200]}', flush=True)
            if exc.status_code in (500, 502, 503, 504) and attempt < RETRY_COUNT:
                time.sleep(10 * (attempt + 1))
                continue
            raise
        except (APIConnectionError, APITimeoutError) as exc:
            print(f'❌ {label}: ошибка соединения/таймаут: {exc}', flush=True)
            if attempt == RETRY_COUNT:
                raise
            time.sleep(10 * (attempt + 1))


def to_html(markdown_text):
    # markdown + экранирование HTML: безопаснее при анализе внешних логов.
    try:
        import markdown as md
        return md.markdown(markdown_text, extensions=['tables', 'fenced_code', 'sane_lists'], output_format='html5')
    except ImportError:
        escaped = html.escape(markdown_text)
        return '<pre style="white-space:pre-wrap">' + escaped + '</pre>'


def save_html(path, title, body, meta):
    doc = '''<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>__TITLE__</title>
<style>body{font:16px/1.65 system-ui,Arial;background:#f3f6f8;color:#23313c;margin:0}
main{max-width:1100px;margin:36px auto;padding:38px 48px;background:white;border-radius:16px;box-shadow:0 5px 25px #182a3712}
h1,h2,h3{color:#17634a;line-height:1.3}h1{border-bottom:3px solid #36a37a;padding-bottom:14px}
table{border-collapse:collapse;width:100%;display:block;overflow-x:auto;margin:20px 0}
th,td{border:1px solid #dce6e3;padding:10px 13px;text-align:left;vertical-align:top;min-width:110px}
th{background:#e8f4ee}tr:nth-child(even){background:#f8fbf9}
pre{overflow-x:auto;background:#f3f6f8;padding:16px;border-radius:8px}code{overflow-wrap:anywhere}
.meta{color:#687b77;font-size:14px;margin-bottom:24px} @media(max-width:650px){main{margin:0;padding:20px}}</style>
</head><body><main><h1>__TITLE__</h1><div class="meta">__META__</div>__BODY__</main></body></html>'''
    doc = doc.replace('__TITLE__', html.escape(title)).replace('__META__', html.escape(meta)).replace('__BODY__', body)
    path.write_text(doc, encoding='utf-8')


def analyze_one(activity_path, cfg_path):
    activity = read_table(activity_path)
    cfg = read_table(cfg_path)
    if activity.empty:
        raise ValueError('Файл activity пуст')
    if cfg.empty:
        raise ValueError('Файл cfg_op_short пуст')
    # Не перемешиваем сотрудников и рабочие станции: порядок строк сохраняется внутри групп.
    group_cols = [c for c in ('worker_id', 'computer_name') if c in activity.columns]
    chunks = []
    groups = activity.groupby(group_cols, sort=False, dropna=False) if group_cols else [(None, activity)]
    for _, group in groups:
        chunks.extend(dataframe_chunks(group, MAX_CHARS_PER_CHUNK))
    print(f'{activity_path.name}: {len(activity)} строк, {len(chunks)} частей; '
          f'предел {MAX_CHARS_PER_CHUNK} символов данных/часть', flush=True)
    cfg_json = cfg.to_json(orient='records', force_ascii=False)
    if len(cfg_json) > 100000:
        raise ValueError('Справочник cfg_op_short слишком велик; необходимо уменьшить его перед отправкой')
    reports = []
    for i, rows in enumerate(tqdm(chunks, desc=f'Части {activity_path.name}', unit='часть'), 1):
        messages = [
            {'role': 'system', 'content': PROMPT + '\nОтвечай на русском языке в Markdown. Не придумывай факты, длительности и экономический эффект. Отмечай недостаток данных.'},
            {'role': 'user', 'content': (
                f'Часть {i}/{len(chunks)}. Проанализируй данные Task Mining. '
                'Это только фрагмент общего журнала: не делай окончательных выводов о целом рабочем дне. '
                'Используй справочник cfg_op_short для сопоставления действий. '
                'Поле ms_time содержит длительность в миллисекундах; не считай его временем начала действия. '
                'Не суммируй пересекающиеся интервалы без проверки. Не делай выводов о полной операции, если она пересекает границу частей.\n\n'
                f'cfg_op_short:\n{cfg_json}\n\nactivity:\n{json.dumps(rows, ensure_ascii=False)}'
            )}
        ]
        print(f'  Отправка части {i}/{len(chunks)}: {len(rows)} строк, '
              f'{len(messages[1]["content"]):,} символов запроса', flush=True)
        report = ask_llm(messages, f'{activity_path.name}: часть {i}/{len(chunks)}')
        reports.append(report)
        # Промежуточные результаты сохраняются даже при ошибке на следующей части.
        (OUTPUT_FOLDER / f'{activity_path.stem}_part_{i:03d}.md').write_text(report, encoding='utf-8')

    # Консолидация нескольких частей; для большого числа частей сводим поэтапно.
    round_no = 0
    while len(reports) > 1:
        round_no += 1
        grouped = []
        group = []
        size = 0
        for report in reports:
            if group and size + len(report) > MAX_CHARS_PER_CHUNK:
                grouped.append(group)
                group, size = [], 0
            group.append(report)
            size += len(report)
        if group:
            grouped.append(group)
        # Гарантия прогресса при единичных очень длинных отчётах
        if all(len(g) == 1 for g in grouped):
            grouped = [reports[i:i+2] for i in range(0, len(reports), 2)]
        new_reports = []
        for j, group in enumerate(tqdm(grouped, desc='Объединение выводов', unit='группа'), 1):
            if len(group) == 1:
                new_reports.append(group[0])
                continue
            combined = ask_llm([
                {'role': 'system', 'content': PROMPT + '\nПодготовь единый связный отчет в Markdown на русском. Удали повторы. Не суммируй пересекающиеся показатели. Разделяй факты и гипотезы.'},
                {'role': 'user', 'content': 'Объедини частичные отчёты, сохрани примеры и обоснования:\n\n' + '\n\n---\n\n'.join(group)}
            ], f'Объединение {j}/{len(grouped)}')
            new_reports.append(combined)
        if len(new_reports) >= len(reports):
            raise RuntimeError('Консолидация не уменьшила число отчётов — остановлено во избежание цикла')
        reports = new_reports
        for k, report in enumerate(reports, 1):
            (OUTPUT_FOLDER / f'{activity_path.stem}_merge_{round_no:02d}_{k:03d}.md').write_text(report, encoding='utf-8')

    final = reports[0]
    (OUTPUT_FOLDER / f'{activity_path.stem}_final.md').write_text(final, encoding='utf-8')
    dest = OUTPUT_FOLDER / f'{activity_path.stem}_analysis.html'
    save_html(dest, f'Анализ рабочего дня — {activity_path.stem}', to_html(final),
              f'Источник: {activity_path.name} · строк: {len(activity)} · дата формирования: {datetime.now():%d.%m.%Y %H:%M}')
    return dest


activity_files = sorted(p for p in INPUT_FOLDER.iterdir()
                        if p.suffix.lower() in ('.xlsx', '.xlsm', '.csv')
                        and not p.name.startswith('~$')
                        and 'cfg_op_short' not in p.stem.lower())
cfg_files = sorted(p for p in INPUT_FOLDER.iterdir()
                   if 'cfg_op_short' in p.stem.lower() and p.suffix.lower() in ('.xlsx', '.xlsm', '.csv'))
if not cfg_files:
    raise FileNotFoundError('Положите файл cfg_op_short.xlsx или cfg_op_short.csv в папку input_tm_esc')
if not activity_files:
    raise FileNotFoundError('Не найдены файлы activity (.xlsx/.csv) в папке input_tm_esc')

print(f'Найдено файлов activity: {len(activity_files)}; справочник: {cfg_files[0].name}', flush=True)
results, failures = [], []
for path in tqdm(activity_files, desc='Файлы', unit='файл'):
    try:
        out = analyze_one(path, cfg_files[0])
        results.append(out)
        print(f'✅ Готово: {out}', flush=True)
    except Exception as exc:
        print(f'❌ ОШИБКА при обработке {path.name}: {type(exc).__name__}: {exc}', flush=True)
        failures.append((path.name, f'{type(exc).__name__}: {exc}'))
        (OUTPUT_FOLDER / f'{path.stem}_error.txt').write_text(traceback.format_exc(), encoding='utf-8')

print(f'ИТОГО: успешно {len(results)}, ошибок {len(failures)}. Папка: {OUTPUT_FOLDER.resolve()}', flush=True)
for p in results:
    display(HTML(f'<p><b>HTML-отчёт:</b> {html.escape(str(p))}</p>'))
