#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py — 詞彙庫 雲端版 後端 API
把 vocab.py 裡呼叫 DeepSeek / Gemini 的邏輯包成一個小型 Flask API，
讓網頁前端（input.html）可以在瀏覽器按一下「查詢」就拿到結構化詞條，
而 DEEPSEEK_API_KEY / GEMINI_API_KEY 這些機密金鑰永遠留在伺服器端，
不會出現在瀏覽器或前端原始碼裡。

Supabase 的新增/查重複/刪除仍然交給前端用 anon key 直接呼叫
（跟現有 index.html 一樣），因為 anon key 本來就設計成可公開、
搭配 Supabase 的 RLS（Row Level Security）規則來控制權限。

部署：Render Web Service
  Build Command : pip install -r requirements.txt
  Start Command : gunicorn app:app
  環境變數      : DEEPSEEK_API_KEY, GEMINI_API_KEY, AI_MODEL（可選）
"""

import json
import logging
import os
import urllib.error
import urllib.request

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("vocab-web")

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)  # 如果前端日後改放到別的網域（例如 GitHub Pages），需要這個

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_MODEL = os.getenv("AI_MODEL", "deepseek").lower()
MAX_RETRIES = 2

FIELDS = [
    "word_or_phrase", "translation", "meaning", "part_of_speech",
    "pronunciation_ipa", "derived_or_related", "relationship_menu",
    "cross_reference", "entry_type", "pronunciation_link",
    "register_and_variety", "collocations_and_expressions",
    "source", "additional_notes", "ai_agent",
]


# ── 系統提示詞（與 vocab.py 共用同一份規格）─────────────────
def build_system_prompt(ai_name: str) -> str:
    return f"""你是專業詞典編輯 Ayu，精通英語、廣東話與繁體中文。
使用者給你一個單字或片語，你必須回傳一個合法 JSON 物件（不加任何 Markdown 或說明文字），包含以下 15 個欄位：

{{
  "word_or_phrase": "待查詞（原文）",
  "translation": "最精準的單一翻譯（英→繁中 / 中或粵→英）",
  "meaning": "【解1】: (詞性) 定義\\n• \\"例句1.\\" (翻譯)\\n\\n• \\"例句2.\\" (翻譯)\\n\\n【解2】: ...",
  "part_of_speech": "詞性中英並列（如：名詞 noun）",
  "pronunciation_ipa": "/IPA 音標/",
  "derived_or_related": "衍生或相關字彙（無則空字串）",
  "relationship_menu": "",
  "cross_reference": "",
  "entry_type": "General",
  "pronunciation_link": "https://dictionary.cambridge.org/dictionary/english/WORD",
  "register_and_variety": "語域（如：formal / informal / neutral）",
  "collocations_and_expressions": "• 搭配詞1\\n\\n• 搭配詞2",
  "source": "Cambridge Dictionary, Oxford Learner's Dictionaries",
  "additional_notes": "【近義辨析】\\n• ...\\n\\n【語源】\\n• ...\\n\\n【使用觀察】\\n• ...",
  "ai_agent": "{ai_name}"
}}

品質要求：
- meaning 最多 4 個義項（解1–解4），每義項至少 2 個道地例句
- additional_notes 必須包含【近義辨析】【語源】【使用觀察】三個段落
- 語言偵測：英文→繁中翻譯；中文或廣東話→英文翻譯
- source 欄位禁止填入 AI 模型名稱
- 回傳內容只能是合法 JSON，不得有任何多餘文字"""


def extract_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rstrip("`").strip()
    return json.loads(raw)


def call_deepseek(word: str) -> dict:
    system_prompt = build_system_prompt("DeepSeek")
    url = "https://api.deepseek.com/v1/chat/completions"
    payload = json.dumps({
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"請查詢這個單字：{word}"},
        ],
        "temperature": 0.3,
        "max_tokens": 3000,
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    raw = body["choices"][0]["message"]["content"]
    return extract_json(raw)


def call_gemini(word: str) -> dict:
    system_prompt = build_system_prompt("Gemini")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    payload = json.dumps({
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"parts": [{"text": f"請查詢這個單字：{word}"}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 3000},
    }).encode("utf-8")

    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        body = json.loads(resp.read().decode("utf-8"))

    raw = body["candidates"][0]["content"]["parts"][0]["text"]
    return extract_json(raw)


def call_ai(word: str, model: str) -> dict:
    return call_gemini(word) if model == "gemini" else call_deepseek(word)


def validate_entry(entry: dict, word: str) -> list:
    errors = []
    missing = [f for f in FIELDS if f not in entry]
    if missing:
        errors.append(f"缺少欄位：{', '.join(missing)}")
    wop = entry.get("word_or_phrase", "").strip()
    if not wop:
        errors.append("word_or_phrase 為空")
    elif word.lower() not in wop.lower():
        errors.append(f"word_or_phrase='{wop}' 不含查詢詞 '{word}'")
    if not entry.get("translation", "").strip():
        errors.append("translation 為空")
    meaning = entry.get("meaning", "").strip()
    if len(meaning) < 20:
        errors.append(f"meaning 過短（{len(meaning)} 字元），可能不完整")
    if not entry.get("ai_agent", "").strip():
        errors.append("ai_agent 為空")
    return errors


# ── 路由 ─────────────────────────────────────────────────

@app.get("/")
def serve_input_page():
    return send_from_directory(app.static_folder, "input.html")


@app.get("/view")
def serve_view_page():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/health")
def health():
    return jsonify({
        "ok": True,
        "deepseek_configured": bool(DEEPSEEK_API_KEY),
        "gemini_configured": bool(GEMINI_API_KEY),
        "default_model": DEFAULT_MODEL,
    })


@app.post("/api/lookup")
def lookup():
    data = request.get_json(silent=True) or {}
    word = (data.get("word") or "").strip()
    model = (data.get("model") or DEFAULT_MODEL).lower()
    if not word:
        return jsonify({"ok": False, "error": "word 不可為空"}), 400
    if model not in ("deepseek", "gemini"):
        model = DEFAULT_MODEL
    if model == "deepseek" and not DEEPSEEK_API_KEY:
        return jsonify({"ok": False, "error": "伺服器未設定 DEEPSEEK_API_KEY"}), 500
    if model == "gemini" and not GEMINI_API_KEY:
        return jsonify({"ok": False, "error": "伺服器未設定 GEMINI_API_KEY"}), 500

    last_errors = []
    entry = None
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            entry = call_ai(word, model)
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8")
            except Exception:
                pass
            log.warning("HTTP error on attempt %s for '%s': %s %s", attempt, word, e.code, body[:200])
            last_errors = [f"HTTP {e.code}：{e.reason}"]
            entry = None
            continue
        except Exception as e:
            log.warning("Call failed on attempt %s for '%s': %s", attempt, word, e)
            last_errors = [str(e)]
            entry = None
            continue

        errs = validate_entry(entry, word)
        if not errs:
            return jsonify({"ok": True, "entry": entry, "attempts": attempt, "warnings": []})
        last_errors = errs

    if entry is not None:
        # 已達最大重試次數，回傳最後一次結果並標註警告，交給使用者人工確認
        return jsonify({"ok": True, "entry": entry, "attempts": MAX_RETRIES + 1, "warnings": last_errors})

    return jsonify({"ok": False, "error": "；".join(last_errors) or "查詢失敗"}), 502


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
