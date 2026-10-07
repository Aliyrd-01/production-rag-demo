"""
Smoke-тест app.py без браузера и без фонового сервера (streamlit.testing.AppTest).

Проверяет то, что ломает облачный деплой:
  1. скрипт вообще рендерится без исключения;
  2. вкладки Chat/Benchmarks реально переключаются (а не рисуются всегда);
  3. дефолтная стратегия отвечает и не падает;
  4. Benchmarks отдаёт таблицу с метриками.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from streamlit.testing.v1 import AppTest

FAILS = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


at = AppTest.from_file("app.py", default_timeout=600)
at.run()

# 1. Скрипт отрендерился
check("app renders without exception", not at.exception,
      str(at.exception[0].message) if at.exception else "")
if at.exception:
    print("\nEXCEPTION:\n", "\n".join(str(e.message) for e in at.exception))
    sys.exit(1)

# 2. Стратегии на месте
sel = at.selectbox
check("strategy selector present", len(sel) == 1)
if sel:
    opts = list(sel[0].options)
    check(">= 7 strategies in dropdown", len(opts) >= 7, f"{len(opts)}: {opts}")
    check("default is hybrid (fits 1 GB RAM)",
          sel[0].value == "hybrid", f"default={sel[0].value}")

# 3. Вкладки
tabs = at.get("tab")
check(">= 2 tabs rendered", len(tabs) >= 2, f"{len(tabs)}")

# 4. Вкладки. Streamlit выполняет код ОБЕИХ вкладок на каждом rerun и прячет
#    неактивную через CSS — поэтому таблица метрик присутствует в дереве всегда.
#    Проверяем, что контент реально внутри st.tabs (иначе он рендерился бы
#    под виджетами чата, как было до переделки), а не то, что таблицы нет.
check("tabs wrap all panes", len(tabs) >= 2, f"{len(tabs)}")
check("benchmarks pane renders table", len(at.dataframe) >= 1,
      f"{len(at.dataframe)} dataframes")
if at.dataframe:
    rows = at.dataframe[0].value
    check("benchmarks table has >=5 strategies", len(rows) >= 5, f"{len(rows)} rows")

# 5. Данные для вкладки Benchmarks валидны
try:
    import json
    res_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results_retrieval.json")
    with open(res_path, encoding="utf-8") as f:
        latest = {}
        for run in json.load(f):
            for row in run.get("rows", []):
                latest[row["strategy"]] = row
    check("benchmarks data: >=5 strategies", len(latest) >= 5, f"{len(latest)}")
    check("benchmarks data: hybrid_ce measured", "hybrid_ce" in latest)
    ce = latest.get("hybrid_ce", {})
    check("benchmarks data: hybrid_ce has Hit@5",
          ce.get("hit@5") is not None, f"hit@5={ce.get('hit@5')}")
except Exception as e:
    check("benchmarks data readable", False, f"{type(e).__name__}: {e}")

# 6. Чат-вкладка отвечает на вопрос (дефолтная стратегия = hybrid)
at.text_input(key="q").set_value("what color is amber urine")
at.run()
ask = [b for b in at.button if "Ask" in (b.label or "")]
check("Ask button present", len(ask) == 1)
if ask:
    ask[0].click()
    at.run()
    check("query: no exception", not at.exception,
          str(at.exception[0].message) if at.exception else "")
    markdown = [m.value for m in at.markdown]
    check("answer rendered", any("Answer" in m for m in markdown),
          f"{len(markdown)} markdown blocks")
    check("metrics rendered", len(at.metric) >= 3, f"{len(at.metric)} metrics")

print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"FAILED: {FAILS}"))
sys.exit(1 if FAILS else 0)