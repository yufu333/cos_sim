import math

import numpy as np
import pandas as pd
from js import document
from pyodide.ffi import create_proxy


# =========================================================
# 基本データ
# =========================================================

TARGET_USER = "Aさん"
RECOMMEND_MOVIE = "スター・ウォーズ"

BASE_DATA = {
    "タイタニック":       [5, 4, 1, 5, 3, 2, 3],
    "シックスセンス":     [2, 5, 2, 4, 4, 5, 1],
    "ジュラシックパーク": [3, 2, 5, 3, 2, 3, 1],
    "スター・ウォーズ":   [np.nan, 5, 2, 4, 5, 3, 4],
}

PEOPLE = [
    "Aさん",
    "Bさん",
    "Cさん",
    "Dさん",
    "Eさん",
    "Fさん",
    "Gさん",
]


# =========================================================
# DOM操作用の補助関数
# =========================================================

def set_html(element_id: str, content: str) -> None:
    document.getElementById(element_id).innerHTML = content


def set_text(element_id: str, content: str) -> None:
    document.getElementById(element_id).textContent = content


def get_rating(select_id: str) -> int:
    value = document.getElementById(select_id).value
    return int(value)


# =========================================================
# DataFrameを作る
# =========================================================

def make_dataframe() -> pd.DataFrame:
    df = pd.DataFrame(BASE_DATA, index=PEOPLE)

    # Aさんの3作品だけ、画面のプルダウン値で上書き
    df.loc[TARGET_USER, "タイタニック"] = get_rating(
        "rating-titanic"
    )
    df.loc[TARGET_USER, "シックスセンス"] = get_rating(
        "rating-sixth"
    )
    df.loc[TARGET_USER, "ジュラシックパーク"] = get_rating(
        "rating-jurassic"
    )

    # スター・ウォーズは未視聴のまま
    df.loc[TARGET_USER, RECOMMEND_MOVIE] = np.nan

    return df


# =========================================================
# コサイン類似度
# =========================================================

def cosine_similarity(
    df: pd.DataFrame,
    person1: str,
    person2: str,
) -> float:
    a = df.loc[person1].to_numpy(dtype=float)
    b = df.loc[person2].to_numpy(dtype=float)

    # 両者とも評価している映画だけを使用
    common = ~np.isnan(a) & ~np.isnan(b)

    if not np.any(common):
        return np.nan

    a_common = a[common]
    b_common = b[common]

    denominator = (
        np.linalg.norm(a_common)
        * np.linalg.norm(b_common)
    )

    if denominator == 0:
        return np.nan

    return float(
        np.dot(a_common, b_common) / denominator
    )


def make_similarity_table(df: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(
        index=PEOPLE,
        columns=PEOPLE,
        dtype=float,
    )

    for person1 in PEOPLE:
        for person2 in PEOPLE:
            result.loc[person1, person2] = (
                cosine_similarity(df, person1, person2)
            )

    return result


# =========================================================
# HTML作成用
# =========================================================

def rating_cell(value: float) -> str:
    if pd.isna(value):
        return '<td class="unseen">未視聴</td>'

    return f"<td>{int(value)}</td>"


def similarity_class(value: float) -> str:
    if value >= 0.95:
        return "sim sim-very-high"

    if value >= 0.85:
        return "sim sim-high"

    if value >= 0.70:
        return "sim sim-mid"

    return "sim sim-low"


def make_rating_html(df: pd.DataFrame) -> str:
    movie_headers = "".join(
        f"<th>{movie}</th>"
        for movie in df.columns
    )

    rows = []

    for person in PEOPLE:
        cells = "".join(
            rating_cell(float(value))
            for value in df.loc[person].to_numpy()
        )

        row_class = (
            ' class="target-row"'
            if person == TARGET_USER
            else ""
        )

        rows.append(
            f"<tr{row_class}>"
            f"<th>{person}</th>"
            f"{cells}"
            f"</tr>"
        )

    return f'''
    <table class="data-table rating-table">
      <thead>
        <tr>
          <th>ユーザー</th>
          {movie_headers}
        </tr>
      </thead>
      <tbody>
        {"".join(rows)}
      </tbody>
    </table>
    '''


def make_similarity_html(
    df_cos: pd.DataFrame,
    most_similar: str,
) -> str:
    headers = "".join(
        f"<th>{person}</th>"
        for person in PEOPLE
    )

    rows = []

    for i, person1 in enumerate(PEOPLE):
        cells = []

        for j, person2 in enumerate(PEOPLE):
            value = float(
                df_cos.loc[person1, person2]
            )

            # 下三角部分は重複なので「−」
            if i > j:
                cells.append(
                    '<td class="duplicate">−</td>'
                )
                continue

            # 対角線
            if i == j:
                cells.append(
                    '<td class="self-sim">1.000</td>'
                )
                continue

            if math.isnan(value):
                cells.append(
                    '<td class="no-data">—</td>'
                )
                continue

            css_class = similarity_class(value)

            # Aさんと最も似ている人のセルを強調
            if (
                person1 == TARGET_USER
                and person2 == most_similar
            ):
                css_class += " best-match"

            cells.append(
                f'<td class="{css_class}">'
                f"{value:.3f}"
                f"</td>"
            )

        row_class = (
            ' class="target-row"'
            if person1 == TARGET_USER
            else ""
        )

        rows.append(
            f"<tr{row_class}>"
            f"<th>{person1}</th>"
            f'{"".join(cells)}'
            f"</tr>"
        )

    return f'''
    <table class="data-table similarity-table">
      <thead>
        <tr>
          <th>ユーザー</th>
          {headers}
        </tr>
      </thead>
      <tbody>
        {"".join(rows)}
      </tbody>
    </table>
    '''


def stars(score: float) -> str:
    rounded = max(
        0,
        min(5, int(round(score)))
    )

    return (
        "★" * rounded
        + "☆" * (5 - rounded)
    )


# =========================================================
# 再計算
# =========================================================

def recalculate(event=None) -> None:
    df = make_dataframe()
    df_cos = make_similarity_table(df)

    target_index = PEOPLE.index(TARGET_USER)

    similarity_values = (
        df_cos.iloc[target_index]
        .to_numpy(dtype=float, copy=True)
    )

    # 自分自身との類似度は除外
    similarity_values[target_index] = np.nan

    most_similar_index = int(
        np.nanargmax(similarity_values)
    )

    most_similar = PEOPLE[most_similar_index]

    similarity_score = float(
        similarity_values[most_similar_index]
    )

    recommend_score = float(
        df.loc[most_similar, RECOMMEND_MOVIE]
    )

    # 評価表を更新
    set_html(
        "rating-table",
        make_rating_html(df),
    )

    # 類似度表を更新
    set_html(
        "similarity-table",
        make_similarity_html(
            df_cos,
            most_similar,
        ),
    )

    # 推薦結果を更新
    set_text(
        "most-similar",
        most_similar,
    )

    set_text(
        "similarity-score",
        f"{similarity_score:.3f}",
    )

    set_text(
        "movie-score-label",
        f'{most_similar}の「{RECOMMEND_MOVIE}」評価',
    )

    set_text(
        "movie-score",
        f"{recommend_score:.1f} / 5",
    )

    set_text(
        "stars",
        stars(recommend_score),
    )

    set_text(
        "recommendation-text",
        (
            f'{TARGET_USER}に「{RECOMMEND_MOVIE}」を'
            "推薦する材料になります。"
        ),
    )


# =========================================================
# ボタンとPython関数を接続
# =========================================================

button = document.getElementById("recalculate")

# Python関数をJavaScriptイベントから呼べるProxyにする
recalculate_proxy = create_proxy(recalculate)

button.addEventListener(
    "click",
    recalculate_proxy,
)

# 初回表示
recalculate()
