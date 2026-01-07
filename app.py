import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


st.set_page_config(page_title="Car Price Predictor", layout="wide")


@st.cache_resource
def load_pipeline(path: str = "model_pipeline.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


pipe = load_pipeline()
EXPECTED_COLS = list(getattr(pipe, "feature_names_in_", []))


def get_num_cat_cols(pipeline) -> tuple[list[str], list[str]]:
    try:
        ct = pipeline.named_steps["preprocess"]
    except Exception:
        return [], []

    num_cols, cat_cols = [], []
    for name, _, cols in getattr(ct, "transformers_", []):
        if name == "num":
            num_cols = list(cols)
        elif name == "cat":
            cat_cols = list(cols)
    return num_cols, cat_cols


NUM_COLS, CAT_COLS = get_num_cat_cols(pipe)


def prepare_X(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()

    for c in CAT_COLS:
        if c in X.columns:
            X[c] = X[c].astype("string").fillna("missing")

    for c in NUM_COLS:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
            if X[c].isna().any():
                med = X[c].median()
                if pd.isna(med):
                    med = 0.0
                X[c] = X[c].fillna(med)

    return X


def hist_plot(s: pd.Series, title: str):
    fig = plt.figure()
    plt.hist(s.dropna().values, bins=40)
    plt.title(title)
    plt.xlabel(s.name)
    plt.ylabel("count")
    st.pyplot(fig)


st.title("Car Price Predictor")


with st.expander("📊 EDA", expanded=True):
    eda_file = st.file_uploader("Загрузить CSV для EDA", type=["csv"])
    if eda_file is None:
        st.info("Загрузите CSV, чтобы увидеть графики")
    else:
        df_eda = pd.read_csv(eda_file)
        st.dataframe(df_eda.head(30), use_container_width=True)

        if "selling_price" in df_eda.columns:
            hist_plot(df_eda["selling_price"], "Target: selling_price")

        num_cols = df_eda.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            chosen = st.multiselect("Числовые колонки", num_cols, default=num_cols[:4])
            for c in chosen:
                hist_plot(df_eda[c], f"Distribution: {c}")

st.subheader("🧠 Предсказание цены")

if EXPECTED_COLS:
    st.caption("Ожидаемые признаки: " + ", ".join(EXPECTED_COLS))
else:
    st.warning("Не удалось определить список входных колонок (feature_names_in_)")

mode = st.radio("Ввод данных", ["CSV файл", "Ручной ввод"], index=0)

if mode == "CSV файл":
    infer_file = st.file_uploader("Загрузить CSV с признаками", type=["csv"], key="infer")
    if infer_file is not None:
        df = pd.read_csv(infer_file)

        if EXPECTED_COLS:
            missing = [c for c in EXPECTED_COLS if c not in df.columns]
            if missing:
                st.error(f"Не хватает колонок: {missing}")
                st.stop()
            X = df[EXPECTED_COLS]
        else:
            X = df

        X = prepare_X(X)
        preds = pipe.predict(X)

        out = df.copy()
        out["predicted_price"] = preds

        st.success("Готово")
        st.dataframe(out.head(50), use_container_width=True)

        st.download_button(
            "Скачать predictions.csv",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )

else:
    if not EXPECTED_COLS:
        st.error("Нет списка колонок для ручного ввода. Используйте режим CSV")
        st.stop()

    values = {}
    cols = st.columns(3)
    cat_set = set(CAT_COLS)

    for i, name in enumerate(EXPECTED_COLS):
        with cols[i % 3]:
            if name in cat_set:
                values[name] = st.text_input(name, value="")
            else:
                values[name] = st.number_input(name, value=0.0)

    X_one = prepare_X(pd.DataFrame([values]))

    if st.button("Предсказать"):
        pred = float(pipe.predict(X_one)[0])
        st.metric("Предсказанная цена", f"{pred:,.0f}")

st.subheader("🧾 Веса модели")

try:
    ct = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]

    feature_names = ct.get_feature_names_out()
    coefs = model.coef_

    df_coef = pd.DataFrame({"feature": feature_names, "coef": coefs})
    df_coef["abs"] = df_coef["coef"].abs()
    df_coef = df_coef.sort_values("abs", ascending=False)

    topn = st.slider("Топ признаков", 10, 80, 25)
    top = df_coef.head(topn)

    st.dataframe(top[["feature", "coef"]], use_container_width=True)

    fig = plt.figure()
    plt.barh(top["feature"][::-1], top["coef"][::-1])
    plt.title("Top coefficients")
    plt.xlabel("coef")
    st.pyplot(fig)

except Exception as e:
    st.warning("Не удалось построить веса")
    st.exception(e)
