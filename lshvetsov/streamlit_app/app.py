from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st

from lshvetsov.data import SEOUL_TIMEZONE
from lshvetsov.data.features import FEATURE_LABELS
from lshvetsov.data.protocol import SCHEME_LABELS
from lshvetsov.streamlit_app.predict import load_model_schemes, predict_rentals


VISIBILITY = "Visibility (10m)"
WEATHER_INPUTS = {
    "Temperature(°C)": ("Температура, °C", None, None),
    "Wind speed (m/s)": ("Скорость ветра, м/с", 0.0, None),
    VISIBILITY: ("Видимость, км", 0.0, None),
    "Rainfall(mm)": ("Дождь, мм", 0.0, None),
    "Humidity(%)": ("Влажность, %", 0.0, 100.0),
    "Solar Radiation (MJ/m2)": ("Солнечная радиация, МДж/м²", 0.0, None),
    "Snowfall (cm)": ("Снег, см", 0.0, None),
}
REBUILD_MESSAGE = (
    "Чтобы создать файлы моделей, выполните команды train и finalize из README проекта."
)


def main() -> None:
    st.set_page_config(page_title="Прогноз велопроката в Сеуле", layout="centered")
    st.title("Прогноз велопроката в Сеуле")
    st.caption("Оценка числа аренд за один час. Дата и время — местные, по Сеулу.")

    try:
        models = load_model_schemes()
    except FileNotFoundError:
        st.error(REBUILD_MESSAGE)
        return
    except (KeyError, TypeError, ValueError) as error:
        st.error(f"Ошибка чтения манифеста моделей. {error}")
        return

    scheme = st.selectbox(
        "Схема проверки", list(models), format_func=SCHEME_LABELS.__getitem__, key="evaluation_scheme"
    )
    feature_set = st.selectbox(
        "Сценарий", list(models[scheme]), format_func=FEATURE_LABELS.__getitem__, key="feature_set"
    )
    model = models[scheme][feature_set]

    now = datetime.now(ZoneInfo(SEOUL_TIMEZONE))
    date_column, hour_column = st.columns([2, 1])
    with date_column:
        day = st.date_input("Дата", value=now.date(), format="DD.MM.YYYY", key="rental_date")
    with hour_column:
        hour = st.selectbox(
            "Час",
            range(24),
            index=now.hour,
            format_func=lambda value: f"{value:02d}:00",
            key="rental_hour",
        )
    is_holiday = st.checkbox("Праздничный день")
    status = st.radio("Статус проката", ["Работает", "Закрыт"], horizontal=True)
    is_open = status == "Работает"

    weather = {}
    if is_open and model.weather_columns:
        st.subheader("Погодные условия")
        columns = st.columns(2)
        for index, field in enumerate(model.weather_columns):
            label, minimum, maximum = WEATHER_INPUTS[field]
            with columns[index % len(columns)]:
                value = st.number_input(
                    label,
                    min_value=minimum,
                    max_value=maximum,
                    value=None,
                    step=0.1,
                    key=f"weather:{field}",
                )
            if value is not None:
                # The dataset records visibility in units of ten metres.
                weather[field] = value * 100 if field == VISIBILITY else value

    forecast = None
    if st.button("Рассчитать прогноз", type="primary"):
        if day is None:
            st.error("Укажите дату.")
        elif is_open and len(weather) != len(model.weather_columns):
            st.error("Заполните все погодные поля. Если осадков нет, укажите 0.")
        else:
            try:
                forecast = predict_rentals(
                    model,
                    day=day,
                    hour=hour,
                    is_holiday=is_holiday,
                    is_open=is_open,
                    weather=weather,
                )
            except FileNotFoundError:
                st.error(REBUILD_MESSAGE)
            except ValueError as error:
                st.error(str(error))

    with st.container(border=True):
        if forecast is not None:
            st.metric("Прогноз, аренд за час", f"{forecast:.0f}")
        if forecast is not None and not is_open:
            st.caption("Прокат закрыт. Нулевой прогноз рассчитан по правилу закрытия.")

        st.caption(f"Отложенный тест выбранной схемы · {model.test_hours} открытых часов")
        overall, peaks = st.columns(2)
        overall.metric("RMSE", f"{model.test_rmse:.2f}".replace(".", ","))
        peaks.metric(
            "Peak RMSE",
            f"{model.test_peak_rmse:.2f}".replace(".", ","),
            help="RMSE на 10% открытых часов с самым высоким фактическим спросом.",
        )
        overall.metric("MAE", f"{model.test_mae:.2f}".replace(".", ","))
        peaks.metric("R²", f"{model.test_r2:.2f}".replace(".", ","))

        alpha = model.parameters.get("alpha")
        alpha_label = "" if alpha is None else f" · α = {alpha:g}"
        algorithm = (
            "Медиана + поправка бустинга"
            if model.algorithm == "MedianResidualBoosting" else model.algorithm
        )
        st.caption(f"{algorithm}{alpha_label}")
        rmse = f"{model.validation_rmse:.2f}".replace(".", ",")
        st.caption(f"Модель выбрана по валидационной RMSE {rmse} аренд за час.")
        if "max_leaf_nodes" in model.parameters:
            st.caption(
                f"{model.parameters['max_iter']} итераций · "
                f"до {model.parameters['max_leaf_nodes']} листьев · "
                f"от {model.parameters['min_samples_leaf']} наблюдений в листе · "
                f"скорость обучения {model.parameters['learning_rate']:g}"
            )
        if model.weather_columns:
            st.caption("Метрики рассчитаны с фактической погодой из датасета.")


if __name__ == "__main__":
    main()
