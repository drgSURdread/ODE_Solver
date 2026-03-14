import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Union, List, Tuple, Dict, Any
from pathlib import Path


class Plotter:
    """
    Класс для визуализации решений систем ОДУ и других данных.

    Предоставляет статические методы для создания качественных графиков
    с настраиваемым оформлением и поддержкой различных форматов вывода.

    Examples:
        >>> solver = ODESolver(oscillator)
        >>> solver.solve_fixed_step((1.0, 0.0), (0, 10), 0.01)
        >>>
        >>> # Базовый график
        >>> Plotter.plot(
        ...     x_values=solver.t_solution,
        ...     y_values=solver.y_solution[0],
        ...     title_name="Гармонический осциллятор",
        ...     label_names=("Время t, c.", "Перемещение x, м.")
        ... )
        >>>
        >>> # График с несколькими кривыми
        >>> Plotter.plot_multiple(
        ...     x_values=solver.t_solution,
        ...     y_values_list=[
        ...         solver.y_solution[0],
        ...         solver.y_solution[1]
        ...     ],
        ...     labels=("Перемещение x, м.", "Скорость v, м/с"),
        ...     colors=("blue", "red"),
        ...     title_name="Переменные",
        ...     save_path="oscillator.png"
        ... )
    """

    # Стандартные настройки оформления
    DEFAULT_FIGSIZE = (12, 8)
    DEFAULT_GRID_COLOR_MAJOR = "#DDDDDD"
    DEFAULT_GRID_COLOR_MINOR = "#EEEEEE"
    DEFAULT_LINE_COLOR = "blue"
    DEFAULT_LINE_WIDTH = 3
    DEFAULT_FONT_SIZE = 15
    DEFAULT_FONT_WEIGHT = "bold"

    @staticmethod
    def plot(
        x_values: Union[List, np.ndarray],
        y_values: Union[List, np.ndarray],
        fig_size: Tuple[int, int] = DEFAULT_FIGSIZE,
        x_lim: Optional[Tuple[float, float]] = None,
        y_lim: Optional[Tuple[float, float]] = None,
        label_names: Optional[Tuple[str, str]] = None,
        title_name: Optional[str] = None,
        line_color: str = DEFAULT_LINE_COLOR,
        line_width: int = DEFAULT_LINE_WIDTH,
        line_style: str = "-",
        marker: Optional[str] = None,
        marker_size: int = 6,
        grid: bool = True,
        font_size: int = DEFAULT_FONT_SIZE,
        font_weight: str = DEFAULT_FONT_WEIGHT,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True,
        ax: Optional[plt.Axes] = None,
        **kwargs,
    ) -> Optional[plt.Figure]:
        """
        Построение графика одной кривой.

        Args:
            x_values: Массив значений по оси X.
            y_values: Массив значений по оси Y.
            fig_size: Размер фигуры `(ширина, высота)` в дюймах.
                По умолчанию `(12, 8)`.
            x_lim: Пределы по оси X `(x_min, x_max)`.
                По умолчанию None (автоматический масштаб).
            y_lim: Пределы по оси Y `(y_min, y_max)`.
                По умолчанию None (автоматический масштаб).
            label_names: Подписи осей `(x_label, y_label)`.
                По умолчанию None.
            title_name: Заголовок графика. По умолчанию None.
            line_color: Цвет линии. По умолчанию `blue`.
            line_width: Толщина линии. По умолчанию `3`.
            line_style: Стиль линии. По умолчанию `-` (сплошная).
                Другие варианты: `--` (пунктир), `:` (точки), `-.` (штрих-пунктир).
            marker: Маркер для точек. По умолчанию None.
                Примеры: `o` (кружки), `s` (квадраты), `^` (треугольники).
            marker_size: Размер маркера. По умолчанию `6`.
            grid: Включить/выключить сетку. По умолчанию `True`.
            font_size: Размер шрифта подписей. По умолчанию `15`.
            font_weight: Насыщенность шрифта. По умолчанию `bold`.
            save_path: Путь для сохранения графика. По умолчанию None (не сохранять).
            show: Показать график. По умолчанию `True`.
            ax: Существующая ось для добавления графика. По умолчанию None.
            **kwargs: Дополнительные параметры для plt.plot().

        Returns:
            Optional[plt.Figure]: Объект фигуры, если создавалась новая,
                иначе None.

        Raises:
            ValueError: Если входные данные некорректны.

        Examples:
            >>> # Простой график
            >>> Plotter.plot(
            ...     x_values=[0, 1, 2, 3, 4],
            ...     y_values=[0, 1, 4, 9, 16],
            ...     title_name="Квадратичная функция",
            ...     label_names=("x", "y = x²")
            ... )
            >>>
            >>> # График с маркерами и сохранением
            >>> Plotter.plot(
            ...     x_values=time,
            ...     y_values=solution,
            ...     line_color="red",
            ...     line_style="--",
            ...     marker="o",
            ...     marker_size=4,
            ...     save_path="plot.png",
            ... )
        """
        x_values = np.array(x_values)
        y_values = np.array(y_values)

        if x_values.shape != y_values.shape:
            raise ValueError(
                f"Размеры x_values {x_values.shape} и y_values {y_values.shape} "
                f"должны совпадать"
            )

        if ax is None:
            fig, ax = plt.subplots(figsize=fig_size, layout="tight")
            created_new = True
        else:
            fig = ax.get_figure()
            created_new = False

        if grid:
            ax.grid(
                which="major",
                color=Plotter.DEFAULT_GRID_COLOR_MAJOR,
                linewidth=1.5,
                alpha=0.8,
            )
            ax.grid(
                which="minor",
                color=Plotter.DEFAULT_GRID_COLOR_MINOR,
                linestyle=":",
                linewidth=1,
                alpha=0.8,
            )
            ax.minorticks_on()

        line_args = {
            "color": line_color,
            "linewidth": line_width,
            "linestyle": line_style,
            **kwargs,
        }

        if marker:
            line_args["marker"] = marker
            line_args["markersize"] = marker_size

        ax.plot(x_values, y_values, **line_args)

        if x_lim is not None:
            ax.set_xlim(x_lim[0], x_lim[1])
        if y_lim is not None:
            ax.set_ylim(y_lim[0], y_lim[1])

        if label_names is not None:
            ax.set_xlabel(label_names[0], fontsize=font_size, fontweight=font_weight)
            ax.set_ylabel(label_names[1], fontsize=font_size, fontweight=font_weight)

        if title_name is not None:
            ax.set_title(title_name, fontsize=font_size + 2, fontweight=font_weight)

        if save_path is not None:
            plt.savefig(save_path, dpi=100, bbox_inches="tight")

        if show:
            plt.show()
        elif not show and created_new:
            plt.close(fig)

        return fig if created_new else None

    @staticmethod
    def plot_multiple(
        x_values: Union[List, np.ndarray],
        y_values_list: List[Union[List, np.ndarray]],
        labels: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        line_styles: Optional[List[str]] = None,
        fig_size: Tuple[int, int] = DEFAULT_FIGSIZE,
        x_lim: Optional[Tuple[float, float]] = None,
        y_lim: Optional[Tuple[float, float]] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        title_name: Optional[str] = None,
        grid: bool = True,
        legend: bool = True,
        legend_loc: str = "",
        font_size: int = DEFAULT_FONT_SIZE,
        save_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> plt.Figure:
        """
        Построение нескольких кривых на одном графике.

        Args:
            x_values: Общий массив значений по оси X для всех кривых.
            y_values_list: Список массивов значений по оси Y для каждой кривой.
            labels: Подписи кривых для легенды. По умолчанию None.
            colors: Цвета кривых. По умолчанию None (автоматический выбор).
            line_styles: Стили линий для каждой кривой. По умолчанию None.
            fig_size: Размер фигуры (ширина, высота).
            x_lim: Пределы по оси X.
            y_lim: Пределы по оси Y.
            x_label: Подпись оси X.
            y_label: Подпись оси Y.
            title_name: Заголовок графика.
            grid: Включить сетку.
            legend: Включить легенду.
            legend_loc: Расположение легенды.
            font_size: Размер шрифта.
            save_path: Путь для сохранения.
            **kwargs: Дополнительные параметры для настройки линий.

        Returns:
            plt.Figure: Объект фигуры.

        Examples:
            >>> # Перемещение и скорость осциллятора
            >>> Plotter.plot_multiple(
            ...     x_values=solver.t_solution,
            ...     y_values_list=[
            ...         solver.y_solution[0],
            ...         solver.y_solution[1]
            ...     ],
            ...     labels=["Смещение", "Скорость"],
            ...     colors=["blue", "red"],
            ...     x_label="Время t",
            ...     y_label="Значение",
            ...     title_name="Фазовые переменные осциллятора"
            ... )
        """
        fig, ax = plt.subplots(figsize=fig_size, layout="tight")

        if grid:
            ax.grid(True, alpha=0.3)

        if colors is None:
            colors = plt.cm.tab10(np.linspace(0, 1, len(y_values_list)))

        for i, y_values in enumerate(y_values_list):
            line_args = {
                "color": colors[i] if i < len(colors) else None,
                "linewidth": kwargs.get("linewidth", 2),
                **kwargs,
            }
            if line_styles and i < len(line_styles):
                line_args["linestyle"] = line_styles[i]

            label = labels[i] if labels and i < len(labels) else None
            ax.plot(x_values, y_values, label=label, **line_args)

        if x_lim:
            ax.set_xlim(x_lim)
        if y_lim:
            ax.set_ylim(y_lim)
        if x_label:
            ax.set_xlabel(x_label, fontsize=font_size)
        if y_label:
            ax.set_ylabel(y_label, fontsize=font_size)
        if title_name:
            ax.set_title(title_name, fontsize=font_size + 2)
        if legend and labels:
            ax.legend(loc=legend_loc)

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=kwargs.get("dpi", 100))

        plt.show()
        return fig
