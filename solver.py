import numpy as np
from scipy.integrate import odeint
from typing import Callable, Optional, List, Union
import warnings


class ODESolver:
    """
    Решатель систем обыкновенных дифференциальных уравнений (ОДУ).

    Класс предоставляет различные методы численного интегрирования:
        - Фиксированный шаг (Эйлер, Рунге-Кутта 4 порядка)
        - Адаптивный шаг с контролем точности
        - Интегрирование через `scipy.integrate.odeint`

    Attribures:
        system (Callable): Функция, определяющая систему ОДУ.
            Должна иметь сигнатуру `system(y: array_like, t: float) -> array_like`
        t_solution (Optional[np.ndarray]): Временные точки решения.
            Заполняется после вызова метода `solve_*`.
        y_solution (Optional[np.ndarray]): Значения переменных в каждой временной точке.
            Формат: строки - переменные, столбцы - временные точки.

    Example:
        >>> # Определение системы ОДУ (гармонический осциллятор)
        >>> def oscillator(y, t):
        ...     return [y[1], -y[0]]
        >>>
        >>> # Создание и использование решателя
        >>> solver = ODESolver(oscillator)
        >>> solver.solve_fixed_step((1.0, 0.0), (0.0, 10.0), 0.01)
        >>> print(solver.y_solution.shape)
        (2, 1001)

    Notes:
        Все методы `solve_*` сохраняют результаты в атрибутах класса
        и не возвращают значений (None).
    """

    def __init__(self, system: Callable):
        """
        Инициализация решателя ОДУ.

        Args:
            system (Callable): Функция, описывающая правую часть системы ОДУ.
                Сигнатура функции: `system(y: array_like, t: float) -> array_like`
                где:
                    y (array_like): Текущие значения переменных
                    t (float): Текущее время
                    Возвращает: array_like - производные переменных по времени

        Example:
            >>> def pendulum(state, t):
            ...     theta, omega = state
            ...     return [omega, -9.81 * sin(theta)]
            >>> solver = ODESolver(pendulum)
        """
        self.system = system
        self.t_solution: Optional[np.ndarray] = None
        self.y_solution: Optional[np.ndarray] = None

    def solve_scipy(
        self,
        init_values: tuple,
        t_span: tuple,
        t_eval: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        """
        Численное интегрирование с использованием `scipy.integrate.odeint`.

        Использует высокоточный адаптивный решатель из библиотеки `SciPy`.
        Подходит для жестких систем и когда требуется высокая точность.

        Args:
            init_values (tuple): Начальные условия в момент времени `t_start`.
            t_span (tuple): Интервал интегрирования `(t_start, t_end)`.
            t_eval (Optional[np.ndarray]): Массив временных точек для вывода решения.
                Если None, используется равномерная сетка из 1000 точек.
            **kwargs: Дополнительные параметры, передаваемые в odeint:
                - rtol (float): Относительная точность (по умолчанию `1.49e-8`)
                - atol (float): Абсолютная точность (по умолчанию `1.49e-8`)
                - mxstep (int): Максимальное число шагов

        Notes:
            Результат транспонируется для удобства: строки соответствуют переменным,
            столбцы - временным точкам.

        Example:
            >>> solver.solve_scipy(
            ...     init_values=(1.0, 0.0),
            ...     t_span=(0, 10),
            ...     t_eval=np.linspace(0, 10, 500),
            ...     rtol=1e-6
            ... )
        """
        t_start, t_end = t_span

        if t_eval is None:
            self.t_solution = np.linspace(t_start, t_end, 1000)
        else:
            self.t_solution = np.array(t_eval)

        self.y_solution = odeint(self.system, init_values, self.t_solution, **kwargs)
        self.y_solution = self.y_solution.T

    def solve_fixed_step(
        self, init_values: tuple, t_span: tuple, step_size: float, method: str = "rk4"
    ) -> None:
        """
        Решение системы ОДУ с фиксированным шагом интегрирования.

        Args:
            init_values (tuple) : Начальные условия для всех переменных системы.
                Порядок должен соответствовать порядку функций в `system`.
            t_span (tuple) : Интервал интегрирования `(t_start, t_end)`.
            step_size (float) : Размер шага интегрирования. Должен быть положительным.
            method (srt) : Метод интегрирования (`euler` или `rk4`), по умолчанию `rk4`.

        Raises:
            ValueError: Если указан неизвестный метод интегрирования `method`.

        Returns:
            None
        """
        t_start, t_end = t_span
        n_steps = int((t_end - t_start) / step_size) + 1

        self.t_solution = np.linspace(t_start, t_end, n_steps)
        y = np.array(init_values)
        self.y_solution = [y]

        for i in range(1, n_steps):
            t = self.t_solution[i - 1]
            dt = self.t_solution[i] - t

            if method.lower() == "euler":
                y = y + dt * np.array(self.system(y, t))
            elif method.lower() == "rk4":
                y = self.__rk4_step(y, t, dt)
            else:
                raise ValueError(f"Неизвестный метод: {method}")

            self.y_solution.append(y)

        self.y_solution = np.array(self.y_solution).T

    def solve_adaptive(
        self,
        init_values: tuple,
        t_span: tuple,
        accuracy: float,
        method: str = "rk4",
        max_step: float = 1.0,
        min_step: float = 1e-10,
    ) -> None:
        """
        Численное интегрирование системы ОДУ с адаптивным шагом и контролем точности.

        Реализует интегрирование с автоматическим подбором шага для достижения
        заданной точности. Использует правило Рунге для оценки локальной погрешности
        и динамически корректирует размер шага.

        Args:
            init_values (tuple): Начальные условия для всех переменных системы в момент
                времени `t_start`. Длина кортежа должна соответствовать количеству
                уравнений в системе.
            t_span (tuple): Интервал интегрирования `(t_start, t_end)`.
            accuracy (float): Требуемая локальная точность интегрирования.
            method (str): Метод численного интегрирования:
                - "euler" : метод Эйлера 1-го порядка
                - "rk4" : метод Рунге-Кутты 4-го порядка
                По умолчанию "rk4".
            max_step (float): Максимально допустимый размер шага интегрирования.
            min_step (float): Минимально допустимый размер шага интегрирования.

        Raises:
            ValueError: Если указан неизвестный метод интегрирования `method`.

        Warns:
            UserWarning: Если шаг интегрирования становится меньше `min_step`.
                В этом случае процесс останавливается, а полученное до этого
                момента решение сохраняется.

        Returns:
            None

        Examples:
            >>> # Пример 1: Гармонический осциллятор
            >>> def oscillator(y, t):
            ...     return [y[1], -y[0]]
            >>>
            >>> solver = ODESolver(oscillator)
            >>> solver.solve_adaptive(
            ...     init_values=(1.0, 0.0),
            ...     t_span=(0, 10),
            ...     accuracy=1e-6,
            ...     method='rk4'
            ... )
        """
        t_start, t_end = t_span
        t_current = t_start
        y_current = np.array(init_values)

        self.t_solution = [t_current]
        self.y_solution = [y_current]

        step_size = min(accuracy ** (0.5 if method == "euler" else 0.2), max_step)
        max_iterations = 100000
        iteration = 0

        while t_current < t_end and iteration < max_iterations:
            if step_size < min_step:
                warnings.warn(
                    f"Шаг стал слишком маленьким: {step_size}. Уменьшите допустимый минимальный шаг `min_step`. Интегрирование прервано"
                )
                break

            if t_current + step_size > t_end:
                step_size = t_end - t_current

            if method.lower() == "euler":
                y_new, error, step_size = self.__adaptive_euler_step(
                    y_current, t_current, step_size, accuracy
                )
            elif method.lower() == "rk4":
                y_new, error, step_size = self.__adaptive_rk4_step(
                    y_current, t_current, step_size, accuracy
                )
            else:
                raise ValueError(f"Неизвестный метод: {method}")

            if error < accuracy:
                t_current += step_size
                y_current = y_new

                self.t_solution.append(t_current)
                self.y_solution.append(y_current)

                if error > 0:
                    step_size = min(
                        step_size * min(2.0, 0.9 * (accuracy / error) ** 0.2), max_step
                    )
            else:
                pass

            iteration += 1

        self.t_solution = np.array(self.t_solution)
        self.y_solution = np.array(self.y_solution).T

    def __rk4_step(self, y: np.ndarray, t: float, dt: float) -> np.ndarray:
        """
        Выполнение одного шага интегрирования методом Рунге-Кутты 4-го порядка.

        Реализует классический метод Рунге-Кутты 4-го порядка для численного
        решения системы ОДУ. Метод использует четыре промежуточных вычисления
        производных для достижения 4-го порядка точности.

        Args:
            y (np.ndarray): Текущее состояние системы.
            t (float): Текущее время.
            dt (float): Предлагаемый размер шага.

        Returns:
            np.ndarray: Вектор новых значений переменных после шага интегрирования.

        Private method:
            Этот метод предназначен для внутреннего использования и
            не должен вызываться напрямую пользователем.
        """
        k1 = np.array(self.system(y, t))
        k2 = np.array(self.system(y + dt / 2 * k1, t + dt / 2))
        k3 = np.array(self.system(y + dt / 2 * k2, t + dt / 2))
        k4 = np.array(self.system(y + dt * k3, t + dt))

        return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def __adaptive_rk4_step(
        self, y: np.ndarray, t: float, dt: float, accuracy: float
    ) -> tuple:
        """
        Выполнение одного адаптивного шага методом Рунге-Кутты 4 порядка.

        Реализует правило Рунге для оценки локальной ошибки путем сравнения
        результатов с полным и половинным шагом.

        Args:
            y (np.ndarray): Текущее состояние системы.
            t (float): Текущее время.
            dt (float): Предлагаемый размер шага.
            accuracy (float): Требуемая точность на шаге.

        Returns:
            tuple: Кортеж из трех элементов:
                - y_new (np.ndarray): Новое состояние системы (если шаг принят)
                или исходное состояние (если шаг отвергнут).
                - error (float): Оценка локальной ошибки.
                - dt_new (float): Скорректированный размер шага для следующей итерации.

        Notes:
            Алгоритм:
                1. Вычисляется решение с шагом `dt (y_full)`
                2. Вычисляется решение с двумя шагами `dt/2 (y_half)`
                3. Ошибка оценивается как `max|y_half - y_full|`
                4. Если `ошибка > accuracy`, шаг уменьшается
                5. Иначе шаг может быть увеличен для следующей итерации

        Private method:
            Этот метод предназначен для внутреннего использования и
            не должен вызываться напрямую пользователем.
        """
        y_full = self.__rk4_step(y, t, dt)

        y_half = self.__rk4_step(y, t, dt / 2)
        y_half = self.__rk4_step(y_half, t + dt / 2, dt / 2)

        error = np.max(np.abs(y_half - y_full))

        if error > accuracy and error > 0:
            dt = dt * max(0.5, 0.9 * (accuracy / error) ** 0.2)
            return y, error, dt

        return y_full, error, dt

    def __adaptive_euler_step(
        self, y: np.ndarray, t: float, dt: float, accuracy: float
    ) -> tuple:
        """
        Выполнение одного адаптивного шага методом Эйлера.

        Args:
            y (np.ndarray): Текущее состояние системы.
            t (float): Текущее время.
            dt (float): Предлагаемый размер шага.
            accuracy (float): Требуемая точность на шаге.

        Returns:
            tuple: Кортеж из трех элементов:
                - y_new (np.ndarray): Новое состояние системы (если шаг принят)
                или исходное состояние (если шаг отвергнут).
                - error (float): Оценка локальной ошибки.
                - dt_new (float): Скорректированный размер шага для следующей итерации.

        Notes:
            Алгоритм:
                1. Вычисляется решение с шагом `dt (y_full)`
                2. Вычисляется решение с двумя шагами `dt/2 (y_half)`
                3. Ошибка оценивается как `max|y_half - y_full|`
                4. Если `ошибка > accuracy`, шаг уменьшается
                5. Иначе шаг может быть увеличен для следующей итерации

        Private method:
            Этот метод предназначен для внутреннего использования и
            не должен вызываться напрямую пользователем.
        """
        k = np.array(self.system(y, t))

        y_full = y + dt * k

        y_half1 = y + dt / 2 * k
        k_half = np.array(self.system(y_half1, t + dt / 2))
        y_half_full = y_half1 + dt / 2 * k_half

        error = np.max(np.abs(y_half_full - y_full))

        if error > accuracy and error > 0:
            dt = dt * max(0.5, 0.9 * (accuracy / error))
            return y, error, dt

        return y_full, error, dt
