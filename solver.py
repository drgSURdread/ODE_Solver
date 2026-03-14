import numpy as np
from scipy.integrate import odeint
from typing import Callable, Optional, List, Union
import warnings


class ODESolver:
    """
    Решатель систем обыкновенных дифференциальных уравнений
    """

    def __init__(self, system: Callable):
        """
        Инициализация решателя

        Args:
            system: функция, представляющая систему ОДУ
                   должна принимать (y, t) и возвращать производные
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
        Решение с использованием scipy.integrate.odeint

        Args:
            init_values: начальные условия
            t_span: интервал интегрирования (t_start, t_end)
            t_eval: точки для вывода решения
            **kwargs: дополнительные параметры для odeint
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
        Решение с фиксированным шагом

        Args:
            init_values: начальные условия
            t_span: интервал интегрирования (t_start, t_end)
            step_size: размер шага
            method: метод ('euler' или 'rk4')
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
        Решение с адаптивным шагом

        Args:
            init_values: начальные условия
            t_span: интервал интегрирования (t_start, t_end)
            accuracy: требуемая точность
            method: метод ('euler' или 'rk4')
            max_step: максимальный размер шага
            min_step: минимальный размер шага
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
        """Один шаг метода Рунге-Кутты 4-го порядка"""
        k1 = np.array(self.system(y, t))
        k2 = np.array(self.system(y + dt / 2 * k1, t + dt / 2))
        k3 = np.array(self.system(y + dt / 2 * k2, t + dt / 2))
        k4 = np.array(self.system(y + dt * k3, t + dt))

        return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def __adaptive_rk4_step(
        self, y: np.ndarray, t: float, dt: float, accuracy: float
    ) -> tuple:
        """
        Адаптивный шаг метода Рунге-Кутты 4-го порядка

        Returns:
            (y_new, error, new_dt)
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
        Адаптивный шаг метода Эйлера

        Returns:
            (y_new, error, new_dt)
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
