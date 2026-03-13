from solver import Solver
from plotter import Plotter

# ОДУ, которое необходимо решить
def equations(y: list, t: float) -> list:
	dydt = y[1]
	d2ydt2 = -0.1 * y[1] - 3 * y[0]

	return [dydt, d2ydt2]

# Исходные данные для расчета
initial_values = (0.1, 0.0)
integration_time = 10.0

# Создание объекта решателя
sol = Solver(equations)

# Запуск решателя
sol.eiler_solve(initial_values, integration_time, 0.001)

# Визуализация решения
Plotter.plot(sol.t_solution, sol.y_solution[0])