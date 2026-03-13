import numpy as np
from scipy.integrate import odeint


class Solver:
	def __init__(self, system: object) -> object:
		"""
		Конструктор класса Solver

		Args:
			system (function): ссылка на функцию, представляющая систему ОДУ

		Returns:
			object: объект класса Solver
		"""
		self.system = system
		self.t_solution = []
		self.y_solution = []

	def solve_by_user_step(
		self, 
		init_values: tuple,
		integration_time: float,
		step_size: float,
		t_eval: list = None
	) -> None:
		"""
		Функция решателя, получающая решение для `self.system`.
		Указанный решатель получает решение с постоянным шагом,
		если не переданы временные точки интегрирования в параметр
		`t_eval`.

		Args:
			init_values (tuple): набор начальных условий
			integration_time (float): время интегрирования
			step_size (float): шаг решателя
			t_eval (list, optional): пользовательские точки 
			интегрирования. Defaults to None

		Returns:
			None 
		"""
		if t_eval is None:
			self.t_solution = np.linspace(0.0, integration_time, int(integration_time / step_size))
		else:
			self.t_solution = t_eval

		self.y_solution = odeint(self.system, init_values, self.t_solution)
		self.y_solution = np.transpose(self.y_solution)

	def eiler_solve(
		self,
		init_values: tuple,
		integration_time: float,
		accuracy: float,
	) -> None:
		"""
		Функция решателя, получающая решение для `self.system`
		методом Эйлера.

		Args:
			init_values (tuple): набор начальных условий
			integration_time (float): время интегрирования
			accuracy (float): точность интегрирования

		Returns:
			None 
		"""
		system_order = len(init_values)
		step_size = accuracy ** 0.5

		self.t_solution = [0.0]
		self.y_solution = [[init_value] for init_value in init_values]

		while self.t_solution[-1] < integration_time:
			derived_values = self.system(
				[self.y_solution[i][-1] for i in range(system_order)],
				self.t_solution[-1]
			)

			next_step_values = [
				self.y_solution[i][-1] + step_size * derived_values[i]
				for i in range(system_order)
			]
			next_2_step_values = [
				self.y_solution[i][-1] + step_size / 2.0 * derived_values[i]
				for i in range(system_order)
			]

			if all(abs(next_2_step_values[i] - next_step_values[i]) < accuracy for i in range(system_order)):
				self.t_solution.append(self.t_solution[-1] + step_size)

				for i in range(system_order):
					self.y_solution[i].append(next_step_values[i])
					
				step_size = 2.0 * step_size
			else:
				step_size = step_size / 2.0

		self.y_solution = np.array(self.y_solution)

	def __get_rk_coefficients(self, step_size:float) -> tuple:
		"""
		Функция вычисляющая коэффициенты для вычисления шага
		в методе Рунге-Кутты 4-го порядка

		Args:
			step_size (float): текущий шаг интегрирования

		Returns:
			tuple: значения поправок в значениях функции и производной на следующем шаге 
		"""
		temp_function_values = self.y_solution[-1]
		
		temp_system_values = self.system(temp_function_values, self.t_solution[-1])
		k1 = step_size * temp_system_values[0]
		l1 = step_size * temp_system_values[1]

		temp_function_values = [temp_function_values[0] + k1 / 2.0, temp_function_values[1] + l1 / 2.0]
		temp_system_values = self.system(temp_function_values, self.t_solution[-1] + step_size / 2.0)
		k2 = step_size * temp_system_values[0]
		l2 = step_size * temp_system_values[1]

		temp_function_values = [temp_function_values[0] + k2 / 2.0, temp_function_values[1] + l2 / 2.0]
		temp_system_values = self.system(temp_function_values, self.t_solution[-1] + step_size / 2.0)
		k3 = step_size * temp_system_values[0]
		l3 = step_size * temp_system_values[1]

		temp_function_values = [temp_function_values[0] + k3, temp_function_values[1] + l3]
		temp_system_values = self.system(temp_function_values, self.t_solution[-1] + step_size)
		k4 = step_size * temp_system_values[0]
		l4 = step_size * temp_system_values[1]

		return (k1 + 2.0 * k2 + 2.0 * k3 + k4, l1 + 2.0 * l2 + 2.0 * l3 + l4)

	def rk4_solve(
		self,
		init_values: tuple,
		integration_time: float,
		accuracy: float,
	) -> None:
		"""
		Функция решателя, получающая решение для `self.system`
		методом Рунге-Кутты 4-го порядка.

		Args:
			init_values (tuple): набор начальных условий
			integration_time (float): время интегрирования
			accuracy (float): точность интегрирования

		Returns:
			None 
		"""
		step_size = accuracy ** 0.5

		self.t_solution = [0.0]
		self.y_solution = [[init_value] for init_value in init_values]

		while self.t_solution[-1] < integration_time:
			rk4_coefficients = self.__get_rk_coefficients(step_size)
			rk4_small_step_coefficients = self.__get_rk_coefficients(step_size / 2.0)

			next_step_function_value = self.y_solution[0][-1] + 1.0/6.0 * rk4_coefficients[0]
			next_step_dot_function_value = self.y_solution[1][-1] + 1.0/6.0 * rk4_coefficients[1]

			next_2_step_function_value = self.y_solution[0][-1] + 1.0/6.0 * rk4_small_step_coefficients[0]
			next_2_step_dot_function_value = self.y_solution[1][-1] + 1.0/6.0 * rk4_small_step_coefficients[1]

			if abs(next_2_step_function_value - next_step_function_value) < accuracy:
				self.t_solution.append(self.t_solution[-1] + step_size)
				self.y_solution[0].append(next_step_function_value)
				self.y_solution[1].append(next_step_dot_function_value)
				step_size = 2.0 * step_size
			else:
				step_size = step_size / 2.0

		self.y_solution = np.array(self.y_solution)