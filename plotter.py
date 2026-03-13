import matplotlib.pyplot as plt 

class Plotter:
	
	@staticmethod
	def plot(
		x_values:list,
		y_values:list,
		fig_size:tuple = (12, 8),
		x_scale:tuple = None,
		y_scale:tuple = None,
		label_names:tuple = None,
		title_name:str = None,
	) -> None:
		fig, ax = plt.subplots(figsize=fig_size, layout='tight')

		ax.grid(which='major', color='#DDDDDD', linewidth=1.5)
		ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=1)
		ax.minorticks_on()
		ax.grid(True)

		ax.plot(x_values, y_values, color='blue', linewidth=3)

		if not(x_scale is None):
			ax.set_xlim(x_scale[0], x_scale[1])
		if not(y_scale is None):
			ax.set_ylim(y_scale[0], y_scale[1])

		if not(label_names is None):
			plt.xlabel(label_names[0], fontsize=15, fontweight='bold')
			plt.ylabel(label_names[1], fontsize=15, fontweight='bold')

		if not(title_name is None):
			plt.title(title_name, fontsize=15, fontweight='bold')

		plt.show()
