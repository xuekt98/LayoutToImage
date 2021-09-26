from torch.nn import Module
import time

class ModuleExt(Module):
	def __init__(self):
		super(ModuleExt, self).__init__()
		self.init_b = time.time()
		return self.init_b

	def time_from_b(self):
		return time.time() - self.init_b

	def time_cur(self):
		pass
