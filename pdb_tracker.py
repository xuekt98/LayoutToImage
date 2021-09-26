import os
import re

class linesWrapper(object):
		def __init__(self, pattern, filepath):
				self.file = filepath
				self.pattern = pattern
				self.__load_lines__()

		def __load_lines__(self):
				self.lines_id = []
				with open(self.file, 'r') as f:
						lines = f.readlines()

				for i, line in enumerate(lines):
						if re.match(self.pattern, line):
								self.lines_id.append(i+1)

		def __str__(self):
				lines = '\n\t'.join(self.lines)
				return f"{self.file}:{lines}"

		def get_context(self):
				with open(self.file, 'r') as f:
						lines = f.readlines()
				for i, line in enumerate(lines):
						if re.match(self.pattern, line):
								print(f"{i+1}:")
								try:
									print(lines[i-1])
								except:
										pass
								try:
									print(lines[i+1])
								except:
										pass

				return

		def delete(self, line_id):
				with open(self.file, 'r') as f:
						lines = f.readlines()
				with open(self.file, 'w') as f:
						for i, line in enumerate(lines):
								if re.match(self.pattern, line) is None:
										f.write(line)
								elif line_id != i+1:
										f.write(line)
				self.__load_lines__()
				return

		def clear(self):
				with open(self.file, 'r') as f:
						lines = f.readlines()
				with open(self.file, 'w') as f:
						for line in lines:
								if re.match(self.pattern, line) is None:
										f.write(line)
				self.__load_lines__()
				return


def tracker(path='./'):
		breakpoints = []
		if os.path.isdir(path):
				for subp in os.listdir(path):
						breakpoints += tracker(os.path.join(path, subp))
		elif re.match('.*?\.py$', path):
				breakpoints.append(linesWrapper('.*?pdb\.set_trace\(\)', path))
		return breakpoints

def fire_all(path='./'):
		breakpoints = tracker(path)
		for points in breakpoints:
				if len(points.lines_id) > 0:
					points.clear()

import argparse

if __name__ == '__main__':
		parser = argparse.ArgumentParser()
		parser.add_argument("--path", type=str, default='./')
		args = parser.parse_args()
		fire_all(path=args.path)