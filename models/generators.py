from neurodiffeq.generators import BaseGenerator, Generator1D

class CurriculumGenerator1D(BaseGenerator):
    def __init__(self, steps, sizes, ts_min=0.0, ts_max=1.0, methods='uniform', noise_stds=None):
        self.steps = steps
        self.sizes = sizes if isinstance(sizes, list) else [sizes]*self.steps
        self.ts_min = ts_min if isinstance(ts_min, list) else [ts_min]*self.steps
        self.ts_max = ts_max if isinstance(ts_max, list) else [ts_max]*self.steps
        self.methods = methods if isinstance(methods, list) else [methods]*self.steps
        self.noise_stds = noise_stds if isinstance(noise_stds, list) else [noise_stds]*self.steps
        self.curriculum_step = 0
        self.generator = Generator1D(self.sizes[0], self.ts_min[0], self.ts_max[0], self.methods[0], self.noise_stds[0])
        self.t_min = self.generator.t_min
        self.t_max = self.generator.t_max

    def schedule_step(self, step=None):
        if step is not None:
            self.curriculum_step = step
        else:
            self.curriculum_step += 1
        self.generator = Generator1D(self.sizes[self.curriculum_step], self.ts_min[self.curriculum_step], self.ts_max[self.curriculum_step], self.methods[self.curriculum_step], self.noise_stds[self.curriculum_step])
        self.t_min = self.generator.t_min
        self.t_max = self.generator.t_max

    def get_examples(self):
        return self.generator.get_examples()
    
    @property
    def size(self):
        return self.generator.size