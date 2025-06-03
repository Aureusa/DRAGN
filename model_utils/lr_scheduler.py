class DiscScheduler:
    def __init__(self, update_freq):
        self.update_freq = update_freq
        self.current_step = 0
        self.loss = []

    def step(self):
        self.current_step += 1
        if self.current_step % self.update_freq == 0:
            self._update_freq()
            return True
        return False
    
    def get_num_steps(self):
        return self.current_step
    
    def log_step(self, loss):
        self.loss.append(loss)

    def reset(self):
        self.current_step = 0
        self.loss = []

    def _update_freq(self):
        if len(self.loss) < 2:
            # Not enough data to determine trend
            return
        if self.loss[-1] > self.loss[0]:
            # If the loss is increasing, decrease the update frequency
            self.update_freq *= 2
        else:
            # If the loss is decreasing, increase the update frequency
            self.update_freq = max(1, self.update_freq // 2)
