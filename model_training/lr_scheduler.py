class DiscScheduler:
    def __init__(self, update_freq, max_update_freq=1000, min_update_freq=1):
        self.update_freq = update_freq

        self.max_update_freq = max_update_freq # Upper limit for update frequency
        self.min_update_freq = min_update_freq # Lower limit for update frequency

        # Initialize the current step and success step counters
        self.current_step = 0
        self.success_step = 0

        # Initialize a list to store the loss values of the generator
        # This will be used to adjust the update frequency based on the loss trend
        self.loss = []

    def step(self):
        self.current_step += 1
        if self.current_step % self.update_freq == 0:
            self._update_freq()
            self.success_step += 1
            return True
        return False
    
    def get_num_steps(self):
        return self.success_step
    
    def log_step(self, loss):
        self.loss.append(loss)

    def reset(self):
        self.current_step = 0
        self.success_step = 0
        self.loss = []

    def _update_freq(self):
        window = min(20, len(self.loss) // 2)

        if window < 1:
            # Not enough data to compute averages, do not change update frequency
            return
        
        # Compute the mean of the first and last N losses
        first_avg = sum(self.loss[:window]) / window
        last_avg = sum(self.loss[-window:]) / window

        if last_avg > first_avg:
            # Loss is increasing, decrease update frequency
            self.update_freq = min(self.update_freq * 2, self.max_update_freq)
        else:
            # Loss is decreasing, increase update frequency
            self.update_freq = max(self.min_update_freq, self.update_freq // 2)
