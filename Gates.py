class Gates:
    def __init__(self):
        self.alpha, self.beta, self.state = 0, 0, 0

    def updateGateState(self, deltaT):
        self.state += (deltaT * (self.alpha * (1 - self.state) - self.beta * self.state))
        return self.state
