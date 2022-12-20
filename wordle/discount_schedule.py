class DiscountSchedule:
    def get(self, iteration: int) -> float:
        ...


class ConstantDiscountSchedule(DiscountSchedule):
    def __init__(self, gamma: float):
        super().__init__()
        self.gamma = gamma

    def get(self, iteration: int) -> float:
        return self.gamma
