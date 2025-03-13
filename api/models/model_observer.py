class ModelObserver:
    def __init__(self, name: str):
        self.name = name
        self._active = False

    def update(self, active_model_name: str):
        if self.name != active_model_name:
            self._active = False

    @property
    def active(self) -> bool:
        return self._active

    @active.setter
    def active(self, value: bool):
        self._active = value


class ModelSubject:
    def __init__(self, classification: str):
        self.classification = classification
        self.observers = {}

    def add_observer(self, observer: ModelObserver):
        self.observers[observer.name] = observer

    def set_active(self, active_model_name: str):
        for model_name, observer in self.observers.items():
            if model_name == active_model_name:
                observer.active = True
            observer.update(active_model_name)

    def get_active_model(self):
        for model_name, observer in self.observers.items():
            if observer.active:
                return model_name
        return None
