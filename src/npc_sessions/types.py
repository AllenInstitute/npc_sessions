import abc
import datetime
from collections.abc import MutableMapping

class Metadata(abc.ABC):
    """Abstract base class for metadata classes, with some convenience properties/methods."""
    
    @property
    @abc.abstractmethod
    def df(self) -> pd.DataFrame:
        ...

    @property
    def state(self) -> MutableMapping:
        ...

class Mouse(Metadata):
    ...

class Project(Metadata):
    ...

class Session(Metadata):

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.mouse=}, {self.date=}_{self.time=})'

    
    mouse: Mouse
    date: datetime.date
    time: datetime.time

    @property
    def dt(self) -> datetime.datetime:
        return datetime.datetime.combine(self.date, self.time)