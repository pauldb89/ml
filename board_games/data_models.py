from pydantic import BaseModel


class City(BaseModel):
    id: int
    name: str
