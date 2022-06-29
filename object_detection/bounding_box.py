from __future__ import annotations

from typing import List
from typing import NamedTuple


class BBox(NamedTuple):
    x: float
    y: float
    w: float
    h: float

    @property
    def area(self) -> float:
        return self.h * self.w

    @property
    def center_x(self):
        return self.x + self.w / 2

    @property
    def center_y(self):
        return self.y + self.h / 2

    def intersect(self, other: BBox) -> BBox:
        new_x1 = max(self.x, other.x)
        new_y1 = max(self.y, other.y)

        new_x2 = min(self.x + self.w, other.x + other.w)
        new_y2 = min(self.y + self.h, other.y + other.h)

        new_w = max(0.0, new_x2 - new_x1)
        new_h = max(0.0, new_y2 - new_y1)

        return BBox(x=new_x2, y=new_y2, w=new_w, h=new_h)

    def iou(self, other: BBox) -> float:
        intersection = self.intersect(other)
        return intersection.area / (self.area + other.area - intersection.area)

    @classmethod
    def from_array(cls, bbox: List[float], centered: bool = False):
        if not centered:
            return BBox(x=bbox[0], y=bbox[1], w=bbox[2], h=bbox[3])
        else:
            return BBox(x=bbox[0] - bbox[2] / 2, y=bbox[1] - bbox[3] / 2, w=bbox[2], h=bbox[3])

    def to_array(self, centered: bool = False) -> List[float]:
        if not centered:
            return [self.x, self.y, self.w, self.h]
        else:
            return [self.center_x, self.center_y, self.w, self.h]
