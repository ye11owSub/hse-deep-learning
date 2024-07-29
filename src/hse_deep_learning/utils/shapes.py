from pydantic import BaseModel, computed_field


class Rect(BaseModel):
    left: float
    top: float
    width: float
    height: float

    @computed_field
    @property
    def right(self) -> float:
        return self.left + self.width - 1

    @computed_field
    @property
    def bottom(self) -> float:
        return self.top + self.height - 1

    @computed_field
    @property
    def center_x(self) -> float:
        return self.left + self.width / 2

    @computed_field
    @property
    def center_y(self) -> float:
        return self.top + self.height / 2

    @computed_field
    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    @computed_field
    @property
    def area(self) -> float:
        return self.width * self.height

    def resize(self, target_width: float, target_height: float) -> "Rect":
        target_aspect_ratio = target_width / target_height

        new_width = target_aspect_ratio * self.height
        new_left = self.left - (new_width - self.width) / 2

        return Rect(left=new_left, top=self.top, width=new_width, height=self.height)

    def clip(self, other: "Rect") -> "Rect":
        left = max(other.left, self.left)
        right = min(other.right, self.right)
        top = max(other.top, self.top)
        bottom = min(other.bottom, self.bottom)

        new_width = right - left + 1
        new_height = bottom - top + 1

        return Rect(left=left, top=top, width=new_width, height=new_height)
