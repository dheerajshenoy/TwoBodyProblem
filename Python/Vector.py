import numpy as np

class Vector:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x + other.x, self.y + other.y, self.z + other.z)
    
        elif isinstance(other, (int, float)):
            return Vector(self.x + other, self.y + other, self.z + other)

    def __ladd__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x + other.x, self.y + other.y, self.z + other.z)
    
        elif isinstance(other, (int, float)):
            return Vector(self.x + other, self.y + other, self.z + other)

    def __radd__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x + other.x, self.y + other.y, self.z + other.z)
    
        elif isinstance(other, (int, float)):
            return Vector(self.x + other, self.y + other, self.z + other)

    def __sub__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x * other.x, self.y * other.y, self.z * other.z)

        elif isinstance(other, (int, float)):
            return Vector(self.x * other, self.y * other, self.z * other)

    def __rmul__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x * other.x, self.y * other.y, self.z * other.z)

        elif isinstance(other, (int, float)):
            return Vector(self.x * other, self.y * other, self.z * other)

    def __lmul__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x * other.x, self.y * other.y, self.z * other.z)

        elif isinstance(other, (int, float)):
            return Vector(self.x * other, self.y * other, self.z * other)

    def __truediv__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x / other.x, self.y / other.y, self.z / other.z)

        elif isinstance(other, (int, float)):
            return Vector(self.x / other, self.y / other, self.z / other)

    def __getitem__(self, item):
        if item == 0:
            return self.x
        elif item == 1:
            return self.y
        elif item == 2:
            return self.z
        else:
            return IndexError("Index is not valid")

    def __str__(self):
        return f"({self.x}, {self.y}, {self.z})"

    def norm(self):
        """
            Returns the magnitude of the vector
        """
        return np.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def hat(self):
        n = self.norm()
        return Vector(self.x / n, self.y / n, self.z / n)

class Vector2d:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        if isinstance(other, Vector2d):
            return Vector2d(self.x + other.x, self.y + other.y)
    
        elif isinstance(other, (int, float)):
            return Vector2d(self.x + other, self.y + other)

    def __ladd__(self, other):
        if isinstance(other, Vector2d):
            return Vector2d(self.x + other.x, self.y + other.y)
    
        elif isinstance(other, (int, float)):
            return Vector2d(self.x + other, self.y + other)

    def __radd__(self, other):
        if isinstance(other, Vector2d):
            return Vector2d(self.x + other.x, self.y + other.y)
    
        elif isinstance(other, (int, float)):
            return Vector2d(self.x + other, self.y + other)

    def __sub__(self, other):
        if isinstance(other, Vector2d):
            return Vector2d(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        if isinstance(other, Vector2d):
            return Vector2d(self.x * other.x, self.y * other.y)

        elif isinstance(other, (int, float)):
            return Vector2d(self.x * other, self.y * other)

    def __rmul__(self, other):
        if isinstance(other, Vector2d):
            return Vector2d(self.x * other.x, self.y * other.y)

        elif isinstance(other, (int, float)):
            return Vector2d(self.x * other, self.y * other)

    def __lmul__(self, other):
        if isinstance(other, Vector2d):
            return Vector2d(self.x * other.x, self.y * other.y)

        elif isinstance(other, (int, float)):
            return Vector2d(self.x * other, self.y * other)

    def __truediv__(self, other):
        if isinstance(other, Vector2d):
            return Vector2d(self.x / other.x, self.y / other.y)

        elif isinstance(other, (int, float)):
            return Vector2d(self.x / other, self.y / other)

    def __getitem__(self, item):
        if item == 0:
            return self.x
        elif item == 1:
            return self.y
        else:
            return IndexError("Index is not valid")

    def __str__(self):
        return f"({self.x}, {self.y})"

    def norm(self):
        """
            Returns the magnitude of the vector
        """
        return np.sqrt(self.x**2 + self.y**2)

    def hat(self):
        n = self.norm()
        return Vector2d(self.x / n, self.y / n)

