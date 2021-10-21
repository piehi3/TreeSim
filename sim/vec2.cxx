#include "vec2.hxx"


Vec2::Vec2() : x(0), y(0) {}


Vec2::Vec2(int _x, int _y)
: x(_x), y(_y) {}


Vec2 Vec2::zero() {
  return Vec2(0, 0);
}

Vec2 Vec2::basis_x() {
  return Vec2(1, 0);
}

Vec2 Vec2::basis_y() {
  return Vec2(0, 1);
}


// Operators


bool operator==(const Vec2& p1, const Vec2& p2) {
  return p1.x == p2.x && p1.y == p2.y;
}


Vec2 operator+(const Vec2& p1, const Vec2& p2) {
  Vec2 ret = {
      p1.x + p2.x,
      p1.y + p2.y
  };
  return ret;
}


Vec2 operator-(const Vec2& p1, const Vec2& p2) {
  Vec2 ret = {
      p1.x - p2.x,
      p1.y - p2.y
  };
  return ret;
}


Vec2 operator-(const Vec2& p) {
  return (Vec2) {
    -p.x,
    -p.y
  };
}

// dot product
int operator*(const Vec2& p1, const Vec2& p2) {
  return (p1.x * p2.x) + (p1.y * p2.y);
}


Vec2 operator*(const int d, const Vec2& p) {
  Vec2 ret = {
      d * p.x,
      d * p.y
  };
  return ret;
}


Vec2 operator*(const Vec2& p, const int d) {
  return d * p;
}


Vec2 operator/(const Vec2& p, const int d) {
  return (1/d) * p;
}


int dot(const Vec2& p1, const Vec2& p2) {
  return p1 * p2;
}



double norm(const Vec2 p) {
  return sqrt((p.x * p.x) + (p.y * p.y));
}


int norm2(const Vec2 p) {
  return (p.x*p.x) + (p.y*p.y);
}


Vec2 normalized(const Vec2 p) {
  return p / norm(p);
}



std::ostream& operator<<(std::ostream &os, const Vec2& p) {
  return os << "Vec2{" << p.x << ", " << p.y << "}";
}
