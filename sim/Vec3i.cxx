#include "Vec3i.hxx"
#include <math.h>

Vec3i::Vec3i() : x(0), y(0), z(0) {}


Vec3i::Vec3i(int _x, int _y, int _z)
: x(_x), y(_y), z(_z) {}


Vec3i Vec3i::zero() {
  return Vec3i(0, 0, 0);
}

Vec3i Vec3i::basis_x() {
  return Vec3i(1, 0, 0);
}

Vec3i Vec3i::basis_y() {
  return Vec3i(0, 1, 0);
}

Vec3i Vec3i::basis_z() {
  return Vec3i(0, 0, 1);
}

Vec3i abs(const Vec3i& p1){
  return Vec3i(abs(p1.x), abs(p1.y), abs(p1.z));
}


// Operators


bool operator==(const Vec3i& p1, const Vec3i& p2) {
  return p1.x == p2.x && p1.y == p2.y && p1.z == p2.z;
}


Vec3i operator+(const Vec3i& p1, const Vec3i& p2) {
  Vec3i ret = {
      p1.x + p2.x,
      p1.y + p2.y,
      p1.z + p2.z,
  };
  return ret;
}


Vec3i operator-(const Vec3i& p1, const Vec3i& p2) {
  Vec3i ret = {
      p1.x - p2.x,
      p1.y - p2.y,
      p1.z - p2.z,
  };
  return ret;
}


Vec3i operator-(const Vec3i& p) {
  return (Vec3i) {
    -p.x,
    -p.y,
    -p.z
  };
}

// dot product
int operator*(const Vec3i& p1, const Vec3i& p2) {
  return (p1.x * p2.x) + (p1.y * p2.y) + (p1.z * p2.z);
}


Vec3i operator*(const int d, const Vec3i& p) {
  Vec3i ret = {
      d * p.x,
      d * p.y,
      d * p.z
  };
  return ret;
}


Vec3i operator*(const Vec3i& p, const int d) {
  return d * p;
}


Vec3i operator/(const Vec3i& p, const int d) {
  return (1/d) * p;
}


int dot(const Vec3i& p1, const Vec3i& p2) {
  return p1 * p2;
}


Vec3i cross(const Vec3i p1, const Vec3i p2) {
  Vec3i ret = {
      (p1.y * p2.z) - (p1.z * p2.y),
      (p1.z * p2.x) - (p1.x * p2.z),
      (p1.x * p2.y) - (p1.y * p2.x)
  };
  return ret;
}


int norm(const Vec3i p) {
  return sqrt((p.x * p.x) + (p.y * p.y) + (p.z * p.z));
}


int norm2(const Vec3i p) {
  return (p.x*p.x) + (p.y*p.y) + (p.z*p.z);
}


Vec3i normalized(const Vec3i p) {
  return p / norm(p);
}


bool approxeq(const Vec3i& p1, const Vec3i& p2) {
  return
    (fabs(p1.x - p2.x) < APPROX) &&
    (fabs(p1.y - p2.y) < APPROX) &&
    (fabs(p1.z - p2.z) < APPROX);
}


std::ostream& operator<<(std::ostream &os, const Vec3i& p) {
  return os << "Vec3i{" << p.x << ", " << p.y << ", " << p.z << "}";
}
