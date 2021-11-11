#ifndef __VEC3I
#define __VEC3I

#include <ostream>

#define APPROX 1e-9

// represents positions as well (for convinience, even though not all operations make sense)
typedef struct Vec3i {
  Vec3i();
  Vec3i(int _x, int _y, int _z);

  int x;
  int y;
  int z;

  static Vec3i zero();
  static Vec3i basis_x();
  static Vec3i basis_y();
  static Vec3i basis_z();

  template <typename T>
  static Vec3i basis(T i) {
    //tatic_assert(std::is_integral<T>::value, "Cannot index with a nonintegral type.");
    switch (i) {
      case 0:
        return basis_x();
      case 1:
        return basis_y();
      case 2:
        return basis_z();
      default:
        throw "Invalid index.";
    }
  }

  template <typename T>
  static Vec3i ortho(T i, T j) {
    //static_assert(std::is_integral<T>::value, "Cannot index with a nonintegral type.");
    if (i == j) throw "Multiple cases.";
    switch (i) {
      case 0:
        switch (j) {
          case 1:
            return basis_z();
          case 2:
            return basis_y();
          default:
            throw "Invalid index.";
        }
      case 1:
        switch (j) {
          case 0:
            return basis_z();
          case 1:
            return basis_x();
          default:
            throw "Invalid index.";
        }
      case 2:
        switch (j) {
          case 0:
            return basis_y();
          case 1:
            return basis_x();
          default:
            throw "Invalid index.";
        }
      default:
        throw "Invalid index.";
    }
  }

  template <typename T>
  int operator[](T i) const {
    //static_assert(std::is_integral<T>::value, "Cannot index with a nonintegral type.");
    switch (i) {
      case 0:
        return this->x;
      case 1:
        return this->y;
      case 2:
        return this->z;
      default:
        throw "Invalid index.";
    }
  }
} Vec3i;


bool operator==(const Vec3i& p1, const Vec3i& p2);
Vec3i operator+(const Vec3i& p1, const Vec3i& p2);
Vec3i operator-(const Vec3i& p1, const Vec3i& p2);

int operator*(const Vec3i& p1, const Vec3i& p2);
// both dot and cross are defined because '*' for Vec3i is the dot product,
//   while for Quaternion it's the Hamilton product. This just makes it a
//   little clearer what's happening.
int dot(const Vec3i& p1, const Vec3i& p2);
Vec3i   cross(const Vec3i p1, const Vec3i p2);
Vec3i abs(const Vec3i& p1);

Vec3i operator-(const Vec3i& p);
Vec3i operator*(const int d, const Vec3i& p);
Vec3i operator*(const Vec3i& p, const int d);
Vec3i operator/(const Vec3i& p, const int d);

int norm(const Vec3i p);
int norm2(const Vec3i p);  // norm squared
Vec3i   normalized(const Vec3i p);

bool approxeq(const Vec3i& p1, const Vec3i& p2);

std::ostream &operator<<(std::ostream& os, const Vec3i& p);


#endif
