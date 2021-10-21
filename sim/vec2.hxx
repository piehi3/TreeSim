#ifndef __VEC2
#define __VEC2

// represents positions as well (for convinience, even though not all operations make sense)
typedef struct Vec2 {
  Vec2();
  Vec2(int _x, int _y);

  int x;
  int y;

  static Vec2 zero();
  static Vec2 basis_x();
  static Vec2 basis_y();

  template <typename T>
  static Vec2 basis(T i) {
    static_assert(std::is_integral<T>::value, "Cannot index with a nonintegral type.");
    switch (i) {
      case 0:
        return basis_x();
      case 1:
        return basis_y();
      default:
        throw "Invalid index.";
    }
  }


  template <typename T>
  double operator[](T i) const {
    static_assert(std::is_integral<T>::value, "Cannot index with a nonintegral type.");
    switch (i) {
      case 0:
        return this->x;
      case 1:
        return this->y;
      default:
        throw "Invalid index.";
    }
  }
} Vec2;


bool operator==(const Vec2& p1, const Vec2& p2);
Vec2 operator+(const Vec2& p1, const Vec2& p2);
Vec2 operator-(const Vec2& p1, const Vec2& p2);

int operator*(const Vec2& p1, const Vec2& p2);
// both dot and cross are defined because '*' for Vec2 is the dot product,
//   while for Quaternion it's the Hamilton product. This just makes it a
//   little clearer what's happening.
int dot(const Vec2& p1, const Vec2& p2);

Vec2 operator-(const Vec2& p);
Vec2 operator*(const int d, const Vec2& p);
Vec2 operator*(const Vec2& p, const int d);
Vec2 operator/(const Vec2& p, const int d);

int norm(const Vec2 p);
double norm2(const Vec2 p);  // norm squared
Vec2   normalized(const Vec2 p);

std::ostream &operator<<(std::ostream& os, const Vec2& p);


#endif
