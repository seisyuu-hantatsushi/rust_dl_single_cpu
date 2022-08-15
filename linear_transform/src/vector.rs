
use std::{f64,ops,fmt};

#[derive(Debug,Copy,Clone,PartialEq)]
pub struct Vector2 (pub f64, pub f64);

impl ops::Add for Vector2 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
	Self(self.0+other.0, self.1+other.1)
    }
}

impl ops::Sub for Vector2 {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
	Self(self.0-other.0, self.1-other.1)
    }
}

/* dot product for Vector2 */
impl ops::Mul for Vector2 {
    type Output = f64;
    fn mul(self, other: Self) -> f64 {
	self.0*other.0+self.1*other.1
    }
}

impl ops::Div<f64> for Vector2 {
    type Output = Self;
    fn div(self, other: f64) -> Self {
	Self(self.0/other,self.1/other)
    }
}

impl Vector2 {
    #[allow(dead_code)]
    pub fn square(self) -> f64 {
	self * self
    }

    /* length of Vector2 */
    #[allow(dead_code)]
    pub fn length(self) -> f64 {
	self.square().sqrt()
    }
}

#[derive(Debug,Copy,Clone,PartialEq)]
pub struct Vector3 (pub f64, pub f64, pub f64);

#[allow(dead_code)]
impl fmt::Display for Vector3 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
	write!(f,"Vector3:[{},{},{}]", self.0, self.1, self.2)
    }
}

impl ops::Index<usize> for Vector3 {
    type Output = f64;
    fn index(&self, index:usize) -> &Self::Output {
	match index {
	    0 => &self.0,
	    1 => &self.1,
	    2 => &self.2,
	    _ => panic!("out of index")
	}
    }
}

impl ops::Add for Vector3 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
	Self(self.0+other.0, self.1+other.1, self.2+other.2)
    }
}

impl ops::Sub for Vector3 {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
	Self(self.0-other.0, self.1-other.1, self.2-other.2)
    }
}

/* dot product for Vector3 */
impl ops::Mul for Vector3 {
    type Output = f64;
    fn mul(self, other: Self) -> f64 {
	self.0*other.0+self.1*other.1+self.2*other.2
    }
}

impl ops::Mul<Vector3> for f64 {
    type Output = Vector3;
    fn mul(self, other: Vector3) -> Vector3 {
	Vector3(self*other.0, self*other.1, self*other.2)
    }
}

impl ops::Div<f64> for Vector3 {
    type Output = Vector3;
    fn div(self, other:f64) -> Vector3 {
	Vector3(self.0/other, self.1/other, self.2/other)
    }
}

impl ops::Neg for Vector3 {
    type Output = Self;
    fn neg(self) -> Self {
	Self(-self.0, -self.1, -self.2)
    }
}

impl Vector3 {
    #[allow(dead_code)]
    pub fn make_from_tuple(v:(f64,f64,f64)) -> Self {
	Self(v.0, v.1, v.2)
    }

    #[allow(dead_code)]
    pub fn square(self) -> f64 {
	self * self
    }

    /* length of Vector3 */
    #[allow(dead_code)]
    pub fn length(self) -> f64 {
	self.square().sqrt()
    }

    #[allow(dead_code)]
    pub fn normalize(self) -> Vector3 {
	self/self.length()
    }

    /* Cross Product of Vector3 */
    pub fn cross_product(lhs: Self, other: Self) -> Self {
	let x = lhs.1 * other.2 - lhs.2 * other.1;
	let y = lhs.2 * other.0 - lhs.0 * other.2;
	let z = lhs.0 * other.1 - lhs.1 * other.0;
	Self(x, y, z)
    }

    /* シリアライズ */
    pub fn serialize(self) -> [f64; 3] {
	[self.0, self.1, self.2]
    }

    pub fn serialize_f32(self) -> [f32; 3] {
	[self.0 as f32, self.1 as f32, self.2 as f32]
    }
}

impl ops::BitAnd for Vector3 {
    type Output = Vector3;
    fn bitand(self, rhs: Self) -> Self::Output {
	Vector3::cross_product(self, rhs)
    }
}

#[derive(Debug,Copy,Clone,PartialEq)]
pub struct Vector4 (pub f64, pub f64, pub f64, pub f64);

impl Vector4 {
    #[allow(dead_code)]
    pub fn make_from_tuple(v:(f64,f64,f64,f64)) -> Self {
	Self(v.0, v.1, v.2, v.3)
    }

    /* シリアライズ */
    pub fn serialize(self) -> [f64; 4] {
	[self.0, self.1, self.2, self.3]
    }

    pub fn serialize_f32(self) -> [f32; 4] {
	[self.0 as f32, self.1 as f32, self.2 as f32, self.3 as f32]
    }
}
