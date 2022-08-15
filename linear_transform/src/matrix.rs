
use std::{f64,ops,fmt};

#[derive(Debug,Copy,Clone, PartialEq)]
pub struct Matrix4x4 {
    pub v : [[f64; 4]; 4]
}

#[allow(dead_code)]
impl fmt::Display for Matrix4x4 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
	let mut str = "Matrix4x4:\n".to_string();
	for i in 0..4 {
	    let v_str = format!("[{},{},{},{}]\n",
				self.v[i][0],self.v[i][1],self.v[i][2],self.v[i][3]);
	    str.push_str(&v_str);
	}
	write!(f,"{}", str)
    }
}

impl ops::Add for Matrix4x4 {
    type Output = Self;
    fn add(self, other: Self) -> Self {
	let mut m = Self::identity();
	for i in 0..4 {
	    for j in 0..4 {
		m.v[i][j] = self.v[i][j]+other.v[i][j]
	    }
	}
	m
    }
}

impl ops::Sub for Matrix4x4 {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
	let mut m = Self::identity();
	for i in 0..4 {
	    for j in 0..4 {
		m.v[i][j] = self.v[i][j]-other.v[i][j]
	    }
	}
	m
    }
}

impl ops::Index<usize> for Matrix4x4 {
    type Output = [f64; 4];
    fn index(&self, index:usize) -> &Self::Output {
	&self.v[index]
    }
}

fn matrix4x4_entry(l:Matrix4x4, r:Matrix4x4, i:usize, j:usize) -> f64 {
    let v:[f64; 4] = [ l[i][0]*r[0][j], l[i][1]*r[1][j], l[i][2]*r[2][j], l[i][3]*r[3][j] ];
    v.iter().fold(0.0, |acc, e| acc+e)
}

impl ops::Mul for Matrix4x4 {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
	Matrix4x4 {
	    v :
	    [
		[matrix4x4_entry(self, other, 0, 0), matrix4x4_entry(self, other, 0, 1), matrix4x4_entry(self, other, 0, 2), matrix4x4_entry(self, other, 0, 3)],
		[matrix4x4_entry(self, other, 1, 0), matrix4x4_entry(self, other, 1, 1), matrix4x4_entry(self, other, 1, 2), matrix4x4_entry(self, other, 1, 3)],
		[matrix4x4_entry(self, other, 2, 0), matrix4x4_entry(self, other, 2, 1), matrix4x4_entry(self, other, 2, 2), matrix4x4_entry(self, other, 2, 3)],
		[matrix4x4_entry(self, other, 3, 0), matrix4x4_entry(self, other, 3, 1), matrix4x4_entry(self, other, 3, 2), matrix4x4_entry(self, other, 3, 3)]
	    ]
	}
    }
}

/* OpenGL側に渡すときは転置する必要がある !! */
impl Matrix4x4 {

    pub fn zero() -> Matrix4x4 {
	Matrix4x4 {
	    v :
	    [
		[0.0, 0.0, 0.0, 0.0],
		[0.0, 0.0, 0.0, 0.0],
		[0.0, 0.0, 0.0, 0.0],
		[0.0, 0.0, 0.0, 0.0]
	    ]
	}
    }

    /* 単位行列 */
    pub fn identity() -> Matrix4x4 {
	Matrix4x4 {
	    v :
	    [
		[1.0, 0.0, 0.0, 0.0],
		[0.0, 1.0, 0.0, 0.0],
		[0.0, 0.0, 1.0, 0.0],
		[0.0, 0.0, 0.0, 1.0]
	    ]
	}
    }

    /* 転置行列 */
    pub fn transport(self) -> Matrix4x4 {
	Matrix4x4 {
	    v :
	    [
		[self.v[0][0], self.v[1][0], self.v[2][0], self.v[3][0] ],
		[self.v[0][1], self.v[1][1], self.v[2][1], self.v[3][1] ],
		[self.v[0][2], self.v[1][2], self.v[2][2], self.v[3][2] ],
		[self.v[0][3], self.v[1][3], self.v[2][3], self.v[3][3] ]
	    ]
	}
    }

    /* シリアライズ */
    pub fn serialize(self) -> [f64; 16] {
	[
	    self.v[0][0], self.v[0][1], self.v[0][2], self.v[0][3],
	    self.v[1][0], self.v[1][1], self.v[1][2], self.v[1][3],
	    self.v[2][0], self.v[2][1], self.v[2][2], self.v[2][3],
	    self.v[3][0], self.v[3][1], self.v[3][2], self.v[3][3]
	]
    }

    pub fn serialize_f32(self) -> [f32; 16] {
	[
	    self.v[0][0] as f32, self.v[0][1] as f32, self.v[0][2] as f32, self.v[0][3] as f32,
	    self.v[1][0] as f32, self.v[1][1] as f32, self.v[1][2] as f32, self.v[1][3] as f32,
	    self.v[2][0] as f32, self.v[2][1] as f32, self.v[2][2] as f32, self.v[2][3] as f32,
	    self.v[3][0] as f32, self.v[3][1] as f32, self.v[3][2] as f32, self.v[3][3] as f32
	]
    }
}

#[derive(Debug,Copy,Clone, PartialEq)]
pub struct Matrix3x3 {
    pub v : [[f64; 3]; 3]
}

#[allow(dead_code)]
impl fmt::Display for Matrix3x3 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
	let mut str = "Matrix3x3:\n".to_string();
	for i in 0..3 {
	    let v_str = format!("[{},{},{}]\n",
				self.v[i][0],self.v[i][1],self.v[i][2]);
	    str.push_str(&v_str);
	}
	write!(f,"{}", str)
    }
}

impl Matrix3x3 {

    pub fn zero() -> Self {
	Matrix3x3 {
	    v :
	    [
		[0.0, 0.0, 0.0],
		[0.0, 0.0, 0.0],
		[0.0, 0.0, 0.0]
	    ]
	}
    }

    /* 単位行列 */
    pub fn identity() -> Matrix3x3 {
	Matrix3x3 {
	    v :
	    [
		[1.0, 0.0, 0.0],
		[0.0, 1.0, 0.0],
		[0.0, 0.0, 1.0]
	    ]
	}
    }

    /* 転置行列 */
    pub fn transport(self) -> Self {
	Matrix3x3 {
	    v :
	    [
		[self.v[0][0], self.v[1][0], self.v[2][0] ],
		[self.v[0][1], self.v[1][1], self.v[2][1] ],
		[self.v[0][2], self.v[1][2], self.v[2][2] ],
	    ]
	}
    }

    /* シリアライズ */
    pub fn serialize(self) -> [f64; 9] {
	[
	    self.v[0][0], self.v[0][1], self.v[0][2],
	    self.v[1][0], self.v[1][1], self.v[1][2],
	    self.v[2][0], self.v[2][1], self.v[2][2],
	]
    }

    pub fn serialize_f32(self) -> [f32; 9] {
	[
	    self.v[0][0] as f32, self.v[0][1] as f32, self.v[0][2] as f32,
	    self.v[1][0] as f32, self.v[1][1] as f32, self.v[1][2] as f32,
	    self.v[2][0] as f32, self.v[2][1] as f32, self.v[2][2] as f32
	]
    }

}

impl ops::Add for Matrix3x3 {
    type Output = Self;
    fn add(self, other: Self) -> Self {
	let mut m = Self::identity();
	for i in 0..3 {
	    for j in 0..3 {
		m.v[i][j] = self.v[i][j]+other.v[i][j]
	    }
	}
	m
    }
}

impl ops::Sub for Matrix3x3 {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
	let mut m = Self::identity();
	for i in 0..3 {
	    for j in 0..3 {
		m.v[i][j] = self.v[i][j]-other.v[i][j]
	    }
	}
	m
    }
}

impl ops::Index<usize> for Matrix3x3 {
    type Output = [f64; 3];
    fn index(&self, index:usize) -> &Self::Output {
	&self.v[index]
    }
}

fn matrix3x3_entry(l:Matrix3x3, r:Matrix3x3, i:usize, j:usize) -> f64 {
    let v:[f64; 3] = [ l[i][0]*r[0][j], l[i][1]*r[1][j], l[i][2]*r[2][j] ];
    v.iter().fold(0.0, |acc, e| acc+e)
}

impl ops::Mul for Matrix3x3 {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
	Matrix3x3 {
	    v :
	    [
		[matrix3x3_entry(self, other, 0, 0), matrix3x3_entry(self, other, 0, 1), matrix3x3_entry(self, other, 0, 2)],
		[matrix3x3_entry(self, other, 1, 0), matrix3x3_entry(self, other, 1, 1), matrix3x3_entry(self, other, 1, 2)],
		[matrix3x3_entry(self, other, 2, 0), matrix3x3_entry(self, other, 2, 1), matrix3x3_entry(self, other, 2, 2)]
	    ]
	}
    }
}
