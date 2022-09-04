use std::{ops,fmt};
use num;

/*
| a_11, a_12, ... , a_1n |
| a_21, a_22, ... , a_2n |
|                        |
| a_m1, a_m2, ... , a_mn |
*/
#[derive(Debug, Clone)]
pub struct MatrixMxN<T> where T: Clone {
    col: usize,
    row: usize,
    v: Box<[T]>
}

#[cfg(dead_code)]
fn type_of<T>(_:T) -> String {
    std::any::type_name::<T>().to_string()
}

impl<T: std::fmt::Display + Clone> fmt::Display for MatrixMxN<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
	let mut str = format!("Matrix ({},{})\n", self.col, self.row);
	str.push_str("[");
	for c in 0..self.col {
	    str.push_str("[");
	    for r in 0..self.row {
		let v_str = format!("{},",self.v[c*self.row+r]);
		str.push_str(&v_str)
	    }
	    str.push_str("]\n");
	}
	str.push_str("]");
	write!(f, "{}", str)
    }
}

impl<T: num::Num+Clone> ops::Index<usize> for MatrixMxN<T> {
    type Output = [T];
    fn index(&self, index:usize) -> &Self::Output {
	let (c,_r) = self.shape();
	assert!(index < c);
	&self.v[index*self.row .. index*self.row+self.row]
    }
}

impl<T: num::Num+Clone> ops::IndexMut<usize> for MatrixMxN<T> {
    fn index_mut(&mut self, index:usize) -> &mut Self::Output {
	let (c,_r) = self.shape();
	assert!(index < c);
	&mut self.v[index*self.row .. index*self.row+self.row]
    }
}

impl<T: num::Num + Copy> ops::Add for MatrixMxN<T> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
	assert_eq!(self.shape(),other.shape());
	let (c,r) = self.shape();
	let mut v:Vec<T> = Vec::with_capacity(r*c);
	unsafe { v.set_len(r*c) };
	for i in 0..(r*c) {
	    v[i] = self.v[i]+other.v[i];
	}
	MatrixMxN {
	    row: r,
	    col: c,
	    v : v.into_boxed_slice()
	}
    }
}

impl<T: num::Num + Copy> ops::Add for &MatrixMxN<T> {
    type Output = MatrixMxN<T>;
    fn add(self, other: Self) -> MatrixMxN<T> {
	assert_eq!(self.shape(),other.shape());
	let (c,r) = self.shape();
	let mut v:Vec<T> = Vec::with_capacity(r*c);
	unsafe { v.set_len(r*c) };
	for i in 0..(r*c) {
	    v[i] = self.v[i]+other.v[i];
	}
	MatrixMxN {
	    row: r,
	    col: c,
	    v : v.into_boxed_slice()
	}
    }
}

impl<T: num::Num + Copy> ops::Add<&MatrixMxN<T>> for MatrixMxN<T> {
    type Output = MatrixMxN<T>;
    fn add(self, other: &Self) -> MatrixMxN<T> {
	assert_eq!(self.shape(),other.shape());
	let (c,r) = self.shape();
	let mut v:Vec<T> = Vec::with_capacity(r*c);
	unsafe { v.set_len(r*c) };
	for i in 0..(r*c) {
	    v[i] = self.v[i]+other.v[i];
	}
	MatrixMxN {
	    row: r,
	    col: c,
	    v : v.into_boxed_slice()
	}
    }
}

impl<T: num::Num + Copy> ops::Sub for MatrixMxN<T> {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
	assert_eq!(self.shape(),other.shape());
	let (c,r) = self.shape();
	let mut v:Vec<T> = Vec::with_capacity(r*c);
	unsafe { v.set_len(r*c) };
	for i in 0..(r*c) {
	    v[i] = self.v[i]-other.v[i];
	}
	MatrixMxN {
	    row: r,
	    col: c,
	    v : v.into_boxed_slice()
	}
    }
}

impl<T: num::Num + Copy + ops::AddAssign> ops::Mul for MatrixMxN<T> {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
	let (c1,r1) = self.shape();
	let (c2,r2) = other.shape();
	assert_eq!(r1,c2);

	let mut v:Vec<T> = Vec::with_capacity(r2*c1);
	unsafe { v.set_len(r2*c1) };

	for i in 0..c1 {
	    for j in 0..r2 {
		v[i*r2 + j].set_zero();
		for k in 0..r1 {
		    v[i*r2 + j] += self[i][k]*other[k][j];
		}
	    }
	}

	MatrixMxN {
	    row: r2,
	    col: c1,
	    v : v.into_boxed_slice()
	}
    }
}

impl<T: num::Num + Copy + ops::AddAssign> ops::Mul for &MatrixMxN<T> {
    type Output = MatrixMxN<T>;
    fn mul(self, other: Self) -> MatrixMxN<T> {
	let (c1,r1) = self.shape();
	let (c2,r2) = other.shape();
	assert_eq!(r1,c2);

	let mut v:Vec<T> = Vec::with_capacity(r2*c1);
	unsafe { v.set_len(r2*c1) };

	for i in 0..c1 {
	    for j in 0..r2 {
		v[i*r2 + j].set_zero();
		for k in 0..r1 {
		    v[i*r2 + j] += self[i][k]*other[k][j];
		}
	    }
	}

	MatrixMxN {
	    row: r2,
	    col: c1,
	    v : v.into_boxed_slice()
	}
    }
}

impl<T: num::Num + Copy + ops::AddAssign> ops::Mul<&MatrixMxN<T>> for MatrixMxN<T> {
    type Output = MatrixMxN<T>;
    fn mul(self, other: &Self) -> MatrixMxN<T> {
	let (c1,r1) = self.shape();
	let (c2,r2) = other.shape();
	assert_eq!(r1,c2);

	let mut v:Vec<T> = Vec::with_capacity(r2*c1);
	unsafe { v.set_len(r2*c1) };

	for i in 0..c1 {
	    for j in 0..r2 {
		v[i*r2 + j].set_zero();
		for k in 0..r1 {
		    v[i*r2 + j] += self[i][k]*other[k][j];
		}
	    }
	}

	MatrixMxN {
	    row: r2,
	    col: c1,
	    v : v.into_boxed_slice()
	}
    }
}


impl ops::Mul<MatrixMxN<f32>> for f32 {
    type Output = MatrixMxN<f32>;
    fn mul(self, other:MatrixMxN<f32>) -> Self::Output {
	let (c,r) = other.shape();
	let v:Vec<f32> = other.buffer().into_iter().map(|x| x*self).collect::<Vec<f32>>();
	MatrixMxN {
	    row: r,
	    col: c,
	    v : v.into_boxed_slice()
	}
    }
}

impl<T: num::Num + Copy + ops::AddAssign + ops::MulAssign> ops::Div<T> for MatrixMxN<T> {
    type Output = Self;
    fn div(self, other:T) -> Self {
	let (c,r) = self.shape();
	let v:Vec<T> = self.buffer().into_iter().map(|x| *x/other).collect::<Vec<T>>();
	MatrixMxN {
	    row: r,
	    col: c,
	    v : v.into_boxed_slice()
	}
    }
}

impl<T: PartialEq+Clone> PartialEq for MatrixMxN<T> {
    fn eq(&self, other: &Self) -> bool {
	self.row == other.row && self.col == other.col && self.v == other.v
    }
}

impl<T: PartialEq+Clone> Eq for MatrixMxN<T> { }

pub fn identity<T: Clone>(m: MatrixMxN<T>) -> MatrixMxN<T> {
    m.clone()
}

impl<T:Clone> MatrixMxN<T> {
    pub fn shape(&self) -> (usize, usize) {
	(self.col,self.row)
    }
}

impl<T: num::Num+ops::AddAssign+ops::MulAssign+Clone+Copy> MatrixMxN<T> {

    pub fn zero(m:usize, n:usize) -> MatrixMxN<T> {
	let mut v:Vec<T> = Vec::with_capacity(n*m);
	unsafe { v.set_len(n*m) };

	for i in 0..(n*m) {
	    v[i].set_zero();
	}

	MatrixMxN {
	    row: n,
	    col: m,
	    v: v.into_boxed_slice()
	}
    }

    pub fn from_array(m:usize, n:usize, v:&[T]) -> MatrixMxN<T> {
	assert_eq!(n*m, v.len());
	let mut bv:Vec<T> = Vec::with_capacity(n*m);

	for x in v.iter() {
	    bv.push(*x)
	}

	MatrixMxN {
	    row: n,
	    col: m,
	    v : bv.into_boxed_slice()
	}
    }

    pub fn from_vector(v:Vec<Vec<T>>) -> MatrixMxN<T> {
	let mut bv:Vec<T> = Vec::new();
	let r = v[0].len();
	let c = v.len();

	for vr in v.iter() {
	    for x in vr.iter() {
		bv.push(*x);
	    }
	}

	MatrixMxN {
	    row: r,
	    col: c,
	    v : bv.into_boxed_slice()
	}

    }

    pub fn into_vector(&self) -> Vec<T> {
	self.v.clone().into_vec()
    }


    pub fn buffer(&self) -> &[T] {
	&self.v
    }

    pub fn transpose(&self) -> Self {
	let (c,r) = self.shape();
	let mut v:Vec<T> = Vec::with_capacity(c*r);
	unsafe { v.set_len(c*r) };

	for i in 0 .. c {
	    for j in 0 .. r {
		v[j*c+i] = self.v[i*r+j];
	    }
	}

	MatrixMxN {
	    row: c,
	    col: r,
	    v : v.into_boxed_slice()
	}
    }

    pub fn mul(lhs: &MatrixMxN<T>, rhs: &MatrixMxN<T>) -> MatrixMxN<T> {
	let (lc,lr) = lhs.shape();
	let (rc,rr) = rhs.shape();
	let mut v:Vec<T> = Vec::with_capacity(rr*lc);
	unsafe { v.set_len(rr*lc) };

	assert_eq!(lr, rc);

	for i in 0..lc {
	    for j in 0..rr {
		v[i*rr + j].set_zero();
		for k in 0..lr {
		    v[i*rr + j] += lhs[i][k]*rhs[k][j];
		}
	    }
	}

	MatrixMxN {
	    row: rr,
	    col: lc,
	    v : v.into_boxed_slice()
	}
    }

    pub fn scale(&self, s:T) -> Self {
	let (c,r) = self.shape();
	let b:Vec<T> = self.buffer().iter().map(|v| s*(*v)).collect();
	MatrixMxN::from_array(c, r, &b.into_boxed_slice())
    }
}

impl <T1: std::cmp::PartialOrd+Clone+Copy> MatrixMxN<T1> {
    pub fn max_element_index(&self) -> (usize, usize) {
	let (c,r) = self.shape();
	let mut max_index = (0,0);
	
	for i in 0 .. c {
	    for j in 0 .. r {
		let max_element = self.v[max_index.0*r+max_index.1];
		if max_element < self.v[i*r+j] {
		    max_index = (i, j);
		}
	    }
	}
	return max_index;
    }
}
