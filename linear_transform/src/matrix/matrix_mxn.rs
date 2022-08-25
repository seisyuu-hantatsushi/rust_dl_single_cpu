use std::{f64,ops,fmt};

/*
| a_11, a_12, ... , a_1n |
| a_21, a_22, ... , a_2n |
|                        |
| a_m1, a_m2, ... , a_mn |
*/
#[derive(Debug, Clone)]
pub struct MatrixMxN {
    col: usize,
    row: usize,
    v: Box<[f64]>
}

impl fmt::Display for MatrixMxN {
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

impl ops::Index<usize> for MatrixMxN {
    type Output = [f64];
    fn index(&self, index:usize) -> &Self::Output {
	let (c,_r) = self.shape();
	assert!(index < c);
	&self.v[index*self.row .. index*self.row+self.row]
    }
}

impl ops::IndexMut<usize> for MatrixMxN {
    fn index_mut(&mut self, index:usize) -> &mut Self::Output {
	let (c,_r) = self.shape();
	assert!(index < c);
	&mut self.v[index*self.row .. index*self.row+self.row]
    }
}

impl ops::Add for MatrixMxN {
    type Output = Self;
    fn add(self, other: Self) -> Self {
	let (c,r) = self.shape();
	let init_vec = vec![0.0f64; (r*c) as usize];
	let mut v = init_vec.into_boxed_slice();
	assert_eq!(self.shape(),other.shape());
	for i in 0..(r*c) {
	    v[i] = self.v[i]+other.v[i];
	}
	MatrixMxN {
	    row: r,
	    col: c,
	    v : v
	}
    }
}

impl ops::Sub for MatrixMxN {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
	let (c,r) = self.shape();
	let init_vec = vec![0.0f64; (r*c) as usize];
	let mut v = init_vec.into_boxed_slice();
	assert_eq!(self.shape(),other.shape());
	for i in 0..(r*c) {
	    v[i] = self.v[i]-other.v[i];
	}
	MatrixMxN {
	    row: r,
	    col: c,
	    v : v
	}
    }
}

impl ops::Mul for MatrixMxN {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
	let (c,r) = self.shape();
	let init_vec = vec![0.0f64; (r*c) as usize];
	let mut v = init_vec.into_boxed_slice();
	assert_eq!((r,c),other.shape());

	for i in 0..c {
	    for j in 0..r {
		for k in 0..r {
		    v[i*r + j] += self[i][k]*other[k][j];
		}
	    }
	}

	MatrixMxN {
	    row: r,
	    col: c,
	    v : v
	}
    }
}


impl MatrixMxN {

    pub fn zero(m:usize, n:usize) -> MatrixMxN {
	let init_vec = vec![0.0f64; (n*m) as usize];
	MatrixMxN {
	    row: n,
	    col: m,
	    v: init_vec.into_boxed_slice()
	}
    }

    pub fn from_u32(m:usize, n:usize, v:&[u32]) -> MatrixMxN {
	assert_eq!(n*m, v.len());
	let init_vec = vec![0.0f64; (n*m) as usize];
	let mut cast_v = init_vec.into_boxed_slice();
	for i in 0..((m*n) as usize) {
	    cast_v[i] = v[i] as f64;
	}
	MatrixMxN {
	    row: n,
	    col: m,
	    v : cast_v
	}
    }

    pub fn from_f64(m:usize, n:usize, v:&[f64]) -> MatrixMxN {
	assert_eq!(n*m, v.len());
	let init_vec = vec![0.0f64; (n*m) as usize];
	let mut boxed_v = init_vec.into_boxed_slice();
	for i in 0..((m*n) as usize) {
	    boxed_v[i] = v[i];
	}
	MatrixMxN {
	    row: n,
	    col: m,
	    v : boxed_v
	}
    }

    pub fn shape(&self) -> (usize, usize) {
	(self.col,self.row)
    }

    pub fn transpose(&self) -> Self {
	let (c,r) = self.shape();
	let init_vec = vec![0.0f64; (r*c) as usize];
	let mut v = init_vec.into_boxed_slice();

	for i in 0 .. c {
	    for j in 0 .. r {
		v[j*c+i] = self.v[i*r+j];
	    }
	}

	MatrixMxN {
	    row: c,
	    col: r,
	    v : v
	}
    }

    pub fn mul(lhs: &MatrixMxN, rhs: &MatrixMxN) -> MatrixMxN {
	let (c,r) = lhs.shape();
	let init_vec = vec![0.0f64; (r*c) as usize];
	let mut v = init_vec.into_boxed_slice();
	assert_eq!((r,c),rhs.shape());

	for i in 0..c {
	    for j in 0..r {
		for k in 0..r {
		    v[i*r + j] += lhs[i][k]*rhs[k][j];
		}
	    }
	}

	MatrixMxN {
	    row: r,
	    col: c,
	    v : v
	}
    }
}
