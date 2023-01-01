/* -*- tab-width:4 -*- */

use std::{ops,fmt};
use num;

/*
| a_11, a_12, ... , a_1n |
| a_21, a_22, ... , a_2n |
|                        |
| a_m1, a_m2, ... , a_mn |
 */

/*
  Tensor characxstoristic.
  Tensor is mapping function.
  T: C^n -> C^n, Tensor Rank is n+1.
  T(v) = T_1(v1) + T_2(v2) -> T = T_1 + T_2
  T(v) = kT'(v1)  -> T = kT'. T,T' same Rank Tensor. k is scaler.
*/

#[derive(Debug, Clone)]
pub struct Tensor<T> where T: Clone {
    shape: Vec<usize>,
    v: Box<[T]>
}

#[derive(Debug, Clone)]
pub struct SubTensor<'a, T> where T:Clone {
    shape: Vec<usize>,
    v: &'a [T]
}

fn fmt_recursive<'a, T>(depth:usize, shape:&'a[usize], v:&'a[T]) -> String
where T:std::fmt::Display {
    let indent_element = "    ".to_string();
    let indent = if depth == 0 { "".to_string() } else { (0..depth).fold("".to_string(),|s, _| s + &indent_element) };
    if shape.len() > 1 {
	let stride = shape[1..shape.len()].iter().fold(1,|prod, x| prod * (*x));
		let mut l = indent.clone() + "[\n";
		for i in 0..shape[0] {
			l = l + &fmt_recursive(depth+1, &shape[1..shape.len()], &v[stride*i..(stride*(i+1))]) + "\n";
		}
		l+&indent+"],"
    }
    else {
		indent + "[" + &v.iter().fold("".to_string(),|s, x| s + &x.to_string()+",").clone() + "]"
    }
}

impl<T> fmt::Display for Tensor<T>
where T: fmt::Display + Clone {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		let shape = self.shape();
		let mut disp = format!("Tensor [");
		for s in shape {
			disp = format!("{}{},", disp, s);
		}
		disp = format!("{}]\n",disp);
		disp = format!("{}{}",disp,fmt_recursive(0, &shape[0..shape.len()], &self.v));
		write!(f, "{}", disp)
    }
}

impl<T:Clone> Tensor<T> {

    pub fn shape(&self) -> &[usize] {
		self.shape.as_slice()
    }

    pub fn num_of_elements(&self) -> usize {
		self.shape.iter().fold(1,|prod, &n| prod*n)
    }

    fn index_to_position_inner(index:usize, shape:&[usize]) -> Vec<usize> {
		if shape.len() > 0 {
			let blocksize = shape[1..].iter().fold(1,|prod, &n| prod*n);
			let p = index / blocksize;
			let v = Self::index_to_position_inner(index % blocksize, &shape[1..]);
			let mut r = vec![p];
			r.extend(v);
			r
		}
		else {
			vec!()
		}
    }

    pub fn index_to_position(&self, index:usize) -> Vec<usize> {
		if index < self.num_of_elements() {
			Self::index_to_position_inner(index, self.shape())
		}
		else {
			vec!()
		}
    }
}

impl<T:Clone> Tensor<T> {
    pub fn buffer(&self) -> &[T] {
		&self.v
    }
}

impl<T: PartialEq+Clone> PartialEq for Tensor<T> {
    fn eq(&self, other: &Self) -> bool {
		self.shape() == other.shape() && self.v == other.v
    }
}

impl<T: PartialEq+Clone> Eq for Tensor<T> { }

fn element<'a, T>(index:&[usize], shape:&'a [usize], v:&'a [T]) -> &'a T {
    if index.len() > 1 {
		let down_shape = &shape[1..];
		let stride = down_shape.iter().fold(1,|prod,x| prod * (*x));
		let i = index[0];
		element(&index[1..], &down_shape, &v[(stride*i)..(stride*(i+1))])
}
    else {
		&v[index[0]]
    }
}

fn element_mut<'a, T>(index:&[usize], shape:&'a [usize], v:&'a mut [T]) -> &'a mut T {
    if index.len() > 1 {
		let down_shape = &shape[1..];
		let stride = down_shape.iter().fold(1,|prod,x| prod * (*x));
		let i = index[0];
		element_mut(&index[1..], &down_shape, &mut v[(stride*i)..(stride*(i+1))])
    }
    else {
		&mut v[index[0]]
    }
}

impl<T> Tensor<T>
where T:num::Num+Clone+Copy {

    pub fn zero(shape:&[usize]) -> Tensor<T> {
		let size = shape.iter().fold(1,|prod,&x| prod*x);
		let mut v:Vec<T> = Vec::with_capacity(size);
		unsafe { v.set_len(size) };
		for i in 0..size {
			v[i].set_zero();
		}
		Tensor {
			shape: shape.to_vec(),
			v: v.into_boxed_slice(),
		}
    }

    pub fn one(shape:&[usize]) -> Tensor<T> {
		let size = shape.iter().fold(1,|prod,&x| prod*x);
		let mut v:Vec<T> = Vec::with_capacity(size);
		unsafe { v.set_len(size) };
		for i in 0..size {
			v[i].set_one();
		}
		Tensor {
			shape: shape.to_vec(),
			v: v.into_boxed_slice(),
		}
    }

    pub fn new_set_value(shape:&[usize], x:T) -> Tensor<T> {
		let size = shape.iter().fold(1,|prod,&x| prod*x);
		let mut v:Vec<T> = Vec::with_capacity(size);
		for _ in 0..size {
			v.push(x)
		}
		Tensor {
			shape: shape.to_vec(),
			v: v.into_boxed_slice()
		}
    }

    pub fn from_array<'a>(shape:&'a [usize], elements:&'a [T]) -> Tensor<T> {
		let product = shape.iter().fold(1,|prod, x| prod * (*x));
		assert_eq!(product, elements.len());
		Tensor {
			shape: shape.to_vec(),
			v: elements.to_owned().into_boxed_slice()
		}
    }

    pub fn from_vector<'a>(shape:Vec<usize>, elements:Vec<T>) -> Tensor<T> {
		let product = shape.iter().fold(1,|prod, &x| prod * x);
		assert_eq!(product, elements.len());
		Tensor {
			shape: shape,
			v: elements.into_boxed_slice()
		}
    }

	pub fn set(&mut self, index: &[usize], value:T) -> () {
		assert!(index.len() == self.shape.len());
		let e = element_mut(index, &self.shape, &mut self.v);
		*e = value;
	}

	pub fn get(&mut self, index: &[usize]) -> T {
		assert!(index.len() == self.shape.len());
		*element(index, &self.shape, &mut self.v)
	}

}

impl<T> Tensor<T>
where T:Clone {
    pub fn sub_tensor<'a>(&'a self, index:usize) -> SubTensor<'a, T> {
	assert!(self.shape.len() > 0);
	assert!(index < self.shape[0]);
		let down_shape = self.shape[1..self.shape.len()].to_vec();
		let stride = down_shape.iter().fold(1, |prod, x| prod * (*x));
		SubTensor {
			shape: down_shape,
			v: &self.v[(stride*index)..(stride*(index+1))]
		}
    }
}

impl<T> ops::Index<Vec<usize>> for Tensor<T>
where T:Clone {
    type Output = T;
    fn index(&self, index:Vec<usize>) -> &Self::Output {
		assert!(index.len() == self.shape.len());
		element(&index,&self.shape, &self.v)
    }
}

impl<T> ops::IndexMut<Vec<usize>> for Tensor<T>
where T:Clone {
    fn index_mut(&mut self, index:Vec<usize>) -> &mut T {
		assert!(index.len() == self.shape.len());
		element_mut(&index, &self.shape, &mut self.v)
    }
}

impl <T> Tensor<T>
where T: num::Num+std::cmp::PartialOrd+Clone+Copy+fmt::Debug {
    fn search_max_element(shape:Vec<usize>, v:&[T]) -> (Vec<usize>,T) {
		if shape.len() > 1 {
			let mut holder:Vec<(Vec<usize>, T)> = Vec::new();
	    for n in 0..shape[0] {
			let stride = shape[1..shape.len()].iter().fold(1,|prod,s| prod*(*s));
			let result = Self::search_max_element(shape[1..shape.len()].to_vec(),&v[n*stride..(n+1)*stride]);
			holder.push(result);
	    }
			let mut max_pair:(Vec<usize>, T) = holder[0].clone();
			let mut max_index = 0;
			for n in 0..holder.len() {
				if max_pair.1 < holder[n].1 {
					max_pair = holder[n].clone();
					max_index = n;
				}
			}
			let mut index:Vec<usize> = vec![max_index];
			index.append(&mut max_pair.0);
			(index, max_pair.1)
		}
		else {
			let mut max = v[0];
			let mut max_index = 0;
			for i in 0..shape[0] {
				if v[i] > max {
					max = v[i];
					max_index = i;
				}
			}
			(vec![max_index], max)
		}
    }

    pub fn max_element_index(&self) -> (Vec<usize>,T) {
		let result = Self::search_max_element(self.shape().to_vec(), self.buffer());
		result
    }
}

impl<'a, T> SubTensor<'a,T>
where T:Clone{
    pub fn shape(&self) -> &[usize] {
		self.shape.as_slice()
    }
}

impl<'a, T> fmt::Display for SubTensor<'a,T>
where T:fmt::Display + Clone  {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		let shape = self.shape();
		let mut disp = format!("SubTensor [");
		for s in shape {
			disp = format!("{}{},", disp, s);
		}
		disp = format!("{}]\n",disp);
		disp = format!("{}{}",disp,fmt_recursive(0, &shape[0..shape.len()], self.v));
		write!(f, "{}", disp)
    }
}

impl<'a, T> SubTensor<'a, T>
where T:Clone {
    pub fn sub_tensor(&'a self, index:usize) -> SubTensor<'a, T> {
		assert!(self.shape.len() > 0);
		assert!(index < self.shape[0]);
		let down_shape = self.shape[1..self.shape.len()].to_vec();
		let stride = down_shape.iter().fold(1, |prod, x| prod * (*x));
		SubTensor {
			shape: down_shape,
			v: &self.v[(stride*index)..(stride*(index+1))]
		}
    }

    pub fn into_tensor(&'a self) -> Tensor<T> {
		Tensor {
			shape: self.shape().to_vec(),
			v: self.v.to_vec().into_boxed_slice()
		}
    }

    pub fn buffer(&self) -> &[T] {
		&self.v
    }
}

