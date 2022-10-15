use std::{ops,fmt};
use std::sync::Mutex;
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

fn element<'a, T>(index:Vec<usize>, shape:&'a [usize], v:&'a [T]) -> &'a T {
    if index.len() > 1 {
	let down_shape = &shape[1..shape.len()];
	let stride = down_shape.iter().fold(1,|prod,x| prod * (*x));
	let i = index[0];
	element((&index[1..index.len()]).to_vec(), &down_shape, &v[(stride*i)..(stride*(i+1))])
    }
    else {
	&v[index[0]]
    }
}

fn element_mut<'a, T>(index:Vec<usize>, shape:&'a [usize], v:&'a mut [T]) -> &'a mut T {
    if index.len() > 1 {
	let down_shape = &shape[1..shape.len()];
	let stride = down_shape.iter().fold(1,|prod,x| prod * (*x));
	let i = index[0];
	element_mut((&index[1..index.len()]).to_vec(), &down_shape, &mut v[(stride*i)..(stride*(i+1))])
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

    pub fn from_array<'a>(shape:&'a [usize], elements:&'a [T]) -> Tensor<T> {
	let product = shape.iter().fold(1,|prod, x| prod * (*x));
	assert_eq!(product, elements.len());
	Tensor {
	    shape: shape.to_vec(),
	    v: elements.to_owned().into_boxed_slice()
	}
    }

    pub fn from_vector<'a>(shape:Vec<usize>, elements:Vec<T>) -> Tensor<T> {
	let product = shape.iter().fold(1,|prod, x| prod * (*x));
	assert_eq!(product, elements.len());
	Tensor {
	    shape: shape,
	    v: elements.into_boxed_slice()
	}
    }

    pub fn set(&mut self, index: Vec<usize>, value:T) -> () {
	assert!(index.len() == self.shape.len());
	let e = element_mut(index,&self.shape, &mut self.v);
	*e = value;
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
	element(index,&self.shape, &self.v)
    }
}

impl<T> ops::IndexMut<Vec<usize>> for Tensor<T>
where T:Clone {
    fn index_mut(&mut self, index:Vec<usize>) -> &mut T {
	assert!(index.len() == self.shape.len());
	element_mut(index,&self.shape, &mut self.v)
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

#[test]
fn tensor_test() {
    let shape:[usize;1] = [1];
    let t = Tensor::<f64>::zero(&shape);
    println!("{}", t);

    let shape:[usize;1] = [3];
    let t = Tensor::<f64>::zero(&shape);
    println!("{}", t);
    let m_init:[f32;12] = [ 11.0,12.0,13.0,14.0,
			    21.0,22.0,23.0,24.0,
			    31.0,32.0,33.0,34.0 ];
    let t = Tensor::<f32>::from_array(&[3,4],&m_init);
    println!("{}", t);
    let m_init:[f32;36] = [ 111.0,112.0,113.0,114.0,
			    121.0,122.0,123.0,124.0,
			    131.0,132.0,133.0,134.0,
			    211.0,212.0,213.0,214.0,
			    221.0,222.0,223.0,224.0,
			    231.0,232.0,233.0,234.0,
			    311.0,312.0,313.0,314.0,
			    321.0,322.0,323.0,324.0,
			    331.0,332.0,333.0,334.0 ];
    let t = Tensor::<f32>::from_array(&[3,3,4],&m_init);
    println!("{}", t);

    let m_init:[f32;72] = [ 1111.0,1112.0,1113.0,1114.0,
			    1121.0,1122.0,1123.0,1124.0,
			    1131.0,1132.0,1133.0,1134.0,
			    1211.0,1212.0,1213.0,1214.0,
			    1221.0,1222.0,1223.0,1224.0,
			    1231.0,1232.0,1233.0,1234.0,
			    1311.0,1312.0,1313.0,1314.0,
			    1321.0,1322.0,1323.0,1324.0,
			    1331.0,1332.0,1333.0,1334.0,
			    2111.0,2112.0,2113.0,2114.0,
			    2121.0,2122.0,2123.0,2124.0,
			    2131.0,2132.0,2133.0,2134.0,
			    2211.0,2212.0,2213.0,2214.0,
			    2221.0,2222.0,2223.0,2224.0,
			    2231.0,2232.0,2233.0,2234.0,
			    2311.0,2312.0,2313.0,2314.0,
			    2321.0,2322.0,2323.0,2324.0,
			    2331.0,2332.0,2333.0,2334.0 ];
    let t = Tensor::<f32>::from_array(&[2,3,3,4],&m_init);
    println!("{}", t);

    println!("{}", t[vec![1,2,2,3]]);

    let st = t.sub_tensor(1);
    println!("{}", st);

    let st = st.sub_tensor(0);
    println!("st 2");
    println!("{}", st);

    let mut t = Tensor::<f32>::from_array(&[2,3,3,4],&m_init);
    assert_eq!(t[vec![0,2,2,3]],1334.0);
    t.set(vec![0,2,2,3], 0.0);
    assert_eq!(t[vec![0,2,2,3]],0.0);
    t[vec![0,2,2,3]] = 1335.0;
    assert_eq!(t[vec![0,2,2,3]],1335.0);
    assert_eq!(t[vec![0,2,2,2]],1333.0);

    let t = Tensor::<f64>::from_array(&[1,5], &[1.0, 0.0, 5.0, 1.0, 2.0]);
    assert_eq!(t.max_element_index(), (vec![0,2],5.0));
    let t = Tensor::<f64>::from_array(&[2, 5], &[1.0, 0.0, 5.0, 1.0, 2.0, 1.0, 0.6, 5.0, 6.0, 2.0]);
    assert_eq!(t.max_element_index(), (vec![1,3],6.0));

    let m_init:[f32;30] = [ 1.0, 0.0, 5.0, 1.0, 2.0,
			    1.0, 0.6, 5.0, 6.0, 2.0,

			    1.0, 0.0, 7.0, 1.0, 2.0,
			    1.0, 0.6, 5.0, 6.0, 2.0,

			    1.0, 0.0, 2.0, 1.0, 2.0,
			    1.0, 0.6, 5.0, 6.0, 2.0 ];
    let t = Tensor::<f32>::from_array(&[3,2,5],&m_init);
    assert_eq!(t.max_element_index(), (vec![1,0,2],7.0));
}
