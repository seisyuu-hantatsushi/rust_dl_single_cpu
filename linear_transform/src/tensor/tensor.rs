use std::{ops,fmt};
use std::sync::Mutex;
use std::collections::hash_map::HashMap;
use num;

/*
| a_11, a_12, ... , a_1n |
| a_21, a_22, ... , a_2n |
|                        |
| a_m1, a_m2, ... , a_mn |
 */

#[derive(Debug, Clone)]
enum Element<'a, T> {
    Inst(Box<[T]>),
    Ref(&'a [T])
}

pub struct Tensor<'a, T> where T: Clone {
    shape: Vec<usize>,
    v: Element<'a, T>,
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

impl<T> fmt::Display for Tensor<'_, T>
where T: fmt::Display + Clone {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
	let shape = self.shape();
	let mut disp = format!("Tensor [");
	for s in shape {
	    disp = format!("{}{},", disp, s);
	}
	disp = format!("{}]\n",disp);
	let v:&[T] = match &self.v {
	    Element::Inst(v) => { v },
	    Element::Ref(v)  => { v }
	};
	disp = format!("{}{}",disp,fmt_recursive(0, &shape[0..shape.len()], v));
	write!(f, "{}", disp)
    }
}

impl<T:Clone> Tensor<'_, T> {
    pub fn shape(&self) -> &[usize] {
	self.shape.as_slice()
    }
}

impl<T> Tensor<'_, T>
    where T:num::Num+ops::AddAssign+ops::MulAssign+Clone+Copy {

    pub fn zero(shape:&[usize]) -> Tensor<T> {
	let size = shape.iter().fold(1,|prod,&x| prod*x);
	let mut v:Vec<T> = Vec::with_capacity(size);
	unsafe { v.set_len(size) };
	for i in 0..size {
	    v[i].set_zero();
	}
	Tensor {
	    shape: shape.to_vec(),
	    v: Element::Inst(v.into_boxed_slice()),
	}
    }

    pub fn from_array<'a>(shape:&'a [usize], elements:&'a [T]) -> Tensor<'a, T> {
	let product = shape.iter().fold(1,|prod, x| prod * (*x));
	assert_eq!(product, elements.len());
	Tensor {
	    shape: shape.to_vec(),
	    v: Element::Inst(elements.to_owned().into_boxed_slice()),
	}
    }
}

impl<'b:'a, 'a, T> Tensor<'b, T>
where T:Clone {
    fn sub_tensor(&'b self, index:usize) -> Tensor<'a, T> {
	assert!(self.shape.len() > 0);
	assert!(index < self.shape[0]);
	let down_shape = self.shape[1..self.shape.len()].to_vec();
	let stride = down_shape.iter().fold(1, |prod, x| prod * (*x));
	let v:&[T] = match &self.v {
	    Element::Inst(v) => { v },
	    Element::Ref(v) => { v },
	};
	Tensor{
	    shape: down_shape,
	    v: Element::Ref(&v[(stride*index)..(stride*(index+1))])
	}
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

    let m_init:[f32;72] = [ 111.0,112.0,113.0,114.0,
			    121.0,122.0,123.0,124.0,
			    131.0,132.0,133.0,134.0,
			    211.0,212.0,213.0,214.0,
			    221.0,222.0,223.0,224.0,
			    231.0,232.0,233.0,234.0,
			    311.0,312.0,313.0,314.0,
			    321.0,322.0,323.0,324.0,
			    331.0,332.0,333.0,334.0,
			    111.0,112.0,113.0,114.0,
			    121.0,122.0,123.0,124.0,
			    131.0,132.0,133.0,134.0,
			    211.0,212.0,213.0,214.0,
			    221.0,222.0,223.0,224.0,
			    231.0,232.0,233.0,234.0,
			    311.0,312.0,313.0,314.0,
			    321.0,322.0,323.0,324.0,
			    331.0,332.0,333.0,334.0 ];
    let t = Tensor::<f32>::from_array(&[2,3,3,4],&m_init);
    println!("{}", t);

    let st = t.sub_tensor(1);
    println!("{}", st);
}
