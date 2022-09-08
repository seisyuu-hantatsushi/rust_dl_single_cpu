use std::{ops,fmt};
use num;

/*
| a_11, a_12, ... , a_1n |
| a_21, a_22, ... , a_2n |
|                        |
| a_m1, a_m2, ... , a_mn |
 */

#[derive(Debug, Clone)]
pub struct Tensor<T> where T: Clone {
    shape: Vec<usize>,
    //stride: usize, //for align
    v: Box<[T]>
}


fn fmt_recursive<'a, T>(dim:usize, shape:&'a[usize], v:&'a[T]) -> Option<String>
where T:std::fmt::Display {
    if shape.len() > 0 {
	Some("test".to_string())
    }
    else {
	let l = v.iter().fold("".to_string(),|s, x| s + &x.to_string()).clone();
	Some(l)
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
	if let Some(s) = fmt_recursive(shape[0], &shape[1..shape.len()], &self.v) {
	    disp = format!("{}{}",disp,s);
	}
	write!(f, "{}", disp)
    }
}

impl<T:Clone> Tensor<T> {
    pub fn shape(&self) -> &[usize] {
	self.shape.as_slice()
    }
}

impl<T> Tensor<T>
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
	    v: v.into_boxed_slice()
	}
    }
}

#[test]
fn tensor_test() {
    let shape:[usize;1] = [3];
    let t = Tensor::<f64>::zero(&shape);
    println!("{}", t);

}
