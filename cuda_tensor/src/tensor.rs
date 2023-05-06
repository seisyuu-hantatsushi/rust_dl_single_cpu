use crate::*;
use cuda_binding::*;
use std::{fmt, mem};
use std::cell::RefCell;
use num;

#[derive(Debug)]
pub struct Tensor<'a, T> where T:num::Num {
    context: &'a Context,
    shape: Vec<usize>,
    device_mem: RefCell<cuda::Memory<T>>,
}

fn fmt_recursive<'a, T>(depth:usize, shape:&'a[usize], v:&'a[T]) -> String
where T:fmt::Display {
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

impl<'a, T> fmt::Display for Tensor<'a, T> where T:fmt::Display+num::Num {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let shape = self.shape();
	let elements:Vec<T> = self.device_mem.borrow_mut().to_host().unwrap();

	let mut disp = format!("CUDA Tensor [");
        for s in shape {
            disp = format!("{}{},", disp, s);
        }
        disp = format!("{}]\n", disp);

	disp = format!("{}{}", disp, fmt_recursive(0, &shape[0..shape.len()], &elements));


	write!(f, "{}", disp)
    }
}

impl<'a, T> Tensor<'a, T> where T:num::Num {
    pub fn new(ctx: &'a Context, shape: &[usize]) -> Result<Tensor<'a, T>, CUDAError> {
        let num_of_elements = shape.iter().fold(1, |p, &d| p * d);
        let device_mem = cuda::Memory::malloc(num_of_elements)?;
        Ok(Tensor {
            context: ctx,
            shape: shape.to_vec(),
            device_mem: RefCell::new(device_mem)
        })
    }


    pub fn as_ref_devmem(&self) -> &RefCell<cuda::Memory<T>> {
	&self.device_mem
    }

    pub fn set_elements(&mut self, elements: Vec<T>) -> Result<(), CUDAError> {
        self.device_mem.borrow_mut().from_host(&elements)
    }

    pub fn get_elements(&self) -> Result<Vec<T>, CUDAError> {
	self.device_mem.borrow_mut().to_host()
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

pub fn add<'a, T:num::Num>(x:&Tensor<'a, T>, y:&Tensor<'a, T>) -> Tensor<'a, T> {
    x.context.add(x, y).unwrap()
}

pub fn sub<'a, T:num::Num>(x:&Tensor<'a, T>, y:&Tensor<'a, T>) -> Tensor<'a, T> {
    x.context.sub(x, y).unwrap()
}

pub fn hadamard_product<'a, T:num::Num>(x:&Tensor<'a, T>, y:&Tensor<'a, T>) -> Tensor<'a, T> {
    x.context.hadamard_product(x, y).unwrap()
}
