
use std::fmt;
use std::cell::RefCell;
use std::rc::Rc;
use linear_transform::tensor::tensor_base::Tensor;

use openssl::sha::sha512;

#[derive(Debug, Clone)]
pub struct Neuron<T>
where T:num::Float + Clone {
    hashed_id: [u8;64],
    name: String,
    signal: Rc<RefCell<Tensor<T>>>,
    grad: Option<Rc<RefCell<Tensor<T>>>>
}

impl<T> fmt::Display for Neuron<T>
where T: fmt::Display + Clone + num::Float {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
	let mut disp = format!("Neuron. name:{}\n",self.name);
	disp = format!("{}Singal:{}", disp, self.signal.borrow());
	if let Some(ref g) = self.grad {
	    disp = format!("{}Grad:{}", disp, g.borrow());
	}
	else {
	    disp = format!("{}grad is None", disp);
	}
	write!(f,"{}",disp)
    }
}

impl<T> Neuron<T>
where T:num::Float + num::pow::Pow<T, Output = T> + Clone {

    pub fn create(name:&str, init_signal:Tensor<T>) -> Neuron<T> {
	Neuron {
	    hashed_id: sha512(name.as_bytes()),
	    name: name.to_string(),
	    signal: Rc::new(RefCell::new(init_signal)),
	    grad: None,
	}
    }

    pub fn get_signal(&self) -> Rc<RefCell<Tensor<T>>> {
	Rc::clone(&self.signal)
    }

    pub fn name(&self) -> &str {
	&self.name
    }

    pub fn id(&self) -> &[u8] {
	&self.hashed_id
    }

    pub fn assign(&mut self, signal:Tensor<T>) -> () {
	(*self.signal.borrow_mut()) = signal;
    }

    pub fn set_grad(&mut self, grad:Rc<RefCell<Tensor<T>>>) -> () {
	self.grad = Some(grad);
    }

    pub fn clear_grad(&mut self) -> () {
	self.grad = None;
    }

    pub fn element(&self, index:Vec<usize>) -> T {
	let v = self.signal.borrow()[index];
	v.clone()
    }

    pub fn grad(&self) -> Option<Tensor<T>> {
	if let Some(ref g) = self.grad {
	    Some(g.borrow().clone())
	}
	else {
	    None
	}
    }
}
