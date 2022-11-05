/* -*- tab-width:4 -*- */

use std::fmt;
use std::cell::RefCell;
use std::rc::Rc;
use linear_transform::tensor::tensor_base::Tensor;

use crate::synapse::NNSynapseNode;

pub struct Neuron<T>
where T:num::Float + num::pow::Pow<T, Output = T> + Clone {
	name: String,
	signal: Tensor<T>,
	generator: Option<NNSynapseNode<T>>,
	grad: Option<NNNeuron<T>>
}

impl<T> fmt::Display for Neuron<T>
where T: fmt::Display + Clone + num::Float + num::pow::Pow<T, Output = T> {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		let mut disp = format!("Neuron. name:{}\n",self.name);
		disp = format!("{}Singal:{}", disp, self.signal);
		write!(f,"{}",disp)
	}
}

impl<T> Neuron<T>
where T:num::Float + num::pow::Pow<T, Output = T> + Clone {
	pub fn new(name:&str, init_signal:Tensor<T>) -> Neuron<T> {
		Neuron {
			name: name.to_string(),
			signal: init_signal,
			generator: None,
			grad: None
		}
	}

	pub fn name(&self) -> &str {
		&self.name
	}

	pub fn set_generator(&mut self, s:NNSynapseNode<T>){
		self.generator = Some(Rc::clone(&s))
	}

	pub fn ref_generator(&self) -> &Option<NNSynapseNode<T>> {
		&self.generator
	}

	pub fn assign(&mut self, signal:Tensor<T>) -> () {
		self.signal = signal;
	}

	pub fn ref_signal(&self) -> &Tensor<T> {
		&self.signal
	}

	pub fn ref_grad(&self) -> &Option<NNNeuron<T>> {
		&self.grad
	}

	pub fn set_grad(&mut self, grad:NNNeuron<T>) {
		self.grad = Some(grad)
	}

	pub fn get_grad(&mut self) -> NNNeuron<T> {
		if let Some(ref g) = self.grad {
			Rc::clone(g)
		}
		else {
			let label = "g'".to_string() + self.name();
			let shape = self.signal.shape();
			let new_grad = Rc::new(RefCell::new(Neuron::<T>::new(&label,Tensor::<T>::one(&shape))));
			self.grad = Some(Rc::clone(&new_grad));
			new_grad
		}
	}

	pub fn shape(&self) -> &[usize] {
		self.signal.shape()
	}
}

pub type NNNeuron<T> = Rc<RefCell<Neuron<T>>>;

pub fn nn_neuron_new<T>(name:&str, init_signal:Tensor<T>) -> NNNeuron<T>
where T:num::Float + num::pow::Pow<T, Output = T> + Clone {
	Rc::new(RefCell::new(Neuron::<T>::new(name,init_signal)))
}
