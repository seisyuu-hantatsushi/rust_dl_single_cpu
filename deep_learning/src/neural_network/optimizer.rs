/* -*- tab-width:4 -*- */

use std::fmt;
use std::rc::Rc;
use linear_transform::Tensor;
use crate::neural_network::NNModel;
use crate::neuron::{NeuronPrimType,Neuron,NNNeuron};

pub struct SGD<T>
where T: NeuronPrimType<T> {
	learning_rate: T
}

pub struct MomentumSDG<T>
where T: NeuronPrimType<T> {
	learning_rate: T,
	momentum: T
}

pub enum Optimizer<T>
where T: NeuronPrimType<T> {
    SGD(SGD<T>), MomentumSDG(MomentumSDG<T>)
}

impl<T> SGD<T>
where T: NeuronPrimType<T> {
	pub fn new(learning_rate:T) -> SGD<T> {
		SGD { learning_rate }
	}
}

pub struct NNOptimizer<T>
where T: NeuronPrimType<T> {
	target: NNModel<T>,
	optimizer: Optimizer<T>
}

impl<T> NNOptimizer<T>
where T: NeuronPrimType<T> {

	pub fn new(optimizer:Optimizer<T>, model:&NNModel<T>) -> NNOptimizer<T> {
		NNOptimizer {
			target: Rc::clone(model),
			optimizer: optimizer
		}
	}

	pub fn update(&self) -> Result<(),String> {
		match &self.optimizer {
			Optimizer::SGD(sgd) => {
				for param in self.target.get_params() {
					let feedback = if let Some(ref g) = param.borrow().ref_grad() {
						g.borrow().ref_signal().scale(sgd.learning_rate)
					}
					else {
						return Err("param does not have grad".to_string());
					};
					let update = param.borrow().ref_signal() - feedback;
					//println!("{}", update);
					param.borrow_mut().assign(update);
				}
			},
			Optimizer::MomentumSDG(msgd) => {
			}
		}
		Ok(())
	}

}


