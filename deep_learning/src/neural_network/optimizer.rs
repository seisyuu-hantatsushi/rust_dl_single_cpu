/* -*- tab-width:4 -*- */

use std::fmt;
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashMap;
use linear_transform::Tensor;
use crate::neural_network::NNModel;
use crate::neuron::{NeuronPrimType,Neuron,NNNeuron};

#[derive(Clone)]
pub struct SGD<T>
where T: NeuronPrimType<T> {
	learning_rate: T
}

#[derive(Clone)]
pub struct MomentumSDG<T>
where T: NeuronPrimType<T> {
	learning_rate: T,
	momentum: T
}

#[derive(Clone)]
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

impl<T> MomentumSDG<T>
where T: NeuronPrimType<T> {
	pub fn new(learning_rate:T, momentum:T) -> MomentumSDG<T> {
		MomentumSDG { learning_rate, momentum }
	}
}

struct MomentumSDGContext<T>
where T: NeuronPrimType<T> {
	velocitys: HashMap<* const RefCell<Neuron<T>>, Tensor<T>>
}

enum OptimizerContext<T>
where T: NeuronPrimType<T> {
	MomentumSDG(MomentumSDGContext<T>)
}

pub struct NNOptimizer<T>
where T: NeuronPrimType<T> {
	target: NNModel<T>,
	optimizer: Optimizer<T>,
	optimizer_context: Option<OptimizerContext<T>>
}

impl<T> NNOptimizer<T>
where T: NeuronPrimType<T> {

	pub fn new(optimizer:Optimizer<T>, model:&NNModel<T>) -> NNOptimizer<T> {
		match optimizer {
			Optimizer::SGD(_) => {
				NNOptimizer {
					target: Rc::clone(model),
					optimizer: optimizer,
					optimizer_context: None
				}
			},
			Optimizer::MomentumSDG(_) => {
				NNOptimizer {
					target: Rc::clone(model),
					optimizer: optimizer,
					optimizer_context: Some(
						OptimizerContext::MomentumSDG(
							MomentumSDGContext {
								velocitys: HashMap::new()
							}
						)
					)
				}
			}
		}
	}

	pub fn update(&mut self) -> Result<(),String> {
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
				let mut ctx = if let Some(ref mut ctx) = self.optimizer_context {
					match ctx {
						OptimizerContext::MomentumSDG(ctx) => {
							ctx
						}
					}
				}
				else {
					return Err("optimizer does not context".to_string());
				};

				for param in self.target.get_params() {
					if !ctx.velocitys.contains_key(&Rc::as_ptr(&param)) {
						let v = Tensor::<T>::zero(param.borrow().shape());
						ctx.velocitys.insert(Rc::as_ptr(&param),v);
					}
					if let Some(v) = ctx.velocitys.get(&Rc::as_ptr(&param)) {
						let delta = if let Some(ref g) = param.borrow().ref_grad() {
							let av = v.scale(msgd.momentum);
							av - g.borrow().ref_signal().scale(msgd.learning_rate)
						}
						else {
							return Err("param does not have grad".to_string());
						};
						//println!("{}", update);
						ctx.velocitys.insert(Rc::as_ptr(&param), delta.clone());
						let update = param.borrow().ref_signal() + delta;
						param.borrow_mut().assign(update);
					}
					else {
						panic!("key must be in hashmap\n");
					}
				}
			}
		}
		Ok(())
	}

}


