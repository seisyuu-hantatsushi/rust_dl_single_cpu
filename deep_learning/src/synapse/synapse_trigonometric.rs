/* -*- tab-width:4 -*- */
use std::fmt;
use std::cell::RefCell;
use std::rc::Rc;
use linear_transform::tensor::Tensor;
use crate::synapse::{Synapse,SynapseNode,NNSynapseNode};
use crate::neuron::{NNNeuron,nn_neuron_new};

impl<T> SynapseNode<T>
where T:num::Float + num::FromPrimitive + num::pow::Pow<T, Output = T> + Clone + fmt::Display {
	pub fn sin(x:NNNeuron<T>) -> (NNSynapseNode<T>,NNNeuron<T>) {
		let label = "sin";
		let output = nn_neuron_new::<T>(&label, Tensor::<T>::zero(&[1,1]));

		let s = SynapseNode::<T>::new(&label,
									  vec![Rc::clone(&x)],
									  vec![Rc::clone(&output)],
									  Synapse::<T>::new(
										  |inputs,_opt| {
											  vec![inputs[0].sin()]
										  },
										  |inputs,grads,_opt| {
											  let mut sns:Vec<NNSynapseNode<T>> = Vec::new();
											  let mut outputs:Vec<NNNeuron<T>> = Vec::new();
											  if !inputs[0].borrow().is_constant() {
												  let (sn,output) = Self::cos(Rc::clone(&inputs[0]));
												  sns.push(sn);
												  outputs.push(Rc::clone(&output));
												  let (sn,output) = Self::hadamard_product(Rc::clone(&grads[0]), output);
												  sns.push(sn);
												  outputs.push(Rc::clone(&output));
												  {
													  let mut n = inputs[0].borrow_mut();
													  if let Some(ref g) = n.ref_grad() {
														  outputs.push(Rc::clone(&g));
														  let (sn, output) = Self::add(Rc::clone(&g), output);
														  n.set_grad(Rc::clone(&output));
														  sns.push(sn);
														  outputs.push(output);
													  }
													  else {
														  n.set_grad(output);
													  }
												  }
											  }
											  (sns,outputs)
										  }
									  ));
		let rs = Rc::new(RefCell::new(s));
		output.borrow_mut().set_generator(Rc::clone(&rs));
		rs.borrow().forward();
		(rs, output)
	}

	pub fn cos(x:NNNeuron<T>) -> (NNSynapseNode<T>,NNNeuron<T>) {
		let label = "con";
		let output = nn_neuron_new::<T>(&label, Tensor::<T>::zero(&[1,1]));

		let s = SynapseNode::<T>::new(&label,
									  vec![Rc::clone(&x)],
									  vec![Rc::clone(&output)],
									  Synapse::<T>::new(
										  |inputs,_opt| {
											  vec![inputs[0].cos()]
										  },
										  |inputs,grads,_opt| {
											  let mut sns:Vec<NNSynapseNode<T>> = Vec::new();
											  let mut outputs:Vec<NNNeuron<T>> = Vec::new();
											  if !inputs[0].borrow().is_constant() {
												  let (sn,output) = Self::sin(Rc::clone(&inputs[0]));
												  sns.push(sn);
												  outputs.push(Rc::clone(&output));
												  let (sn,output) = Self::neg(output);
												  sns.push(sn);
												  outputs.push(Rc::clone(&output));
												  let (sn,output) = Self::hadamard_product(Rc::clone(&grads[0]), output);
												  sns.push(sn);
												  outputs.push(Rc::clone(&output));
												  {
													  let mut n = inputs[0].borrow_mut();
													  if let Some(ref g) = n.ref_grad() {
														  outputs.push(Rc::clone(&g));
														  let (sn, output) = Self::add(Rc::clone(&g), output);
														  n.set_grad(Rc::clone(&output));
														  sns.push(sn);
														  outputs.push(output);
													  }
													  else {
														  n.set_grad(output);
													  }
												  }
											  }
											  (sns,outputs)
										  }
									  ));
		let rs = Rc::new(RefCell::new(s));
		output.borrow_mut().set_generator(Rc::clone(&rs));
		rs.borrow().forward();
		(rs, output)
	}
}
