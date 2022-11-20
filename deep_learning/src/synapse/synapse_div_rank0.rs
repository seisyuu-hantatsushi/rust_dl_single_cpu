/* -*- tab-width:4 -*- */

use std::fmt;
use std::cell::RefCell;
use std::rc::Rc;
use linear_transform::tensor::Tensor;
use crate::synapse::{Synapse,SynapseOption,SynapseNode,NNSynapseNode};
use crate::neuron::{NNNeuron,nn_neuron_new,nn_neuron_constant};

impl<T> SynapseNode<T>
where T:num::Float + num::pow::Pow<T, Output = T> + Clone + fmt::Display {
	pub fn div_rank0(x:NNNeuron<T>, y:NNNeuron<T>) -> (NNSynapseNode<T>,NNNeuron<T>) {
		assert_eq!(x.borrow().shape(),&[1,1]);
		assert_eq!(y.borrow().shape(),&[1,1]);
		let label = "div_rank0";
		let output = nn_neuron_new(&label, Tensor::<T>::zero(&[1,1]));
		let s = SynapseNode::<T>::new(&label,
									  vec![Rc::clone(&x),Rc::clone(&y)],
									  vec![Rc::clone(&output)],
									  Synapse::<T>::new(
										  |inputs,_opt| {
											  vec![Tensor::<T>::div_rank0(inputs[0], inputs[1])]
										  },
										  |inputs, grads, _opt| {
										  let mut sns:Vec<NNSynapseNode<T>> = Vec::new();
											  let mut outputs:Vec<NNNeuron<T>> = Vec::new();
											  if !inputs[0].borrow().is_constant() {
												  let (l_sn, l_output) = Self::div_rank0(Rc::clone(&grads[0]), Rc::clone(&inputs[1]));
												  outputs.push(Rc::clone(&grads[0]));
												  outputs.push(Rc::clone(&inputs[1]));
												  sns.push(l_sn);
												  outputs.push(Rc::clone(&l_output));
												  {
													  let mut n = inputs[0].borrow_mut();
													  if let Some(ref g) = n.ref_grad() {
														  outputs.push(Rc::clone(&g));
														  let (sn, output) = Self::add(Rc::clone(&g), l_output);
														  n.set_grad(Rc::clone(&output));
														  sns.push(sn);
														  outputs.push(output);
													  }
													  else {
														  n.set_grad(l_output);
													  }
												  }
											  }

											  if !inputs[1].borrow().is_constant() {
												  let two = nn_neuron_constant("2.0", (Tensor::<T>::one(&[1,1]))+Tensor::<T>::one(&[1,1]));
												  outputs.push(Rc::clone(&two));
												  let (neg, neg_input0) = Self::neg(Rc::clone(&inputs[0]));
												  outputs.push(Rc::clone(&inputs[0]));
												  sns.push(neg);
												  outputs.push(Rc::clone(&neg_input0));
												  let (r_sn, r_output) = Self::pow_rank0(Rc::clone(&inputs[1]),two);
												  sns.push(r_sn);
												  outputs.push(Rc::clone(&r_output));
												  let (r_sn, r_output) = Self::div_rank0(neg_input0, r_output);
												  sns.push(r_sn);
												  outputs.push(Rc::clone(&r_output));
												  let (r_sn, r_output) = Self::mul_rank0(Rc::clone(&grads[0]), r_output);
												  sns.push(r_sn);
												  outputs.push(Rc::clone(&r_output));
												  {
													  let mut n = inputs[1].borrow_mut();
													  if let Some(ref g) = n.ref_grad() {
														  outputs.push(Rc::clone(&g));
														  let (sn, output) = Self::add(Rc::clone(&g), r_output);
														  n.set_grad(Rc::clone(&output));
														  sns.push(sn);
														  outputs.push(output);
													  }
													  else {
														  n.set_grad(r_output);
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
