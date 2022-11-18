/* -*- tab-width:4 -*- */

use std::fmt;
use std::cell::RefCell;
use std::rc::Rc;
use linear_transform::tensor::Tensor;
use crate::synapse::{Synapse,SynapseNode,NNSynapseNode};
use crate::neuron::{NNNeuron,nn_neuron_new,nn_neuron_constant};

impl<T> SynapseNode<T>
where T:num::Float + num::pow::Pow<T, Output = T> + Clone + fmt::Display {

	fn mul_rank0_make_diff_node(inputs: &Vec<NNNeuron<T>>, grads: &Vec<NNNeuron<T>>) -> (Vec<NNSynapseNode<T>>,Vec<NNNeuron<T>>) {
		let mut sns:Vec<NNSynapseNode<T>> = Vec::new();
		let mut outputs:Vec<NNNeuron<T>> = Vec::new();

		if inputs[0].borrow().is_constant() &&
			inputs[1].borrow().is_constant()
		{
			(sns,outputs)
		}
		else
		{
			// !(inputs[0].borrow().is_constant() && inputs[1].borrow().is_constant()) =
			// !inputs[0].borrow().is_constant() || !inputs[1].borrow().is_constant()
			if !inputs[0].borrow().is_constant() {
				let output = if grads[0].borrow().is_constant() && inputs[1].borrow().is_constant() {
					let t = Tensor::<T>::mul_rank0(grads[0].borrow().ref_signal(),inputs[1].borrow().ref_signal());
					let label = "(".to_string() + grads[0].borrow().name() + ") * (" + inputs[1].borrow().name() + ")";
					nn_neuron_constant(&label, t)
				}
				else {
					let (l_sn, l_output) = Self::mul_rank0(Rc::clone(&grads[0]), Rc::clone(&inputs[1]));
					sns.push(l_sn);
					outputs.push(Rc::clone(&inputs[1]));
					outputs.push(Rc::clone(&grads[0]));
					l_output
				};

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
			};

			if !inputs[1].borrow().is_constant() {
				let output = if grads[0].borrow().is_constant() && inputs[0].borrow().is_constant() {
					let t = Tensor::<T>::mul_rank0(grads[0].borrow().ref_signal(),inputs[0].borrow().ref_signal());
					let label = "(".to_string() + grads[0].borrow().name() + ") * (" + inputs[0].borrow().name() + ")";
					nn_neuron_constant(&label, t)
				}
				else {
					let (r_sn, r_output) = Self::mul_rank0(Rc::clone(&grads[0]), Rc::clone(&inputs[0]));
					sns.push(r_sn);
					outputs.push(Rc::clone(&inputs[0]));
					outputs.push(Rc::clone(&grads[0]));
					r_output
				};

				outputs.push(Rc::clone(&output));
				{
					let mut n = inputs[1].borrow_mut();
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
	}

	pub fn mul_rank0(x:NNNeuron<T>, y:NNNeuron<T>) -> (NNSynapseNode<T>,NNNeuron<T>) {
		assert_eq!(x.borrow().shape(),&[1,1]);
		assert_eq!(y.borrow().shape(),&[1,1]);
		let label = "mul_rank0";
		let output = nn_neuron_new(&label, Tensor::<T>::zero(&[1,1]));
		let s = SynapseNode::<T>::new(&label,
									  vec![Rc::clone(&x),Rc::clone(&y)],
									  vec![Rc::clone(&output)],
									  Synapse::<T>::new(
										  |inputs| {
											  vec![Tensor::<T>::mul_rank0(inputs[0], inputs[1])]
										  },
										  Self::mul_rank0_make_diff_node));
		let rs = Rc::new(RefCell::new(s));
		output.borrow_mut().set_generator(Rc::clone(&rs));
		rs.borrow().forward();
		(rs, output)
	}
}
