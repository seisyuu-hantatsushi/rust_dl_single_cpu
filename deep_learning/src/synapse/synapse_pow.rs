/* -*- tab-width:4 -*- */

use std::fmt;
use std::cell::RefCell;
use std::rc::Rc;
use linear_transform::tensor::Tensor;
use crate::synapse::{Synapse,SynapseOption,SynapseNode,NNSynapseNode};
use crate::neuron::{NNNeuron, nn_neuron_new, nn_neuron_constant};

impl<T> SynapseNode<T>
where T:num::Float + num::pow::Pow<T, Output = T> + Clone + fmt::Display {

	fn pow_rank0_forward(inputs: Vec<&Tensor<T>>, _opt: &Option<SynapseOption>)
						 -> Vec<Tensor<T>> {
		vec![inputs[0].pow_rank0(inputs[1][vec![0,0]])]
	}

	fn pow_rank0_backward(inputs: &Vec<NNNeuron<T>>,
						  grads: &Vec<NNNeuron<T>>,
						  _opt: &Option<SynapseOption>)
						  -> (Vec<NNSynapseNode<T>>,Vec<NNNeuron<T>>){
		let mut sns:Vec<NNSynapseNode<T>> = Vec::new();
		let mut outputs:Vec<NNNeuron<T>> = Vec::new();
		if !inputs[0].borrow().is_constant() {
			let dec_index = if inputs[1].borrow().is_constant() {
				let t = inputs[1].borrow().ref_signal() - Tensor::<T>::one(&[1,1]);
				let label = inputs[1].borrow().name().to_string() + "-1.0";
				nn_neuron_constant(&label, t)
			}
			else {
				let one = nn_neuron_constant("1,0", Tensor::<T>::one(&[1,1]));
				outputs.push(Rc::clone(&one));
				let (dec_index_sn, dec_index) = Self::sub(Rc::clone(&inputs[1]),one);
				sns.push(dec_index_sn);
				dec_index
			};
			outputs.push(Rc::clone(&dec_index));
			let (l_sn,l_output) = Self::pow_rank0(Rc::clone(&inputs[0]), dec_index);
			outputs.push(Rc::clone(&inputs[0]));
			sns.push(l_sn);
			outputs.push(Rc::clone(&l_output));
			let (l_sn,l_output) = Self::mul_rank0(Rc::clone(&inputs[1]), l_output);
			outputs.push(Rc::clone(&inputs[1]));
			sns.push(l_sn);
			outputs.push(Rc::clone(&l_output));
			let (l_sn,l_output) = Self::mul_rank0(Rc::clone(&grads[0]),  l_output);
			sns.push(l_sn);
			outputs.push(Rc::clone(&l_output));
			outputs.push(Rc::clone(&grads[0]));
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
			let (r_sn,r_output) = Self::pow_rank0(Rc::clone(&inputs[0]),Rc::clone(&inputs[1]));
			sns.push(r_sn);
			outputs.push(Rc::clone(&r_output));
			let (baselog_sn, baselog) = Self::ln_rank0(Rc::clone(&inputs[0]));
			sns.push(baselog_sn);
			outputs.push(Rc::clone(&baselog));
			let (r_sn,r_output) = Self::mul_rank0(baselog,r_output);
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

	pub fn pow_rank0(a:NNNeuron<T>, x:NNNeuron<T>) -> (NNSynapseNode<T>,NNNeuron<T>) {
		assert_eq!(x.borrow().shape(),&[1,1]);
		assert_eq!(a.borrow().shape(),&[1,1]);
		let label = "pow_rank0";
		let output = nn_neuron_new(&label, Tensor::<T>::zero(&[1,1]));
		let s = SynapseNode::<T>::new(&label,
									  vec![Rc::clone(&a),Rc::clone(&x)],
									  vec![Rc::clone(&output)],
									  Synapse::<T>::new(Self::pow_rank0_forward, Self::pow_rank0_backward));
		let rs = Rc::new(RefCell::new(s));
		output.borrow_mut().set_generator(Rc::clone(&rs));
		rs.borrow().forward();
		(rs, output)
	}

	fn pow_forward(inputs: Vec<&Tensor<T>>, _opt: &Option<SynapseOption>)
				   -> Vec<Tensor<T>> {
		vec![inputs[0].pow(inputs[1][vec![0,0]])]
	}

	fn pow_backward(inputs: &Vec<NNNeuron<T>>,
					grads: &Vec<NNNeuron<T>>,
					_opt: &Option<SynapseOption>)
					-> (Vec<NNSynapseNode<T>>,Vec<NNNeuron<T>>){
		let mut sns:Vec<NNSynapseNode<T>> = Vec::new();
		let mut outputs:Vec<NNNeuron<T>> = Vec::new();
		if !inputs[0].borrow().is_constant() {
			let dec_index = if inputs[1].borrow().is_constant() {
				let t = inputs[1].borrow().ref_signal() - Tensor::<T>::one(&[1,1]);
				let label = inputs[1].borrow().name().to_string() + "-1.0";
				nn_neuron_constant(&label, t)
			}
			else {
				let one = nn_neuron_constant("1,0", Tensor::<T>::one(&[1,1]));
				outputs.push(Rc::clone(&one));
				let (dec_index_sn, dec_index) = Self::sub(Rc::clone(&inputs[1]),one);
				sns.push(dec_index_sn);
				dec_index
			};
			outputs.push(Rc::clone(&dec_index));
			let (l_sn,l_output) = Self::pow(Rc::clone(&inputs[0]), dec_index);
			outputs.push(Rc::clone(&inputs[0]));
			sns.push(l_sn);
			outputs.push(Rc::clone(&l_output));
			let (l_sn,l_output) = Self::hadamard_product(Rc::clone(&inputs[1]), l_output);
			outputs.push(Rc::clone(&inputs[1]));
			sns.push(l_sn);
			outputs.push(Rc::clone(&l_output));
			let (l_sn,l_output) = Self::hadamard_product(Rc::clone(&grads[0]),  l_output);
			sns.push(l_sn);
			outputs.push(Rc::clone(&l_output));
			outputs.push(Rc::clone(&grads[0]));
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
			let (r_sn,r_output) = Self::pow(Rc::clone(&inputs[0]),Rc::clone(&inputs[1]));
			sns.push(r_sn);
			outputs.push(Rc::clone(&r_output));
			let (baselog_sn, baselog) = Self::ln_rank0(Rc::clone(&inputs[0]));
			sns.push(baselog_sn);
			outputs.push(Rc::clone(&baselog));
			let (r_sn,r_output) = Self::hadamard_product(baselog,r_output);
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

	pub fn pow(a:NNNeuron<T>, x:NNNeuron<T>) -> (NNSynapseNode<T>,NNNeuron<T>) {
		assert_eq!(x.borrow().shape(),&[1,1]);
		let label = "pow";
		let output = nn_neuron_new(&label, Tensor::<T>::zero(&[1,1]));
		let s = SynapseNode::<T>::new(&label,
									  vec![Rc::clone(&a),Rc::clone(&x)],
									  vec![Rc::clone(&output)],
									  Synapse::<T>::new(Self::pow_forward, Self::pow_backward));
		let rs = Rc::new(RefCell::new(s));
		output.borrow_mut().set_generator(Rc::clone(&rs));
		rs.borrow().forward();
		(rs, output)

	}
	
}
