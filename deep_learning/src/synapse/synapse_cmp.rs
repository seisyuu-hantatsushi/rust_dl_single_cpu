
/* -*- tab-width:4 -*- */
use std::cell::RefCell;
use std::rc::Rc;
use linear_transform::tensor::Tensor;
use crate::synapse::{Synapse,SynapseOption,SynapseNode,NNSynapseNode};
use crate::neuron::{NeuronPrimType,NNNeuron,nn_neuron_new,nn_neuron_constant};

impl<T> SynapseNode<T>
where T:NeuronPrimType<T> {

	fn max_forward(inputs: Vec<&Tensor<T>>, _opt: &Option<SynapseOption<T>>)
				   -> Vec<Tensor<T>> {
		vec![Tensor::max(inputs)]
	}

	fn max_backward(inputs: &Vec<NNNeuron<T>>,
					grads: &Vec<NNNeuron<T>>,
					_opt: &Option<SynapseOption<T>>)
					-> (Vec<NNSynapseNode<T>>,Vec<NNNeuron<T>>){
		let mut sns:Vec<NNSynapseNode<T>> = Vec::new();
		let mut outputs:Vec<NNNeuron<T>> = Vec::new();

		if inputs.iter().fold(true,|b,n| b & n.borrow().is_constant()) {
			return (sns,outputs);
		}

		let inc_inputs:Vec<NNNeuron<T>> = inputs.iter().map(|n| Rc::clone(&n)).collect();
		let (sn,tmax) = Self::max(inc_inputs);
		sns.push(sn);
		for n in inputs.iter() {
			outputs.push(Rc::clone(n));
		}
		outputs.push(Rc::clone(&tmax));
		let (sn,gx) = Self::hadamard_product(tmax,Rc::clone(&grads[0]));
		sns.push(sn);
		outputs.push(Rc::clone(&grads[0]));
		outputs.push(Rc::clone(&gx));

		for input in inputs.iter() {
			if !input.borrow().is_constant() {
				let mut n = input.borrow_mut();
				if let Some(ref g) = n.ref_grad() {
					let (sn, sumg) = Self::add(Rc::clone(&g), Rc::clone(&gx));
					n.set_grad(Rc::clone(&sumg));
					sns.push(sn);
					outputs.push(sumg);
				}
				else {
					n.set_grad(Rc::clone(&gx))
				}
			}
		}

		(sns,outputs)
	}

	pub fn max(inputs:Vec<NNNeuron<T>>) -> (NNSynapseNode<T>, NNNeuron<T>) {
		let label = "max";
		let output = nn_neuron_new::<T>(&label, Tensor::<T>::zero(&[1,1]));
		let s = Synapse::<T>::new(Self::max_forward,
								  Self::max_backward);
		let cloned_inputs = inputs.iter().map(|n| { Rc::clone(n) }).collect();
		let sn = SynapseNode::<T>::new(&label, cloned_inputs, vec![Rc::clone(&output)], s);
		let rsn = Rc::new(RefCell::new(sn));
		output.borrow_mut().set_generator(Rc::clone(&rsn));
		rsn.borrow().forward();
		(rsn, output)
	}

}
