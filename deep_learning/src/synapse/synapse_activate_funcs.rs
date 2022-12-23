/* -*- tab-width:4 -*- */
use std::cell::RefCell;
use std::rc::Rc;
use linear_transform::tensor::Tensor;
use crate::synapse::{Synapse,SynapseOption,SynapseNode,NNSynapseNode};
use crate::neuron::{NeuronPrimType,NNNeuron,nn_neuron_new,nn_neuron_constant};

impl<T> SynapseNode<T>
where T:NeuronPrimType<T> {

	pub fn sigmod_forward(inputs: Vec<&Tensor<T>>, _opt: &Option<SynapseOption>)
						  -> Vec<Tensor<T>> {
		let one = Tensor::<T>::one(inputs[0].shape());
		vec![Tensor::<T>::hadamard_division(&one,
											&(&one + inputs[0].neg().exp()))]
	}
	pub fn sigmod_backward(inputs: &Vec<NNNeuron<T>>,
						   grads: &Vec<NNNeuron<T>>,
						   _opt: &Option<SynapseOption>)
					-> (Vec<NNSynapseNode<T>>,Vec<NNNeuron<T>>) {
		let mut sns:Vec<NNSynapseNode<T>> = Vec::new();
		let mut outputs:Vec<NNNeuron<T>> = Vec::new();

		if inputs[0].borrow().is_constant() {
			return (sns, outputs)
		}

		let y_shape = inputs[0].borrow().shape().to_vec();
		outputs.push(Rc::clone(&grads[0]));

		let one = nn_neuron_constant("1.0", Tensor::<T>::one(&y_shape));
		outputs.push(Rc::clone(&one));

		let (sn, output) = Self::sub(one, Rc::clone(&inputs[0]));
		sns.push(sn);
		outputs.push(Rc::clone(&output));

		let (sn, output) = Self::hadamard_product(Rc::clone(&inputs[0]),Rc::clone(&output));
		sns.push(sn);
		outputs.push(Rc::clone(&output));

		let (sn, gy) = Self::hadamard_product(Rc::clone(&grads[0]), output);
		sns.push(sn);
		outputs.push(Rc::clone(&gy));

		{
			let mut n = inputs[0].borrow_mut();
			if let Some(ref g) = n.ref_grad(){
				let (sn, output) = Self::add(Rc::clone(&g), Rc::clone(&gy));
				sns.push(sn);
				outputs.push(Rc::clone(&output));
				n.set_grad(output)
			}
			else {
				n.set_grad(gy)
			}
		}

		(sns,outputs)
	}

	pub fn sigmod(x:NNNeuron<T>) -> (NNSynapseNode<T>,NNNeuron<T>) {
		let label = "sigmod";
		let output = nn_neuron_new::<T>(&label, Tensor::<T>::zero(&[1,1]));
		let s = Synapse::<T>::new(Self::sigmod_forward,
								  Self::sigmod_backward);
		let sn = SynapseNode::<T>::new(&label, vec![Rc::clone(&x)], vec![Rc::clone(&output)], s);
		let rsn = Rc::new(RefCell::new(sn));
		output.borrow_mut().set_generator(Rc::clone(&rsn));
		rsn.borrow().forward();
		(rsn, output)
	}
}
