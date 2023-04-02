/* -*- tab-width:4 -*- */
use std::cell::RefCell;
use std::rc::Rc;
use linear_transform::tensor::Tensor;
use crate::synapse::{Synapse,SynapseOption,SynapseNode,NNSynapseNode};
use crate::neuron::{NeuronPrimType,NNNeuron,nn_neuron_new,nn_neuron_constant};

impl<T> SynapseNode<T>
where T:NeuronPrimType<T> {

	fn sigmoid_forward(inputs: Vec<&Tensor<T>>, _opt: &Option<SynapseOption<T>>)
					   -> Vec<Tensor<T>> {
		vec![inputs[0].sigmoid()]
	}

	fn sigmoid_backward(inputs: &Vec<NNNeuron<T>>,
						grads: &Vec<NNNeuron<T>>,
						_opt: &Option<SynapseOption<T>>)
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
		outputs.push(Rc::clone(&inputs[0]));
		let (sn, y) = Self::sigmoid(Rc::clone(&inputs[0]));
		sns.push(sn);
		outputs.push(Rc::clone(&y));
		let (sn, output) = Self::sub(one, Rc::clone(&y));
		sns.push(sn);
		outputs.push(Rc::clone(&output));

		let (sn, output) = Self::hadamard_product(y, output);
		sns.push(sn);
		outputs.push(Rc::clone(&output));

		let (sn, gy) = Self::hadamard_product(Rc::clone(&grads[0]), output);
		sns.push(sn);
		outputs.push(Rc::clone(&gy));

		let mut n = inputs[0].borrow_mut();
		if let Some(ref g) = n.ref_grad(){
			let (sn, output) = Self::add(Rc::clone(&g), Rc::clone(&gy));
			sns.push(sn);
			output.borrow_mut().rename(&format!("{}+", g.borrow().name()));
			outputs.push(Rc::clone(&output));
			n.set_grad(output)
		}
		else {
			gy.borrow_mut().rename(&format!("({})'", n.name()));
			n.set_grad(gy)
		}

		(sns,outputs)
	}

	pub fn sigmoid(x:NNNeuron<T>) -> (NNSynapseNode<T>,NNNeuron<T>) {
		let label = "sigmod";
		let output = nn_neuron_new::<T>(&label, Tensor::<T>::zero(&[1,1]));
		let s = Synapse::<T>::new(Self::sigmoid_forward,
								  Self::sigmoid_backward);
		let sn = SynapseNode::<T>::new(&label, vec![Rc::clone(&x)], vec![Rc::clone(&output)], s);
		let rsn = Rc::new(RefCell::new(sn));
		output.borrow_mut().set_generator(Rc::clone(&rsn));
		rsn.borrow().forward();
		(rsn, output)
	}

	fn relu_forward(inputs: Vec<&Tensor<T>>, _opt: &Option<SynapseOption<T>>)
					   -> Vec<Tensor<T>> {
		vec![inputs[0].relu()]
	}

	fn relu_backward(inputs: &Vec<NNNeuron<T>>,
					 grads: &Vec<NNNeuron<T>>,
					 _opt: &Option<SynapseOption<T>>)
					 -> (Vec<NNSynapseNode<T>>,Vec<NNNeuron<T>>) {
		let mut sns:Vec<NNSynapseNode<T>> = Vec::new();
		let mut outputs:Vec<NNNeuron<T>> = Vec::new();
		let zero = nn_neuron_constant("0.0", Tensor::<T>::zero(&inputs[0].borrow().shape()));

		if inputs[0].borrow().is_constant() {
			return (sns, outputs);
		}

		outputs.push(Rc::clone(&zero));
		outputs.push(Rc::clone(&inputs[0]));
		let (sn, mask) = Self::max(vec![Rc::clone(&inputs[0]),zero]);
		sns.push(sn);
		outputs.push(Rc::clone(&mask));
		outputs.push(Rc::clone(&grads[0]));
		let (sn, gx) = Self::hadamard_product(Rc::clone(&grads[0]), mask);
		sns.push(sn);
		outputs.push(Rc::clone(&gx));
		let mut n = inputs[0].borrow_mut();
		if let Some(ref g) = n.ref_grad() {
			let (sn, output) = Self::add(Rc::clone(&g), Rc::clone(&gx));
			sns.push(sn);
			output.borrow_mut().rename(&format!("{}+", g.borrow().name()));
			outputs.push(Rc::clone(&output));
			n.set_grad(output)
		}
		else {
			n.set_grad(gx);
		}

		(sns,outputs)
	}

	pub fn relu(x:NNNeuron<T>) -> (NNSynapseNode<T>,NNNeuron<T>) {
		let label = "relu";
		let output = nn_neuron_new::<T>(&label, Tensor::<T>::zero(&[1,1]));
		let s = Synapse::<T>::new(Self::relu_forward,
								  Self::relu_backward);
		let sn = SynapseNode::<T>::new(&label, vec![Rc::clone(&x)], vec![Rc::clone(&output)], s);
		let rsn = Rc::new(RefCell::new(sn));
		output.borrow_mut().set_generator(Rc::clone(&rsn));
		rsn.borrow().forward();
		(rsn, output)
	}

}
